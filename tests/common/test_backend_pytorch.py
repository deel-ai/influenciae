# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the PyTorch backend implementation.
These tests verify the PyTorch-specific functionality works correctly.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


pytestmark = pytest.mark.pytorch


def _per_sample_mse_loss(pred, target):
    """Mean MSE per sample."""
    return nn.functional.mse_loss(pred, target, reduction='none').mean(dim=-1)


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    return nn.Sequential(
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, 2)
    )


@pytest.fixture
def backend():
    """Get the PyTorch backend."""
    from deel.influenciae.common.backend_pytorch import PyTorchBackend
    return PyTorchBackend()


@pytest.fixture(autouse=True)
def _set_test_seed():
    """Keep stochastic tests deterministic and reproducible."""
    np.random.seed(1234)
    torch.manual_seed(1234)


def test_framework_property(backend):
    """Test that framework property returns PYTORCH."""
    from deel.influenciae.common import Framework
    assert backend.framework == Framework.PYTORCH

def test_get_model_weights(backend, simple_model):
    """Test getting model weights."""
    weights = backend.get_model_weights(simple_model)

    # Should have 4 weight tensors: 2 weights + 2 biases
    assert len(weights) == 4

    # Check shapes
    assert weights[0].shape == (3, 5)  # First linear weight
    assert weights[1].shape == (3,)    # First linear bias
    assert weights[2].shape == (2, 3)  # Second linear weight
    assert weights[3].shape == (2,)    # Second linear bias

    for weight in weights:
        assert weight.dtype == torch.float32
        assert torch.all(torch.isfinite(weight))


def test_normalize_weights_to_watch(backend, simple_model):
    """normalize_weights_to_watch should keep watched parameters unchanged."""
    weights = backend.get_model_weights(simple_model)

    normalized = backend.normalize_weights_to_watch(weights)
    normalized_twice = backend.normalize_weights_to_watch(normalized)

    assert len(normalized) == len(weights)
    assert len(normalized_twice) == len(weights)

    for original, first_pass, second_pass in zip(weights, normalized, normalized_twice):
        assert first_pass is original
        assert second_pass is original

def test_get_model_weights_specific_layers(backend, simple_model):
    """Test getting weights from specific layers."""
    children = list(simple_model.children())
    linear_layers = [children[0], children[2]]  # Skip ReLU

    weights = backend.get_model_weights(simple_model, linear_layers)
    assert len(weights) == 4

def test_get_num_params(backend, simple_model):
    """Test parameter counting."""
    weights = backend.get_model_weights(simple_model)
    num_params = backend.get_num_params(weights)

    # (5*3 + 3) + (3*2 + 2) = 18 + 8 = 26
    expected = 5 * 3 + 3 + 3 * 2 + 2
    assert num_params == expected

def test_forward(backend, simple_model):
    """Test forward pass."""
    inputs = torch.randn(4, 5)
    output = backend.forward(simple_model, inputs)

    assert output.shape == (4, 2)
    assert torch.all(torch.isfinite(output))

def test_get_layers(backend, simple_model):
    """Test getting all layers."""
    layers = backend.get_layers(simple_model)

    # children() returns direct child modules
    # Sequential(Linear, ReLU, Linear) = 3 children
    assert len(layers) == 3

def test_get_children(backend, simple_model):
    """Test getting direct children."""
    children = backend.get_children(simple_model)

    assert len(children) == 3
    assert isinstance(children[0], nn.Linear)
    assert isinstance(children[1], nn.ReLU)
    assert isinstance(children[2], nn.Linear)


def test_compute_loss(backend, simple_model):
    """Test loss computation."""
    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    loss_fn = nn.MSELoss(reduction='none')

    loss = backend.compute_loss(simple_model, loss_fn, inputs, targets)

    # MSE without reduction returns (batch, output_dim)
    assert loss.shape == (4, 2)
    assert torch.all(torch.isfinite(loss))
    assert torch.all(loss >= 0)

def test_compute_loss_with_sample_weight(backend, simple_model):
    """Test loss computation with sample weights."""
    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    sample_weight = torch.tensor([1.0, 2.0, 0.5, 1.5])
    loss_fn = nn.MSELoss(reduction='none')

    loss_unweighted = backend.compute_loss(simple_model, loss_fn, inputs, targets)
    loss_weighted = backend.compute_loss(simple_model, loss_fn, inputs, targets, sample_weight.unsqueeze(-1))

    # Weighted loss should be different
    assert not torch.allclose(loss_unweighted, loss_weighted)

def test_compute_gradient(backend, simple_model):
    """Test gradient computation."""
    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    weights = backend.get_model_weights(simple_model)

    gradient = backend.compute_gradient(simple_model, weights, _per_sample_mse_loss, inputs, targets)

    num_params = backend.get_num_params(weights)
    assert gradient.shape == (num_params,)
    assert torch.all(torch.isfinite(gradient))
    assert torch.linalg.norm(gradient) > 0  # Gradient should be non-zero

def test_compute_jacobian(backend, simple_model):
    """Test Jacobian computation."""
    batch_size = 3
    inputs = torch.randn(batch_size, 5)
    targets = torch.randn(batch_size, 2)
    weights = backend.get_model_weights(simple_model)

    jacobian = backend.compute_jacobian(simple_model, weights, _per_sample_mse_loss, inputs, targets)

    num_params = backend.get_num_params(weights)
    assert jacobian.shape == (batch_size, num_params)
    assert torch.all(torch.isfinite(jacobian))
    assert torch.linalg.norm(jacobian) > 0


def test_compute_jacobian_vmap_matches_loop(backend, simple_model):
    """torch.func Jacobian path should match the loop reference."""
    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    weights = backend.get_model_weights(simple_model)

    jacobian_loop = backend._compute_jacobian_loop(
        simple_model, weights, _per_sample_mse_loss, inputs, targets
    )
    jacobian_vmap = backend._compute_jacobian_vmap(
        simple_model, weights, _per_sample_mse_loss, inputs, targets
    )

    assert torch.allclose(jacobian_vmap, jacobian_loop)


def test_compute_jacobian_returns_detached_tensor(backend, simple_model):
    """Public Jacobian API should return a detached tensor."""
    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    weights = backend.get_model_weights(simple_model)

    jacobian = backend.compute_jacobian(simple_model, weights, _per_sample_mse_loss, inputs, targets)

    assert not jacobian.requires_grad
    assert jacobian.grad_fn is None


def test_compute_jacobian_vmap_matches_loop_with_sample_weight(backend, simple_model):
    """Weighted torch.func Jacobian path should match the loop reference."""
    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    sample_weight = torch.tensor([1.0, 2.0, 0.5, 1.5]).unsqueeze(-1)
    weights = backend.get_model_weights(simple_model)

    jacobian_loop = backend._compute_jacobian_loop(
        simple_model, weights, _per_sample_mse_loss, inputs, targets, sample_weight
    )
    jacobian_vmap = backend._compute_jacobian_vmap(
        simple_model, weights, _per_sample_mse_loss, inputs, targets, sample_weight
    )

    assert torch.allclose(jacobian_vmap, jacobian_loop)


def test_compute_jacobian_vmap_matches_loop_for_weight_subset(backend, simple_model):
    """torch.func Jacobian path should support watched weight subsets."""
    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    weights = backend.get_weights_for_layer_range(simple_model, start_layer=None)

    jacobian_loop = backend._compute_jacobian_loop(
        simple_model, weights, _per_sample_mse_loss, inputs, targets
    )
    jacobian_vmap = backend._compute_jacobian_vmap(
        simple_model, weights, _per_sample_mse_loss, inputs, targets
    )

    assert torch.allclose(jacobian_vmap, jacobian_loop)


def test_compute_jacobian_falls_back_to_loop_when_vmap_fails(backend, simple_model, monkeypatch):
    """Public Jacobian API should fall back to the loop path when torch.func fails."""
    inputs = torch.randn(3, 5)
    targets = torch.randn(3, 2)
    weights = backend.get_model_weights(simple_model)
    expected = torch.full((3, backend.get_num_params(weights)), 7.0)

    def fail_vmap(*args, **kwargs):
        raise RuntimeError("torch.func transform failure")

    monkeypatch.setattr(backend, "_compute_jacobian_vmap", fail_vmap)
    monkeypatch.setattr(backend, "_compute_jacobian_loop", lambda *args, **kwargs: expected)

    jacobian = backend.compute_jacobian(simple_model, weights, _per_sample_mse_loss, inputs, targets)

    assert torch.equal(jacobian, expected)


def test_compute_jacobian_uses_loop_for_unmapped_weights(backend, simple_model, monkeypatch):
    """Public Jacobian API should skip torch.func for weights outside the model."""
    inputs = torch.randn(3, 5)
    targets = torch.randn(3, 2)
    rogue_weight = nn.Parameter(torch.ones((2, 2)))
    expected = torch.full((3, rogue_weight.numel()), 5.0)

    def fail_vmap(*args, **kwargs):
        raise AssertionError("vmap path should not be used for unmapped weights")

    monkeypatch.setattr(backend, "_compute_jacobian_vmap", fail_vmap)
    monkeypatch.setattr(backend, "_compute_jacobian_loop", lambda *args, **kwargs: expected)

    jacobian = backend.compute_jacobian(
        simple_model,
        [rogue_weight],
        _per_sample_mse_loss,
        inputs,
        targets,
    )

    assert torch.equal(jacobian, expected)


def test_concat(backend):
    """Test tensor concatenation."""
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    result = backend.concat([a, b], axis=0)
    expected = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    assert torch.equal(result, expected)

def test_stack(backend):
    """Test tensor stacking."""
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    result = backend.stack([a, b], axis=0)
    expected = torch.tensor([[1, 2, 3], [4, 5, 6]])

    assert torch.equal(result, expected)

def test_reshape(backend):
    """Test tensor reshape."""
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])

    result = backend.reshape(a, (3, 2))
    assert result.shape == (3, 2)
    assert torch.equal(result.reshape(-1), a.reshape(-1))

def test_reduce_sum(backend):
    """Test reduce sum."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result_full = backend.reduce_sum(a)
    assert result_full.item() == 10.0

    result_axis0 = backend.reduce_sum(a, axis=0)
    assert torch.equal(result_axis0, torch.tensor([4.0, 6.0]))

    result_axis1 = backend.reduce_sum(a, axis=1)
    assert torch.equal(result_axis1, torch.tensor([3.0, 7.0]))

def test_to_numpy(backend):
    """Test numpy conversion."""
    a = torch.tensor([1.0, 2.0, 3.0])

    result = backend.to_numpy(a)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_to_numpy_fallback_without_torch_numpy_bridge(backend):
    """to_numpy should fallback via tolist when torch numpy bridge is unavailable."""

    class FakeTensor:
        """Minimal tensor-like object to trigger the fallback path."""

        def __init__(self, values):
            self._values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            raise RuntimeError("Numpy is not available")

        def tolist(self):
            return self._values

    result = backend.to_numpy(FakeTensor([1.0, 2.0, 3.0]))
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

def test_get_batch_size(backend):
    """Test batch size extraction."""
    a = torch.randn(8, 5, 3)

    assert backend.get_batch_size(a) == 8

def test_abs(backend):
    """Test absolute value."""
    a = torch.tensor([-1.0, 2.0, -3.0])
    result = backend.abs(a)
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(result, expected)

def test_argmax(backend):
    """Test argmax."""
    a = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 2.0]])
    result = backend.argmax(a, axis=1)
    expected = torch.tensor([1, 0])
    assert torch.equal(result, expected)

def test_gather_along_axis(backend):
    """Test gather along axis."""
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = torch.tensor([2, 0])
    result = backend.gather_along_axis(a, indices, axis=1, batch_dims=1)
    expected = torch.tensor([3.0, 4.0])
    assert torch.equal(result, expected)

def test_get_output_shape(backend, simple_model):
    """Test getting model output shape."""
    out_shape = backend.get_output_shape(simple_model)
    assert out_shape == (None, 2)

def test_split_model(backend, simple_model):
    """Test splitting model into two parts."""
    feature_extractor, head = backend.split_model(simple_model, -1)

    inputs = torch.randn(4, 5)

    # Feature extractor output should be (batch, 3) from the hidden layer
    fe_output = feature_extractor(inputs)
    assert fe_output.shape == (4, 3)

    # Head output should be (batch, 2) from the output layer
    head_output = head(fe_output)
    assert head_output.shape == (4, 2)

    original_output = simple_model(inputs)
    assert torch.allclose(head_output, original_output)


def test_find_last_weight_layer(backend, simple_model):
    """Test finding last weight layer."""
    idx = backend.find_last_weight_layer(simple_model)

    # Last Linear layer should be at index -1
    assert idx == -1

def test_find_last_weight_layer_with_trailing_layers(backend):
    """Test finding last weight layer with non-weight layers at the end."""
    model = nn.Sequential(
        nn.Linear(5, 3),
        nn.Linear(3, 2),
        nn.Softmax(dim=-1)
    )

    idx = backend.find_last_weight_layer(model)

    # Second Linear is at index -2 (Softmax has no weights)
    assert idx == -2

def test_get_layer_index_by_int(backend, simple_model):
    """Test layer index lookup by integer."""
    assert backend.get_layer_index(simple_model, 0) == 0
    assert backend.get_layer_index(simple_model, 1) == 1
    assert backend.get_layer_index(simple_model, -1) == 2


def test_find_layer_by_name(backend, simple_model):
    """Test finding a layer by its named_children key."""
    idx, layer = backend.find_layer_by_name(simple_model, '0')

    assert isinstance(layer, nn.Linear)
    assert layer is simple_model[0]
    assert backend.get_layer_index(simple_model, '0') == idx


def test_find_layer_by_name_not_found(backend, simple_model):
    """Test finding non-existent layer raises error."""
    with pytest.raises(ValueError):
        backend.find_layer_by_name(simple_model, 'nonexistent')

def test_get_weights_for_layer_range_single_layer(backend, simple_model):
    """Test getting weights for a single layer."""
    weights = backend.get_weights_for_layer_range(simple_model, start_layer=0)

    # First Linear layer: weight + bias
    assert len(weights) == 2
    assert weights[0].shape == (3, 5)
    assert weights[1].shape == (3,)
    assert torch.all(torch.isfinite(weights[0]))
    assert torch.all(torch.isfinite(weights[1]))

def test_get_weights_for_layer_range_multiple_layers(backend, simple_model):
    """Test getting weights for multiple layers."""
    weights = backend.get_weights_for_layer_range(simple_model, start_layer=0, last_layer=2)

    # All layers (only Linear layers have weights)
    assert len(weights) == 4

def test_get_weights_for_layer_range_auto_detect(backend, simple_model):
    """Test auto-detecting the last weight layer."""
    weights = backend.get_weights_for_layer_range(simple_model, start_layer=None)

    # Should get last Linear layer weights
    assert len(weights) == 2
    assert weights[0].shape == (2, 3)
    assert weights[1].shape == (2,)
    assert torch.all(torch.isfinite(weights[0]))
    assert torch.all(torch.isfinite(weights[1]))

def test_get_weights_for_layer_range_invalid_range(backend, simple_model):
    """Test that invalid layer range raises error."""
    with pytest.raises(AssertionError):
        backend.get_weights_for_layer_range(simple_model, start_layer=2, last_layer=0)


def test_create_influence_model(simple_model):
    """Test creating an InfluenceModel."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = nn.MSELoss(reduction='none')
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    # Should use last layer by default
    expected_params = 3 * 2 + 2  # Last linear layer
    assert influence_model.nb_params == expected_params

def test_influence_model_with_start_layer(simple_model):
    """Test InfluenceModel with specific start layer."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = nn.MSELoss(reduction='none')
    influence_model = InfluenceModel(simple_model, start_layer=0, loss_function=loss_fn)

    # First linear layer: 5*3 + 3 = 18
    expected_params = 5 * 3 + 3
    assert influence_model.nb_params == expected_params

def test_influence_model_forward(simple_model):
    """Test InfluenceModel forward pass."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = nn.MSELoss(reduction='none')
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    inputs = torch.randn(4, 5)
    output = influence_model(inputs)

    assert output.shape == (4, 2)
    assert torch.all(torch.isfinite(output))

def test_influence_model_batch_loss(simple_model):
    """Test InfluenceModel batch loss computation."""
    from deel.influenciae.common import InfluenceModel

    def loss_fn(pred, target):
        return nn.functional.mse_loss(pred, target, reduction='none').mean(dim=-1)

    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    # Create a simple dataset (list of tuples)
    dataset = [
        (torch.randn(2, 5), torch.randn(2, 2)),
        (torch.randn(2, 5), torch.randn(2, 2)),
    ]

    loss = influence_model.batch_loss(dataset)

    assert loss.shape == (4,)
    assert torch.all(torch.isfinite(loss))


def test_influence_model_batch_jacobian(simple_model):
    """Test InfluenceModel batch Jacobian computation."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = nn.MSELoss(reduction='none')
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    dataset = [
        (inputs[:2], targets[:2]),
        (inputs[2:], targets[2:]),
    ]

    jacobian = influence_model.batch_jacobian(dataset)

    assert jacobian.shape == (4, influence_model.nb_params)
    assert torch.all(torch.isfinite(jacobian))


def test_influence_model_batch_gradient(simple_model):
    """Test InfluenceModel batch gradient computation."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = nn.MSELoss(reduction='none')
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 2)
    dataset = [
        (inputs[:2], targets[:2]),
        (inputs[2:], targets[2:]),
    ]

    gradients = influence_model.batch_gradient(dataset)

    assert gradients.shape == (2, influence_model.nb_params)
    assert torch.all(torch.isfinite(gradients))


def test_loss_with_reduction_raises_error(simple_model):
    """Test that reduced PyTorch losses are rejected."""
    from deel.influenciae.common import InfluenceModel

    with pytest.raises(ValueError, match="must not have reduction"):
        InfluenceModel(simple_model, loss_function=nn.MSELoss(reduction='sum'))

    with pytest.raises(ValueError, match="must not have reduction"):
        InfluenceModel(simple_model, loss_function=nn.MSELoss(reduction='mean'))


def test_loss_without_reduction_works(simple_model):
    """Test that a per-sample PyTorch loss is accepted."""
    from deel.influenciae.common import InfluenceModel

    influence_model = InfluenceModel(simple_model, loss_function=nn.MSELoss(reduction="none"))
    assert influence_model is not None


def test_map_dataset_tuple_and_unpacked(backend):
    """Test map_dataset behavior with one-arg and two-arg map functions."""
    dataset = [
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])),
        (torch.tensor([[5.0], [6.0]]), torch.tensor([[7.0], [8.0]])),
    ]

    one_arg = backend.map_dataset(dataset, lambda batch: batch[0] + batch[1])
    two_args = backend.map_dataset(dataset, lambda a, b: a + b)

    assert torch.equal(one_arg[0], torch.tensor([[4.0], [6.0]]))
    assert torch.equal(two_args[1], torch.tensor([[12.0], [14.0]]))

def test_map_dataset_is_lazy_and_reiterable(backend):
    """Test lazy execution and multi-pass behavior of mapped datasets."""
    dataset = [
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])),
        (torch.tensor([[5.0], [6.0]]), torch.tensor([[7.0], [8.0]])),
    ]

    call_count = {"value": 0}

    def map_fn(a, b):
        call_count["value"] += 1
        return a + b

    mapped = backend.map_dataset(dataset, map_fn)
    assert call_count["value"] == 0

    first_pass = list(mapped)
    assert call_count["value"] == 2

    second_pass = list(mapped)
    assert call_count["value"] == 4
    assert torch.equal(first_pass[0], second_pass[0])

def test_map_dataset_output_supports_fluent_transforms(backend):
    """Test backend map output can be chained with fluent dataset ops."""
    dataset = [
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([3.0]),
    ]

    mapped = backend.map_dataset(dataset, lambda x: x + 1.0)
    cached = mapped.batch(2).unbatch().take(2).cache()
    values = list(cached)

    assert len(values) == 2
    assert torch.equal(values[0], torch.tensor([2.0]))
    assert torch.equal(values[1], torch.tensor([3.0]))

def test_map_dataset_one_pass_iterator_materializes(backend):
    """Test one-pass iterators are materialized for safe re-iteration."""

    def dataset_iter():
        for idx in range(3):
            x = torch.tensor([[float(idx)]])
            y = torch.tensor([[1.0]])
            yield x, y

    with pytest.warns(RuntimeWarning, match="one-pass iterator"):
        mapped = backend.map_dataset(dataset_iter(), lambda a, b: a + b)

    first = list(mapped)
    second = list(mapped)
    assert len(first) == 3
    assert len(second) == 3

def test_take_dataset_short_circuits_iteration(backend):
    """Test take_dataset only consumes the requested number of elements."""
    consumed = {"value": 0}

    def dataset_iter():
        for idx in range(10):
            consumed["value"] += 1
            yield torch.tensor([float(idx)])

    taken = backend.take_dataset(dataset_iter(), 3)
    assert consumed["value"] == 0

    values = list(taken)
    assert consumed["value"] == 3
    assert len(values) == 3

def test_cache_dataset_freezes_mapped_results(backend):
    """Test cache_dataset materializes mapped results once."""
    dataset = [
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])),
        (torch.tensor([[5.0], [6.0]]), torch.tensor([[7.0], [8.0]])),
    ]
    call_count = {"value": 0}

    def map_fn(a, b):
        call_count["value"] += 1
        return a + b

    mapped = backend.map_dataset(dataset, map_fn)
    cached = backend.cache_dataset(mapped)
    assert call_count["value"] == 2

    _ = list(cached)
    _ = list(cached)
    assert call_count["value"] == 2

def test_composed_lazy_pipeline_map_batch_unbatch_take_cache(backend):
    """Test composed lazy pipeline map->batch->unbatch->take->cache."""
    dataset = [
        (torch.tensor([1.0]), torch.tensor([10.0])),
        (torch.tensor([2.0]), torch.tensor([20.0])),
        (torch.tensor([3.0]), torch.tensor([30.0])),
        (torch.tensor([4.0]), torch.tensor([40.0])),
        (torch.tensor([5.0]), torch.tensor([50.0])),
    ]
    call_count = {"value": 0}

    def map_fn(x, y):
        call_count["value"] += 1
        return x + 1.0, y * 2.0

    mapped = backend.map_dataset(dataset, map_fn)
    batched = backend.batch_dataset(mapped, batch_size=2)
    unbatched = backend.unbatch_dataset(batched)
    taken = backend.take_dataset(unbatched, 3)

    assert call_count["value"] == 0

    cached = backend.cache_dataset(taken)
    calls_after_cache = call_count["value"]

    assert calls_after_cache >= 3
    assert len(cached) == 3

    first_pass = list(cached)
    second_pass = list(cached)

    assert call_count["value"] == calls_after_cache

    for item in [first_pass, second_pass]:
        assert len(item) == 3
        assert torch.equal(item[0][0], torch.tensor([2.0]))
        assert torch.equal(item[0][1], torch.tensor([20.0]))
        assert torch.equal(item[1][0], torch.tensor([3.0]))
        assert torch.equal(item[1][1], torch.tensor([40.0]))
        assert torch.equal(item[2][0], torch.tensor([4.0]))
        assert torch.equal(item[2][1], torch.tensor([60.0]))

def test_cache_save_load_dataset(backend, tmp_path):
    """Test caching, saving and loading datasets."""
    dataset = [torch.tensor([1.0]), torch.tensor([2.0])]
    cached = backend.cache_dataset(dataset)
    path = str(tmp_path / "pt_dataset.pt")

    backend.save_dataset(cached, path)
    loaded = backend.load_dataset(path)

    assert len(loaded) == 2
    assert torch.equal(loaded[0], torch.tensor([1.0]))

def test_get_dataset_batch_size_and_cardinality(backend):
    """Test dataset batch size and cardinality helpers."""
    dataset = TensorDataset(torch.randn(4, 2), torch.randn(4, 1))
    loader = DataLoader(dataset, batch_size=2)

    assert backend.get_dataset_batch_size(loader) == 2
    assert backend.get_dataset_cardinality(loader) == 2

def test_zip_batch_unbatch_take_dataset(backend):
    """Test zip, batch, unbatch and take operations."""
    d1 = [1, 2, 3]
    d2 = [4, 5, 6]
    zipped = backend.zip_datasets(d1, d2)
    assert zipped[0] == (1, 4)

    samples = [
        (torch.tensor([1.0, 2.0]), torch.tensor([3.0])),
        (torch.tensor([4.0, 5.0]), torch.tensor([6.0])),
        (torch.tensor([7.0, 8.0]), torch.tensor([9.0])),
        (torch.tensor([10.0, 11.0]), torch.tensor([12.0])),
    ]
    batched = backend.batch_dataset(samples, batch_size=2)
    assert len(batched) == 2
    assert batched[0][0].shape == (2, 2)

    unbatched = backend.unbatch_dataset(batched)
    assert len(unbatched) == 4
    assert torch.equal(unbatched[0][0], torch.tensor([1.0, 2.0]))

    taken = backend.take_dataset(unbatched, 2)
    assert len(taken) == 2

def test_create_dataset_from_tensors(backend):
    """Test creating datasets from tensor inputs."""
    tensor_dataset = backend.create_dataset_from_tensors(torch.tensor([1.0, 2.0]), batch_size=4)
    assert len(tensor_dataset) == 1
    assert tensor_dataset[0][0].shape == (1, 2)
    assert torch.equal(tensor_dataset[0][0], torch.tensor([[1.0, 2.0]]))

    tuple_dataset = backend.create_dataset_from_tensors(
        (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
        batch_size=4,
    )
    assert tuple_dataset[0][0].shape == (1, 2)
    assert tuple_dataset[0][1].shape == (1, 2)
    assert torch.equal(tuple_dataset[0][0], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(tuple_dataset[0][1], torch.tensor([[3.0, 4.0]]))

def test_shuffle_dataset_size_and_element_spec(backend):
    """Test shuffle, size and element spec helpers."""
    batched = [
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])),
        (torch.tensor([[5.0], [6.0]]), torch.tensor([[7.0], [8.0]])),
    ]

    shuffled = backend.shuffle_dataset(batched, buffer_size=8)
    assert backend.get_dataset_size(shuffled) == 4

    spec = backend.get_dataset_element_spec(shuffled)
    assert isinstance(spec, tuple)
    assert spec[0]['shape'] == torch.Size([2, 1])

def test_assert_batched_dataset(backend):
    """Test batched dataset assertion for valid and invalid data."""
    valid = [(torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]]))]
    backend.assert_batched_dataset(valid)

    invalid = [torch.tensor(1.0), torch.tensor(2.0)]
    with pytest.raises(ValueError):
        backend.assert_batched_dataset(invalid)


def test_ones_ones_like_and_argsort(backend):
    """Test ones constructors and argsort ordering."""
    ones = backend.ones((2, 3), dtype=backend.float32_dtype())
    ones_like = backend.ones_like(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    sorted_idx = backend.argsort(torch.tensor([3.0, 1.0, 2.0]))

    assert ones.shape == (2, 3)
    assert torch.equal(ones, torch.ones((2, 3)))
    assert torch.equal(ones_like, torch.ones((2, 2)))
    assert torch.equal(sorted_idx, torch.tensor([1, 2, 0]))

def test_assign_variable(backend, simple_model):
    """Test in-place variable assignment."""
    variable = backend.get_model_weights(simple_model)[0]
    new_value = torch.zeros_like(variable)

    backend.assign_variable(variable, new_value)
    assert torch.equal(variable, torch.zeros_like(variable))

def test_compute_hessian(backend):
    """Test Hessian computation shape and finiteness."""
    model = nn.Sequential(nn.Linear(2, 1))
    weights = backend.get_model_weights(model)
    nb_params = backend.get_num_params(weights)

    inputs = torch.tensor([[1.0, 0.0], [0.5, -1.0]])
    targets = torch.tensor([[1.0], [0.0]])
    dataset = [(inputs, targets)]

    def loss_fn(pred, target):
        return nn.functional.mse_loss(pred, target, reduction='none').mean(dim=-1)

    hessian = backend.compute_hessian(model, weights, loss_fn, dataset, nb_params)

    assert hessian.shape == (nb_params, nb_params)
    assert torch.all(torch.isfinite(hessian))
    assert torch.allclose(hessian, hessian.T, atol=1e-5, rtol=1e-5)


def test_compute_hvp_single(backend):
    """Test single-sample Hessian-vector product computation."""
    model = nn.Sequential(nn.Linear(2, 1))
    weights = backend.get_model_weights(model)
    nb_params = backend.get_num_params(weights)
    vector = [
        torch.tensor([[0.3, -0.7]], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
    ]

    inputs = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[1.0]])

    def loss_fn(pred, target):
        return nn.functional.mse_loss(pred, target, reduction='none').mean(dim=-1)

    hvp = backend.compute_hvp_single(model, weights, loss_fn, vector, inputs, targets)

    assert hvp.shape == (nb_params,)
    assert torch.all(torch.isfinite(hvp))
    assert torch.linalg.norm(hvp) > 0

def test_compute_hvp_batch(backend):
    """Test batched Hessian-vector product computation."""
    model = nn.Sequential(nn.Linear(2, 1))
    weights = backend.get_model_weights(model)
    nb_params = backend.get_num_params(weights)
    vector = [torch.ones_like(w) for w in weights]

    inputs = torch.tensor([[1.0, 0.0], [0.5, -1.0]])
    targets = torch.tensor([[1.0], [0.0]])

    def loss_fn(pred, target):
        return nn.functional.mse_loss(pred, target, reduction='none').mean(dim=-1)

    hvp = backend.compute_hvp_batch(model, weights, loss_fn, vector, inputs, targets)

    assert hvp.shape == (nb_params,)
    assert torch.all(torch.isfinite(hvp))


def test_compute_hvp_single_matches_hessian_vector_product(backend):
    """Single-sample HVP should match explicit Hessian-vector multiplication."""
    model = nn.Sequential(nn.Linear(2, 1))
    weights = backend.get_model_weights(model)
    nb_params = backend.get_num_params(weights)
    vector = [
        torch.tensor([[0.3, -0.7]], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
    ]

    inputs = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0]], dtype=torch.float32)
    dataset = [(inputs, targets)]

    def loss_fn(pred, target):
        return nn.functional.mse_loss(pred, target, reduction='none').mean(dim=-1)

    hessian = backend.compute_hessian(model, weights, loss_fn, dataset, nb_params)
    hvp = backend.compute_hvp_single(model, weights, loss_fn, vector, inputs, targets)
    vector_flat = torch.cat([v.reshape(-1) for v in vector])
    vector_col = backend.reshape(vector_flat, (-1, 1))
    hvp_from_hessian = backend.reshape(backend.matmul(hessian, vector_col), (-1,))

    assert hvp.shape == (nb_params,)
    assert torch.allclose(hvp, hvp_from_hessian, atol=1e-5, rtol=1e-5)

def test_compute_output_jacobians(backend):
    """Test output Jacobian computations w.r.t inputs and weights."""
    model = nn.Sequential(nn.Linear(2, 1))
    weights = backend.get_model_weights(model)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    outputs, jac_inputs = backend.compute_output_jacobian(model, inputs)
    outputs_w, jac_weights = backend.compute_output_jacobian_wrt_weights(model, weights, inputs)

    assert outputs.shape == (2, 1)
    assert outputs_w.shape == (2, 1)
    assert jac_inputs.shape == (2, 1, 2)
    assert len(jac_weights) == len(weights)
    assert jac_weights[0].shape[:2] == (2, 1)
    assert torch.all(torch.isfinite(outputs))
    assert torch.all(torch.isfinite(outputs_w))
    assert torch.all(torch.isfinite(jac_inputs))
    assert torch.all(torch.isfinite(jac_weights[0]))
    assert torch.all(torch.isfinite(jac_weights[1]))
    assert torch.allclose(outputs, outputs_w)


def test_map_fn_simple(backend):
    """Test map_fn with both tensor and tuple inputs."""
    elems = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = backend.map_fn(lambda x: x * 2.0, elems)
    assert torch.equal(result, torch.tensor([[2.0, 4.0], [6.0, 8.0]]))

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    tuple_result = backend.map_fn(lambda pair: pair[0] + pair[1], (a, b))
    assert torch.equal(tuple_result, torch.tensor([[6.0, 8.0], [10.0, 12.0]]))

def test_while_loop(backend):
    """Test while_loop helper with simple integer accumulation."""
    def cond_fn(i, total):
        return i < 3

    def body_fn(i, total):
        return [i + 1, total + i]

    result = backend.while_loop(cond_fn, body_fn, [0, 0], maximum_iterations=10)

    assert result[0] == 3
    assert result[1] == 3

def test_random_diag_eig_and_real(backend):
    """Test random normal, diagonal extraction and eigen helpers."""
    random_tensor = backend.random_normal((2, 3), dtype=backend.float32_dtype())
    assert random_tensor.shape == (2, 3)

    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    diag = backend.diag_part(matrix, k=1)
    assert torch.equal(diag, torch.tensor([2.0, 6.0]))

    maindiag = torch.tensor([2.0, 3.0])
    superdiag = torch.tensor([1.0])
    eig_vals, eig_vecs = backend.eigh_tridiagonal(maindiag, superdiag)
    eig_vals_only, eig_vecs_none = backend.eigh_tridiagonal(maindiag, superdiag, eigvals_only=True)

    assert eig_vals.shape == (2,)
    assert eig_vecs.shape == (2, 2)
    assert eig_vals_only.shape == (2,)
    assert eig_vecs_none is None

    eigvals, eigvecs = backend.eig(torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
    real_part = backend.real(eigvals)

    assert eigvals.shape == (2,)
    assert eigvecs.shape == (2, 2)
    assert real_part.shape == (2,)
