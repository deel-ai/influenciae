# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the PyTorch backend implementation.
These tests verify the PyTorch-specific functionality works correctly.
"""
import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except (ImportError, OSError):
    HAS_PYTORCH = False
    torch = None
    nn = None

pytestmark = pytest.mark.skipif(
    not HAS_PYTORCH,
    reason="PyTorch is required for these tests"
)


def almost_equal(a, b, epsilon=1e-4):
    """Check if two arrays are almost equal."""
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    return np.allclose(a, b, atol=epsilon, rtol=epsilon)


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


class TestPyTorchBackendBasics:
    """Test basic backend functionality."""

    def test_framework_property(self, backend):
        """Test that framework property returns PYTORCH."""
        from deel.influenciae.common import Framework
        assert backend.framework == Framework.PYTORCH

    def test_get_model_weights(self, backend, simple_model):
        """Test getting model weights."""
        weights = backend.get_model_weights(simple_model)

        # Should have 4 weight tensors: 2 weights + 2 biases
        assert len(weights) == 4

        # Check shapes
        assert weights[0].shape == (3, 5)  # First linear weight
        assert weights[1].shape == (3,)    # First linear bias
        assert weights[2].shape == (2, 3)  # Second linear weight
        assert weights[3].shape == (2,)    # Second linear bias

    def test_get_model_weights_specific_layers(self, backend, simple_model):
        """Test getting weights from specific layers."""
        children = list(simple_model.children())
        linear_layers = [children[0], children[2]]  # Skip ReLU

        weights = backend.get_model_weights(simple_model, linear_layers)
        assert len(weights) == 4

    def test_get_num_params(self, backend, simple_model):
        """Test parameter counting."""
        weights = backend.get_model_weights(simple_model)
        num_params = backend.get_num_params(weights)

        # (5*3 + 3) + (3*2 + 2) = 18 + 8 = 26
        expected = 5 * 3 + 3 + 3 * 2 + 2
        assert num_params == expected

    def test_forward(self, backend, simple_model):
        """Test forward pass."""
        inputs = torch.randn(4, 5)
        output = backend.forward(simple_model, inputs)

        assert output.shape == (4, 2)

    def test_get_layers(self, backend, simple_model):
        """Test getting all layers."""
        layers = backend.get_layers(simple_model)

        # modules() returns model + all children
        # Sequential(Linear, ReLU, Linear) = 4 modules
        assert len(layers) == 4

    def test_get_children(self, backend, simple_model):
        """Test getting direct children."""
        children = backend.get_children(simple_model)

        assert len(children) == 3
        assert isinstance(children[0], nn.Linear)
        assert isinstance(children[1], nn.ReLU)
        assert isinstance(children[2], nn.Linear)


class TestPyTorchBackendComputation:
    """Test gradient and loss computation."""

    def test_compute_loss(self, backend, simple_model):
        """Test loss computation."""
        inputs = torch.randn(4, 5)
        targets = torch.randn(4, 2)
        loss_fn = nn.MSELoss(reduction='none')

        loss = backend.compute_loss(simple_model, loss_fn, inputs, targets)

        # MSE without reduction returns (batch, output_dim)
        assert loss.shape == (4, 2)

    def test_compute_loss_with_sample_weight(self, backend, simple_model):
        """Test loss computation with sample weights."""
        inputs = torch.randn(4, 5)
        targets = torch.randn(4, 2)
        sample_weight = torch.tensor([1.0, 2.0, 0.5, 1.5])
        loss_fn = nn.MSELoss(reduction='none')

        loss_unweighted = backend.compute_loss(simple_model, loss_fn, inputs, targets)
        loss_weighted = backend.compute_loss(simple_model, loss_fn, inputs, targets, sample_weight.unsqueeze(-1))

        # Weighted loss should be different
        assert not torch.allclose(loss_unweighted, loss_weighted)

    def test_compute_gradient(self, backend, simple_model):
        """Test gradient computation."""
        inputs = torch.randn(4, 5)
        targets = torch.randn(4, 2)
        weights = backend.get_model_weights(simple_model)

        def loss_fn(pred, target):
            return nn.functional.mse_loss(pred, target, reduction='none').mean(dim=-1)

        gradient = backend.compute_gradient(simple_model, weights, loss_fn, inputs, targets)

        num_params = backend.get_num_params(weights)
        assert gradient.shape == (num_params,)
        assert torch.linalg.norm(gradient) > 0  # Gradient should be non-zero

    def test_compute_jacobian(self, backend, simple_model):
        """Test Jacobian computation."""
        batch_size = 3
        inputs = torch.randn(batch_size, 5)
        targets = torch.randn(batch_size, 2)
        weights = backend.get_model_weights(simple_model)

        def loss_fn(pred, target):
            return nn.functional.mse_loss(pred, target, reduction='none').mean(dim=-1)

        jacobian = backend.compute_jacobian(simple_model, weights, loss_fn, inputs, targets)

        num_params = backend.get_num_params(weights)
        assert jacobian.shape == (batch_size, num_params)


class TestPyTorchBackendTensorOps:
    """Test tensor operations."""

    def test_concat(self, backend):
        """Test tensor concatenation."""
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])

        result = backend.concat([a, b], axis=0)
        expected = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        assert torch.equal(result, expected)

    def test_stack(self, backend):
        """Test tensor stacking."""
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        result = backend.stack([a, b], axis=0)
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])

        assert torch.equal(result, expected)

    def test_reshape(self, backend):
        """Test tensor reshape."""
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])

        result = backend.reshape(a, (3, 2))
        assert result.shape == (3, 2)

    def test_reduce_sum(self, backend):
        """Test reduce sum."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result_full = backend.reduce_sum(a)
        assert result_full.item() == 10.0

        result_axis0 = backend.reduce_sum(a, axis=0)
        assert torch.equal(result_axis0, torch.tensor([4.0, 6.0]))

        result_axis1 = backend.reduce_sum(a, axis=1)
        assert torch.equal(result_axis1, torch.tensor([3.0, 7.0]))

    def test_to_numpy(self, backend):
        """Test numpy conversion."""
        a = torch.tensor([1.0, 2.0, 3.0])

        result = backend.to_numpy(a)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_get_batch_size(self, backend):
        """Test batch size extraction."""
        a = torch.randn(8, 5, 3)

        assert backend.get_batch_size(a) == 8

    def test_abs(self, backend):
        """Test absolute value."""
        a = torch.tensor([-1.0, 2.0, -3.0])
        result = backend.abs(a)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_argmax(self, backend):
        """Test argmax."""
        a = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 2.0]])
        result = backend.argmax(a, axis=1)
        expected = torch.tensor([1, 0])
        assert torch.equal(result, expected)

    def test_gather_along_axis(self, backend):
        """Test gather along axis."""
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        indices = torch.tensor([2, 0])
        result = backend.gather_along_axis(a, indices, axis=1, batch_dims=1)
        expected = torch.tensor([3.0, 4.0])
        assert torch.equal(result, expected)

    def test_get_output_shape(self, backend, simple_model):
        """Test getting model output shape."""
        out_shape = backend.get_output_shape(simple_model)
        assert out_shape == (None, 2)

    def test_split_model(self, backend, simple_model):
        """Test splitting model into two parts."""
        feature_extractor, head = backend.split_model(simple_model, -1)

        inputs = torch.randn(4, 5)

        # Feature extractor output should be (batch, 3) from the hidden layer
        fe_output = feature_extractor(inputs)
        assert fe_output.shape == (4, 3)

        # Head output should be (batch, 2) from the output layer
        head_output = head(fe_output)
        assert head_output.shape == (4, 2)


class TestPyTorchBackendLayerOperations:
    """Test layer-related operations."""

    def test_find_last_weight_layer(self, backend, simple_model):
        """Test finding last weight layer."""
        idx = backend.find_last_weight_layer(simple_model)

        # Last Linear layer should be at index -1
        assert idx == -1

    def test_find_last_weight_layer_with_trailing_layers(self, backend):
        """Test finding last weight layer with non-weight layers at the end."""
        model = nn.Sequential(
            nn.Linear(5, 3),
            nn.Linear(3, 2),
            nn.Softmax(dim=-1)
        )

        idx = backend.find_last_weight_layer(model)

        # Second Linear is at index -2 (Softmax has no weights)
        assert idx == -2

    def test_get_layer_index_by_int(self, backend, simple_model):
        """Test layer index lookup by integer."""
        assert backend.get_layer_index(simple_model, 0) == 0
        assert backend.get_layer_index(simple_model, 1) == 1
        assert backend.get_layer_index(simple_model, -1) == 2

    def test_get_weights_for_layer_range_single_layer(self, backend, simple_model):
        """Test getting weights for a single layer."""
        weights = backend.get_weights_for_layer_range(simple_model, start_layer=0)

        # First Linear layer: weight + bias
        assert len(weights) == 2
        assert weights[0].shape == (3, 5)
        assert weights[1].shape == (3,)

    def test_get_weights_for_layer_range_multiple_layers(self, backend, simple_model):
        """Test getting weights for multiple layers."""
        weights = backend.get_weights_for_layer_range(simple_model, start_layer=0, last_layer=2)

        # All layers (only Linear layers have weights)
        assert len(weights) == 4

    def test_get_weights_for_layer_range_auto_detect(self, backend, simple_model):
        """Test auto-detecting the last weight layer."""
        weights = backend.get_weights_for_layer_range(simple_model, start_layer=None)

        # Should get last Linear layer weights
        assert len(weights) == 2
        assert weights[0].shape == (2, 3)
        assert weights[1].shape == (2,)

    def test_get_weights_for_layer_range_invalid_range(self, backend, simple_model):
        """Test that invalid layer range raises error."""
        with pytest.raises(AssertionError):
            backend.get_weights_for_layer_range(simple_model, start_layer=2, last_layer=0)


class TestPyTorchInfluenceModel:
    """Test InfluenceModel with PyTorch backend."""

    def test_create_influence_model(self, simple_model):
        """Test creating an InfluenceModel."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = nn.MSELoss(reduction='none')
        influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

        # Should use last layer by default
        expected_params = 3 * 2 + 2  # Last linear layer
        assert influence_model.nb_params == expected_params

    def test_influence_model_with_start_layer(self, simple_model):
        """Test InfluenceModel with specific start layer."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = nn.MSELoss(reduction='none')
        influence_model = InfluenceModel(simple_model, start_layer=0, loss_function=loss_fn)

        # First linear layer: 5*3 + 3 = 18
        expected_params = 5 * 3 + 3
        assert influence_model.nb_params == expected_params

    def test_influence_model_forward(self, simple_model):
        """Test InfluenceModel forward pass."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = nn.MSELoss(reduction='none')
        influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

        inputs = torch.randn(4, 5)
        output = influence_model(inputs)

        assert output.shape == (4, 2)

    def test_influence_model_batch_loss(self, simple_model):
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

