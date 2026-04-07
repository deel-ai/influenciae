# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the TensorFlow backend implementation.
These tests verify the TensorFlow-specific functionality works correctly.
"""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, LayerNormalization, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, Reduction

from ..utils_test import assert_allclose, allclose, hessian_ground_truth, jacobian_ground_truth


pytestmark = pytest.mark.tensorflow


class _WeightWrapper:
    """Minimal wrapper emulating Keras 3 variable containers."""

    def __init__(self, value):
        self.value = value


def _make_known_linear_case_tf():
    """Build a tiny affine Keras model with hand-computable outputs and losses."""
    model = Sequential([Input(shape=(2,)), Dense(1, name='output')])
    model.layers[0].set_weights(
        [
            np.array([[2.0], [-1.0]], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        ]
    )

    inputs = tf.constant([[1.0, 2.0], [-1.0, 1.0]], dtype=tf.float32)
    targets = tf.constant([[0.0], [1.0]], dtype=tf.float32)
    expected_output = tf.constant([[0.5], [-2.5]], dtype=tf.float32)
    expected_loss = tf.constant([0.25, 12.25], dtype=tf.float32)
    return model, inputs, targets, expected_output, expected_loss


def _make_closed_form_last_layer_case_tf(backend):
    """Create a small linear regression case with closed-form last-layer derivatives."""
    model = Sequential([
        Input(shape=(1, 3)),
        Dense(2, use_bias=False, name='hidden'),
        Dense(1, use_bias=False, name='output'),
    ])
    model.layers[0].set_weights(
        [
            np.array(
                [
                    [0.2, -0.4],
                    [0.5, 0.1],
                    [-0.3, 0.7],
                ],
                dtype=np.float32,
            )
        ]
    )
    model.layers[1].set_weights([np.array([[1.2], [-0.6]], dtype=np.float32)])

    weights = backend.get_weights_for_layer_range(model, start_layer=None)
    inputs = tf.constant(
        [
            [[1.0, -2.0, 0.5]],
            [[-0.5, 1.5, 2.0]],
            [[0.0, 1.0, -1.0]],
        ],
        dtype=tf.float32,
    )
    targets = tf.constant([[0.3], [-1.2], [0.5]], dtype=tf.float32)
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in model.layers], axis=0), (-1,))
    jacobian = tf.stack(
        [tf.reshape(jacobian_ground_truth(inp[0], kernel, target), (-1,)) for inp, target in zip(inputs, targets)],
        axis=0,
    )
    hessian = tf.stack([hessian_ground_truth(tf.squeeze(inp), kernel) for inp in inputs], axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(inputs.shape[0])
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    return model, weights, inputs, targets, dataset, loss_fn, jacobian, hessian


@pytest.fixture
def simple_model():
    """Create a simple TensorFlow model for testing."""
    model = Sequential([
        Input(shape=(5,)),
        Dense(3, activation='relu', name='hidden'),
        Dense(2, name='output')
    ])
    return model


@pytest.fixture
def backend():
    """Get the TensorFlow backend."""
    from deel.influenciae.common.backend_tensorflow import TensorFlowBackend
    return TensorFlowBackend()


@pytest.fixture(autouse=True)
def _set_test_seed():
    """Keep stochastic tests deterministic and reproducible."""
    np.random.seed(1234)
    tf.random.set_seed(1234)


def test_framework_property(backend):
    """Test that framework property returns TENSORFLOW."""
    from deel.influenciae.common import Framework
    assert backend.framework == Framework.TENSORFLOW

def test_get_model_weights(backend, simple_model):
    """Test getting model weights."""
    weights = backend.get_model_weights(simple_model)

    # Should have 4 weight tensors: 2 weights + 2 biases
    assert len(weights) == 4

    # Check shapes (TF uses [in, out] convention)
    assert weights[0].shape == (5, 3)  # First dense weight
    assert weights[1].shape == (3,)    # First dense bias
    assert weights[2].shape == (3, 2)  # Second dense weight
    assert weights[3].shape == (2,)    # Second dense bias

    for weight in weights:
        assert backend.get_dtype(weight) == backend.float32_dtype()
        assert np.all(np.isfinite(backend.to_numpy(weight)))


def test_normalize_weights_to_watch(backend, simple_model):
    """normalize_weights_to_watch should unwrap Keras-style wrappers consistently."""
    weights = backend.get_model_weights(simple_model)
    wrapped_weights = [_WeightWrapper(weight) for weight in weights]

    normalized = backend.normalize_weights_to_watch(wrapped_weights)
    normalized_twice = backend.normalize_weights_to_watch(normalized)

    assert len(normalized) == len(weights)
    assert len(normalized_twice) == len(weights)

    for raw_weight, first_pass, second_pass in zip(weights, normalized, normalized_twice):
        assert first_pass.shape == raw_weight.shape
        assert second_pass.shape == raw_weight.shape
        assert allclose(backend.to_numpy(first_pass), backend.to_numpy(raw_weight))
        assert allclose(backend.to_numpy(second_pass), backend.to_numpy(first_pass))

def test_get_model_weights_specific_layers(backend, simple_model):
    """Test getting weights from specific layers."""
    layers = [simple_model.layers[0], simple_model.layers[1]]

    weights = backend.get_model_weights(simple_model, layers)
    assert len(weights) == 4

def test_get_num_params(backend, simple_model):
    """Test parameter counting."""
    weights = backend.get_model_weights(simple_model)
    num_params = backend.get_num_params(weights)

    # (5*3 + 3) + (3*2 + 2) = 18 + 8 = 26
    expected = 5 * 3 + 3 + 3 * 2 + 2
    assert num_params == expected

def test_forward(backend):
    """Test forward pass."""
    model, inputs, _, expected_output, _ = _make_known_linear_case_tf()
    output = backend.forward(model, inputs)

    assert output.shape == (2, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(output)).numpy())
    assert_allclose(output, model(inputs))
    assert_allclose(output, expected_output)

def test_get_layers(backend, simple_model):
    """Test getting all layers."""
    layers = backend.get_layers(simple_model)

    # 2 Dense layers (Input is not counted in model.layers for Sequential with Input)
    assert len(layers) == 2

def test_find_layer_by_name(backend, simple_model):
    """Test finding layer by name."""
    idx, layer = backend.find_layer_by_name(simple_model, 'hidden')

    assert idx == 0
    assert layer.name == 'hidden'

def test_find_layer_by_name_not_found(backend, simple_model):
    """Test finding non-existent layer raises error."""
    with pytest.raises(ValueError):
        backend.find_layer_by_name(simple_model, 'nonexistent')


def test_compute_loss(backend):
    """Test loss computation."""
    model, inputs, targets, _, expected_loss = _make_known_linear_case_tf()
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    loss = backend.compute_loss(model, loss_fn, inputs, targets)
    expected = loss_fn(targets, model(inputs))

    assert loss.shape == (2,)
    assert bool(tf.reduce_all(tf.math.is_finite(loss)).numpy())
    assert bool(tf.reduce_all(loss >= 0).numpy())
    assert_allclose(loss, expected)
    assert_allclose(loss, expected_loss)

def test_compute_loss_with_sample_weight(backend):
    """Test loss computation with sample weights."""
    model, inputs, targets, _, expected_loss = _make_known_linear_case_tf()
    sample_weight = tf.constant([1.0, 2.0], dtype=tf.float32)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    loss_unweighted = backend.compute_loss(model, loss_fn, inputs, targets)
    loss_weighted = backend.compute_loss(model, loss_fn, inputs, targets, sample_weight)
    expected_weighted = expected_loss * sample_weight

    assert not np.allclose(loss_unweighted.numpy(), loss_weighted.numpy())
    assert_allclose(loss_unweighted, expected_loss)
    assert_allclose(loss_weighted, expected_weighted)

def test_compute_gradient(backend):
    """Test gradient computation."""
    model, weights, inputs, targets, _, loss_fn, jacobian, _ = _make_closed_form_last_layer_case_tf(backend)

    gradient = backend.compute_gradient(model, weights, loss_fn, inputs, targets)
    expected_gradient = tf.reduce_sum(jacobian, axis=0)

    num_params = backend.get_num_params(weights)
    assert gradient.shape == (num_params,)
    assert tf.linalg.norm(gradient) > 0  # Gradient should be non-zero
    assert_allclose(gradient, expected_gradient)

def test_compute_jacobian(backend):
    """Test Jacobian computation."""
    model, weights, inputs, targets, _, loss_fn, expected_jacobian, _ = _make_closed_form_last_layer_case_tf(backend)

    jacobian = backend.compute_jacobian(model, weights, loss_fn, inputs, targets)

    num_params = backend.get_num_params(weights)
    assert jacobian.shape == (inputs.shape[0], num_params)
    assert bool(tf.reduce_all(tf.math.is_finite(jacobian)).numpy())
    assert float(tf.linalg.norm(jacobian).numpy()) > 0.0
    assert_allclose(jacobian, expected_jacobian)


def test_validate_loss_no_reduction(backend):
    """Loss reduction validation should accept NONE and reject reduced losses."""
    backend.validate_loss_no_reduction(MeanSquaredError(reduction=Reduction.NONE))

    with pytest.raises(ValueError, match="must not have reduction"):
        backend.validate_loss_no_reduction(MeanSquaredError())


def test_dense_layer_helpers_and_clone_model(backend, simple_model):
    """Dense-layer helpers should expose structural information consistently."""
    _ = simple_model(tf.zeros((1, 5), dtype=tf.float32))
    output_layer = simple_model.layers[-1]

    assert backend.is_dense_linear_layer(output_layer)
    assert backend.layer_has_bias(output_layer)
    assert backend.get_layer_io_features(output_layer) == (3, 2)
    assert backend.get_linear_weight_axes() == (0, 1)
    assert backend.find_last_weight_layer(simple_model) == -1
    assert backend.get_layer_index(simple_model, None) == 1
    assert backend.get_layer_index(simple_model, 'output') == 1
    assert backend.get_layer_index(simple_model, -1) == 1

    cloned_model = backend.clone_model(simple_model)
    cloned_weights = backend.get_model_weights(cloned_model)
    original_weights = backend.get_model_weights(simple_model)

    for original, cloned in zip(original_weights, cloned_weights):
        assert_allclose(original, cloned)
        assert cloned is not original


def test_create_linear_model(backend):
    """Linear model creation should preserve TensorFlow shape and dtype semantics."""
    reference_weight = tf.Variable(tf.ones((2, 3), dtype=tf.float64))
    linear_model = backend.create_linear_model(
        input_shape=(1, 2),
        out_features=3,
        use_bias=False,
        l2_regularization=0.5,
        reference_weight=reference_weight,
    )

    outputs = linear_model(tf.ones((4, 1, 2), dtype=tf.float64))

    assert outputs.shape == (4, 1, 3)
    assert linear_model.layers[-1].use_bias is False
    assert linear_model.layers[-1].kernel.dtype == tf.float64
    assert len(linear_model.losses) == 1


def test_ensure_per_sample_loss_and_normalize_binary_targets(backend):
    """TensorFlow helpers should normalize common RPS loss/target shapes."""
    matrix_loss = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    per_sample_loss = backend.ensure_per_sample_loss(matrix_loss)
    assert_allclose(per_sample_loss, tf.constant([3.0, 7.0], dtype=tf.float32))

    with pytest.raises(ValueError, match="per-sample"):
        backend.ensure_per_sample_loss(tf.constant(1.0, dtype=tf.float32))

    targets = tf.constant([0.0, 1.0], dtype=tf.float32)
    logits = tf.constant([[0.1], [0.2]], dtype=tf.float32)
    normalized_targets = backend.normalize_binary_targets(targets, logits)
    assert normalized_targets.shape == (2, 1)


def test_compute_gradient_raises_on_disconnected_graph(backend, simple_model):
    """Disconnected gradients should raise an explicit error."""
    inputs = tf.random.normal((4, 5))
    targets = tf.random.normal((4, 2))
    rogue_weight = tf.Variable(tf.ones((2, 2), dtype=tf.float32))
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    with pytest.raises(ValueError, match="disconnected"):
        backend.compute_gradient(simple_model, [rogue_weight], loss_fn, inputs, targets)


def test_compute_jacobian_raises_on_disconnected_graph(backend, simple_model):
    """Disconnected Jacobians should raise an explicit error."""
    inputs = tf.random.normal((4, 5))
    targets = tf.random.normal((4, 2))
    rogue_weight = tf.Variable(tf.ones((2, 2), dtype=tf.float32))
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    with pytest.raises(ValueError, match="disconnected"):
        backend.compute_jacobian(simple_model, [rogue_weight], loss_fn, inputs, targets)


def test_weight_wrappers_are_supported_for_gradients_and_jacobians(backend, simple_model):
    """Keras-style weight wrappers should be normalized before tape.watch."""
    batch_size = 3
    inputs = tf.random.normal((batch_size, 5))
    targets = tf.random.normal((batch_size, 2))
    weights = backend.get_model_weights(simple_model)
    wrapped_weights = [_WeightWrapper(weight) for weight in weights]
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    gradient = backend.compute_gradient(simple_model, wrapped_weights, loss_fn, inputs, targets)
    jacobian = backend.compute_jacobian(simple_model, wrapped_weights, loss_fn, inputs, targets)
    outputs, jac_weights = backend.compute_output_jacobian_wrt_weights(simple_model, wrapped_weights, inputs)

    num_params = backend.get_num_params(weights)
    assert gradient.shape == (num_params,)
    assert jacobian.shape == (batch_size, num_params)
    assert outputs.shape == (batch_size, 2)
    assert len(jac_weights) == len(weights)
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(jacobian)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(outputs)).numpy())
    assert all(bool(tf.reduce_all(tf.math.is_finite(jac)).numpy()) for jac in jac_weights)


def test_eigh(backend):
    """Symmetric eigendecomposition should reconstruct the input matrix."""
    matrix = tf.constant([[2.0, 1.0], [1.0, 3.0]], dtype=tf.float64)

    eigenvalues, eigenvectors = backend.eigh(matrix)
    reconstructed = eigenvectors @ tf.linalg.diag(eigenvalues) @ tf.transpose(eigenvectors)

    assert eigenvalues[0] < eigenvalues[1]
    assert np.allclose(reconstructed.numpy(), matrix.numpy(), atol=1e-10)
    assert np.allclose(
        (tf.transpose(eigenvectors) @ eigenvectors).numpy(),
        np.eye(2, dtype=np.float64),
        atol=1e-10,
    )


def test_kron(backend):
    """Kronecker product should match NumPy."""
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
    b = tf.constant([[0.0, 5.0], [6.0, 7.0]], dtype=tf.float64)

    result = backend.kron(a, b)
    expected = np.kron(a.numpy(), b.numpy())

    assert np.allclose(result.numpy(), expected, atol=1e-12)
    assert result.shape == (4, 4)


def test_outer(backend):
    """Outer product should match NumPy."""
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
    b = tf.constant([4.0, 5.0], dtype=tf.float64)

    result = backend.outer(a, b)
    expected = np.outer(a.numpy(), b.numpy())

    assert np.allclose(result.numpy(), expected, atol=1e-12)
    assert result.shape == (3, 2)


def test_eye(backend):
    """Identity helper should match NumPy."""
    identity = backend.eye(3, dtype=tf.float64)
    assert np.allclose(identity.numpy(), np.eye(3), atol=1e-12)


def test_is_linear_layer(backend):
    """Dense layers should be detected as linear layers."""
    model = Sequential([Input(shape=(3,)), Dense(2)])

    assert backend.is_linear_layer(model.layers[0])
    assert not backend.is_linear_layer(Conv2D(16, 3))
    assert not backend.is_linear_layer(BatchNormalization())


def test_is_conv2d_layer(backend):
    """Conv2D layers should be detected by the backend."""
    model = Sequential([Input(shape=(3,)), Dense(2)])

    assert backend.is_conv2d_layer(Conv2D(16, 3))
    assert not backend.is_conv2d_layer(model.layers[0])
    assert not backend.is_conv2d_layer(LayerNormalization())


def test_get_layer_weight_and_bias_dense_no_bias(backend):
    """Weight extraction should return no bias when disabled."""
    model = Sequential([Input(shape=(4,)), Dense(3, use_bias=False)])
    layer = model.layers[0]

    weight, bias = backend.get_layer_weight_and_bias(layer)
    assert weight is layer.kernel
    assert bias is None


def test_get_layer_weight_and_bias_dense_with_bias(backend):
    """Weight extraction should return layer kernel and bias."""
    model = Sequential([Input(shape=(4,)), Dense(3, use_bias=True)])
    layer = model.layers[0]

    weight, bias = backend.get_layer_weight_and_bias(layer)
    assert weight is layer.kernel
    assert bias is layer.bias


def test_forward_hook(backend):
    """Forward hooks should receive layer input and output."""
    model = Sequential([Input(shape=(3,)), Dense(2)])
    layer = model.layers[0]
    captured = {}

    def hook(module, inp, out):
        captured['input'] = inp
        captured['output'] = out

    handle = backend.register_forward_hook(layer, hook)
    _ = model(tf.random.normal((1, 3)))

    assert 'input' in captured
    assert 'output' in captured
    backend.remove_hook(handle)


def test_remove_hook_stops_capture(backend):
    """Removing a hook should prevent further callbacks."""
    model = Sequential([Input(shape=(3,)), Dense(2)])
    layer = model.layers[0]
    call_count = {'value': 0}

    def hook(module, inp, out):
        call_count['value'] += 1

    handle = backend.register_forward_hook(layer, hook)
    _ = model(tf.random.normal((1, 3)))
    assert call_count['value'] == 1

    backend.remove_hook(handle)
    _ = model(tf.random.normal((1, 3)))
    assert call_count['value'] == 1


def test_backward_hook_registered(backend):
    """Backward hooks should be stored and removed on the layer."""
    model = Sequential([Input(shape=(3,)), Dense(2)])
    layer = model.layers[0]

    def hook(module, grad_input, grad_output):
        del module, grad_input, grad_output

    handle = backend.register_backward_hook(layer, hook)

    assert hasattr(layer, '_kfac_backward_hooks')
    assert hook in layer._kfac_backward_hooks

    backend.remove_hook(handle)
    assert hook not in layer._kfac_backward_hooks


def test_concat(backend):
    """Test tensor concatenation."""
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])

    result = backend.concat([a, b], axis=0)
    expected = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]])

    assert tf.reduce_all(result == expected)

def test_stack(backend):
    """Test tensor stacking."""
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])

    result = backend.stack([a, b], axis=0)
    expected = tf.constant([[1, 2, 3], [4, 5, 6]])

    assert tf.reduce_all(result == expected)

def test_reshape(backend):
    """Test tensor reshape."""
    a = tf.constant([[1, 2, 3], [4, 5, 6]])

    result = backend.reshape(a, (3, 2))
    assert result.shape == (3, 2)
    assert bool(tf.reduce_all(tf.reshape(result, (-1,)) == tf.reshape(a, (-1,))).numpy())

def test_map_fn_output_signature(backend):
    """Test map_fn with output signature and tuple elems."""
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

    def add_pair(values):
        x, y = values
        return x + y

    output_signature = tf.TensorSpec(shape=(2,), dtype=tf.float32)
    result = backend.map_fn(add_pair, (a, b), output_signature=output_signature)
    expected = tf.constant([[6.0, 8.0], [10.0, 12.0]])

    assert tf.reduce_all(result == expected)

def test_reduce_sum(backend):
    """Test reduce sum."""
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    result_full = backend.reduce_sum(a)
    assert result_full.numpy() == 10.0

    result_axis0 = backend.reduce_sum(a, axis=0)
    assert tf.reduce_all(result_axis0 == tf.constant([4.0, 6.0]))

    result_axis1 = backend.reduce_sum(a, axis=1)
    assert tf.reduce_all(result_axis1 == tf.constant([3.0, 7.0]))

def test_to_numpy(backend):
    """Test numpy conversion."""
    a = tf.constant([1.0, 2.0, 3.0])

    result = backend.to_numpy(a)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

def test_get_batch_size(backend):
    """Test batch size extraction."""
    a = tf.random.normal((8, 5, 3))

    # Note: TF returns a tensor, need to evaluate
    batch_size = backend.get_batch_size(a)
    if isinstance(batch_size, tf.Tensor):
        batch_size = batch_size.numpy()
    assert batch_size == 8

def test_abs(backend):
    """Test absolute value."""
    a = tf.constant([-1.0, 2.0, -3.0])
    result = backend.abs(a)
    expected = tf.constant([1.0, 2.0, 3.0])
    assert tf.reduce_all(result == expected)

def test_argmax(backend):
    """Test argmax."""
    a = tf.constant([[1.0, 3.0, 2.0], [4.0, 1.0, 2.0]])
    result = backend.argmax(a, axis=1)
    expected = tf.constant([1, 0], dtype=tf.int64)
    assert tf.reduce_all(result == expected)

def test_gather_along_axis(backend):
    """Test gather along axis."""
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = tf.constant([2, 0])
    result = backend.gather_along_axis(a, indices, axis=1, batch_dims=1)
    expected = tf.constant([3.0, 4.0])
    assert tf.reduce_all(result == expected)

def test_get_output_shape(backend, simple_model):
    """Test getting model output shape."""
    out_shape = backend.get_output_shape(simple_model)
    assert out_shape == (None, 2)

def test_split_model(backend, simple_model):
    """Test splitting model into two parts."""
    feature_extractor, head = backend.split_model(simple_model, -1)

    inputs = tf.random.normal((4, 5))

    # Feature extractor output should be (batch, 3) from the hidden layer
    fe_output = feature_extractor(inputs)
    assert fe_output.shape == (4, 3)

    # Head output should be (batch, 2) from the output layer
    head_output = head(fe_output)
    assert head_output.shape == (4, 2)

    # Combined should equal original model output
    original_output = simple_model(inputs)
    assert allclose(head_output, original_output)


def test_split_model_by_name(backend, simple_model):
    """Splitting by layer name should keep a valid connected Functional graph."""
    feature_extractor, head = backend.split_model(simple_model, 'output')

    inputs = tf.random.normal((4, 5))
    fe_output = feature_extractor(inputs)
    head_output = head(fe_output)

    assert fe_output.shape == (4, 3)
    assert head_output.shape == (4, 2)
    assert allclose(head_output, simple_model(inputs))


def test_find_last_weight_layer(backend, simple_model):
    """Test finding last weight layer."""
    idx = backend.find_last_weight_layer(simple_model)

    # Last Dense layer should be at index -1
    assert idx == -1

def test_find_last_weight_layer_with_trailing_layers(backend):
    """Test finding last weight layer with non-weight layers at the end."""
    model = Sequential([
        Input(shape=(5,)),
        Dense(3),
        Dense(2),
        Flatten()  # No weights
    ])

    idx = backend.find_last_weight_layer(model)

    # Second Dense is at -2, Flatten at -1
    assert idx == -2

def test_get_layer_index_by_int(backend, simple_model):
    """Test layer index lookup by integer."""
    assert backend.get_layer_index(simple_model, 0) == 0
    assert backend.get_layer_index(simple_model, 1) == 1
    assert backend.get_layer_index(simple_model, -1) == 1  # 2 layers, -1 -> 1

def test_get_layer_index_by_name(backend, simple_model):
    """Test layer index lookup by name."""
    idx = backend.get_layer_index(simple_model, 'hidden')
    assert idx == 0

    idx = backend.get_layer_index(simple_model, 'output')
    assert idx == 1

def test_get_weights_for_layer_range_single_layer(backend, simple_model):
    """Test getting weights for a single layer."""
    weights = backend.get_weights_for_layer_range(simple_model, start_layer=0)

    # First Dense layer: weight + bias
    assert len(weights) == 2
    assert weights[0].shape == (5, 3)
    assert weights[1].shape == (3,)
    assert np.all(np.isfinite(backend.to_numpy(weights[0])))
    assert np.all(np.isfinite(backend.to_numpy(weights[1])))

def test_get_weights_for_layer_range_by_name(backend, simple_model):
    """Test getting weights by layer name."""
    weights = backend.get_weights_for_layer_range(simple_model, start_layer='hidden')

    assert len(weights) == 2
    assert weights[0].shape == (5, 3)
    assert np.all(np.isfinite(backend.to_numpy(weights[0])))
    assert np.all(np.isfinite(backend.to_numpy(weights[1])))

def test_get_weights_for_layer_range_multiple_layers(backend, simple_model):
    """Test getting weights for multiple layers."""
    weights = backend.get_weights_for_layer_range(simple_model, start_layer=0, last_layer=1)

    # Both Dense layers
    assert len(weights) == 4

def test_get_weights_for_layer_range_auto_detect(backend):
    """Test auto-detecting the last weight layer."""
    model = Sequential([
        Input(shape=(5,)),
        Dense(4, activation='relu', name='hidden'),
        Dense(3, name='projection'),
        BatchNormalization(name='norm'),
        ReLU(name='relu_out'),
    ])
    model(tf.zeros((1, 5), dtype=tf.float32))

    weights = backend.get_weights_for_layer_range(model, start_layer=None)
    expected_weights = backend.get_model_weights(model, [model.get_layer('norm')])

    assert len(weights) == 2
    assert len(expected_weights) == 2
    assert weights[0].shape == (3,)
    assert weights[1].shape == (3,)
    assert_allclose(weights[0], expected_weights[0])
    assert_allclose(weights[1], expected_weights[1])

def test_get_weights_for_layer_range_invalid_range(backend, simple_model):
    """Test that invalid layer range raises error."""
    with pytest.raises(AssertionError):
        backend.get_weights_for_layer_range(simple_model, start_layer=1, last_layer=0)


def test_create_influence_model(simple_model):
    """Test creating an InfluenceModel."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    # Should use last layer by default (the output layer)
    expected_params = 3 * 2 + 2  # Output dense layer
    assert influence_model.nb_params == expected_params

def test_influence_model_with_start_layer(simple_model):
    """Test InfluenceModel with specific start layer."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(simple_model, start_layer=0, loss_function=loss_fn)

    # First dense layer: 5*3 + 3 = 18
    expected_params = 5 * 3 + 3
    assert influence_model.nb_params == expected_params

def test_influence_model_with_layer_name(simple_model):
    """Test InfluenceModel with layer name."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(simple_model, start_layer='output', loss_function=loss_fn)

    # Output layer: 3*2 + 2 = 8
    expected_params = 3 * 2 + 2
    assert influence_model.nb_params == expected_params

def test_influence_model_forward(simple_model):
    """Test InfluenceModel forward pass."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    inputs = tf.random.normal((4, 5))
    output = influence_model(inputs)

    assert output.shape == (4, 2)
    assert bool(tf.reduce_all(tf.math.is_finite(output)).numpy())
    assert_allclose(output, simple_model(inputs))

def test_influence_model_batch_loss(simple_model):
    """Test InfluenceModel batch loss computation."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    # Create a batched dataset
    inputs = tf.random.normal((4, 5))
    targets = tf.random.normal((4, 2))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)

    loss = influence_model.batch_loss(dataset)
    expected_loss = tf.concat(
        [loss_fn(batch_targets, simple_model(batch_inputs)) for batch_inputs, batch_targets in dataset],
        axis=0,
    )

    assert loss.shape == (4,)
    assert bool(tf.reduce_all(tf.math.is_finite(loss)).numpy())
    assert_allclose(loss, expected_loss)

def test_influence_model_batch_jacobian(simple_model):
    """Test InfluenceModel batch Jacobian computation."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    inputs = tf.random.normal((4, 5))
    targets = tf.random.normal((4, 2))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)

    jacobian = influence_model.batch_jacobian(dataset)
    expected_jacobian = tf.concat(
        [
            influence_model.backend.compute_jacobian(
                simple_model,
                influence_model.weights,
                loss_fn,
                batch_inputs,
                batch_targets,
            )
            for batch_inputs, batch_targets in dataset
        ],
        axis=0,
    )

    assert jacobian.shape == (4, influence_model.nb_params)
    assert bool(tf.reduce_all(tf.math.is_finite(jacobian)).numpy())
    assert_allclose(jacobian, expected_jacobian)

def test_influence_model_batch_gradient(simple_model):
    """Test InfluenceModel batch gradient computation."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    inputs = tf.random.normal((4, 5))
    targets = tf.random.normal((4, 2))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)

    gradients = influence_model.batch_gradient(dataset)
    expected_gradients = tf.stack(
        [
            influence_model.backend.compute_gradient(
                simple_model,
                influence_model.weights,
                loss_fn,
                batch_inputs,
                batch_targets,
            )
            for batch_inputs, batch_targets in dataset
        ],
        axis=0,
    )

    # 2 batches, each producing a gradient
    assert gradients.shape == (2, influence_model.nb_params)
    assert bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy())
    assert_allclose(gradients, expected_gradients)


def test_loss_with_reduction_raises_error(simple_model):
    """Test that loss function with reduction raises ValueError."""
    from deel.influenciae.common import InfluenceModel

    loss_fn_sum = MeanSquaredError(reduction=Reduction.SUM)
    loss_fn_mean = MeanSquaredError(reduction=Reduction.SUM_OVER_BATCH_SIZE)

    with pytest.raises(ValueError):
        InfluenceModel(simple_model, loss_function=loss_fn_sum)

    with pytest.raises(ValueError):
        InfluenceModel(simple_model, loss_function=loss_fn_mean)

def test_loss_without_reduction_works(simple_model):
    """Test that loss function without reduction works."""
    from deel.influenciae.common import InfluenceModel

    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

    assert influence_model is not None


def test_map_dataset(backend):
    """Test mapping a function over a batched dataset."""
    x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
    y = tf.constant([[5.0], [6.0], [7.0], [8.0]])
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

    mapped = backend.map_dataset(dataset, lambda a, b: (a + 1.0, b + 2.0), device="CPU:0")
    first_x, first_y = next(iter(mapped))

    assert allclose(first_x, tf.constant([[2.0], [3.0]]))
    assert allclose(first_y, tf.constant([[7.0], [8.0]]))

def test_cache_save_load_dataset(backend, tmp_path):
    """Test caching, saving and loading datasets."""
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant([1, 2]), tf.constant([3, 4]))).batch(1)
    cached = backend.cache_dataset(dataset)

    save_path = str(tmp_path / "tf_dataset")
    backend.save_dataset(cached, save_path)
    loaded = backend.load_dataset(save_path)

    loaded_values = list(loaded.as_numpy_iterator())
    assert len(loaded_values) == 2
    assert np.array_equal(loaded_values[0][0], np.array([1]))

def test_get_dataset_batch_size_and_cardinality(backend):
    """Test dataset batch size and cardinality helpers."""
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3, 4])).batch(2)

    assert backend.get_dataset_batch_size(dataset) == 2
    assert backend.get_dataset_cardinality(dataset) == 2

def test_zip_batch_unbatch_take_dataset(backend):
    """Test zip, batch, unbatch and take operations."""
    d1 = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3, 4])).batch(2)
    d2 = tf.data.Dataset.from_tensor_slices(tf.constant([5, 6, 7, 8])).batch(2)

    zipped = backend.zip_datasets(d1, d2)
    first_pair = next(iter(zipped))
    assert np.array_equal(first_pair[0].numpy(), np.array([1, 2]))
    assert np.array_equal(first_pair[1].numpy(), np.array([5, 6]))

    unbatched = backend.unbatch_dataset(d1)
    taken = backend.take_dataset(unbatched, 3)
    taken_values = list(taken.as_numpy_iterator())
    assert taken_values == [1, 2, 3]

    rebatched = backend.batch_dataset(unbatched, 2)
    rebatched_values = list(rebatched.as_numpy_iterator())
    assert len(rebatched_values) == 2
    assert np.array_equal(rebatched_values[0], np.array([1, 2]))

def test_create_dataset_from_tensors(backend):
    """Test creating a dataset from a tuple of tensors."""
    tensors = (tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0]))
    dataset = backend.create_dataset_from_tensors(tensors, batch_size=8)
    element = next(iter(dataset))

    assert element[0].shape == (1, 2)
    assert element[1].shape == (1, 2)
    assert np.array_equal(element[0].numpy(), np.array([[1.0, 2.0]], dtype=np.float32))
    assert np.array_equal(element[1].numpy(), np.array([[3.0, 4.0]], dtype=np.float32))


def test_create_dataset_from_tensor_slices(backend):
    """Tensor slice dataset creation should preserve per-sample semantics."""
    tensors = (
        tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32),
        tf.constant([[4.0], [5.0], [6.0]], dtype=tf.float32),
    )
    dataset = backend.create_dataset_from_tensor_slices(tensors, batch_size=2)
    batches = list(dataset.as_numpy_iterator())

    assert len(batches) == 2
    assert np.array_equal(batches[0][0], np.array([[1.0], [2.0]], dtype=np.float32))
    assert np.array_equal(batches[0][1], np.array([[4.0], [5.0]], dtype=np.float32))
    assert np.array_equal(batches[1][0], np.array([[3.0]], dtype=np.float32))
    assert np.array_equal(batches[1][1], np.array([[6.0]], dtype=np.float32))

def test_shuffle_dataset_and_size(backend):
    """Test shuffling and counting dataset elements."""
    x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
    y = tf.constant([[5.0], [6.0], [7.0], [8.0]])
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    shuffled = backend.shuffle_dataset(dataset, buffer_size=4)
    shuffled_batched = backend.batch_dataset(shuffled, 2)
    assert backend.get_dataset_size(shuffled_batched) == 4

def test_get_dataset_element_spec_and_assert_batched(backend):
    """Test element_spec lookup and batch assertion."""
    batched = tf.data.Dataset.from_tensor_slices((tf.constant([1, 2]), tf.constant([3, 4]))).batch(1)
    spec = backend.get_dataset_element_spec(batched)
    assert isinstance(spec, tuple)
    assert spec[0].shape == tf.TensorShape([None])

    backend.assert_batched_dataset(batched)

    unbatched = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2]))
    with pytest.raises(ValueError):
        backend.assert_batched_dataset(unbatched)


def test_ones_ones_like_and_argsort(backend):
    """Test ones constructors and argsort ordering."""
    ones = backend.ones((2, 3), dtype=backend.float32_dtype())
    ones_like = backend.ones_like(tf.constant([[0.0, 0.0], [0.0, 0.0]]))
    sorted_idx = backend.argsort(tf.constant([3.0, 1.0, 2.0]))

    assert ones.shape == (2, 3)
    assert tf.reduce_all(ones == 1.0)
    assert tf.reduce_all(ones_like == 1.0)
    assert np.array_equal(sorted_idx.numpy(), np.array([1, 2, 0]))

def test_assign_variable(backend, simple_model):
    """Test in-place variable assignment."""
    variable = backend.get_model_weights(simple_model)[0]
    new_value = tf.zeros_like(variable)

    backend.assign_variable(variable, new_value)
    assert tf.reduce_all(variable == 0.0)

def test_compute_hessian(backend):
    """Test Hessian computation against the closed-form last-layer Hessian."""
    model, weights, _, _, dataset, loss_fn, _, hessian_stack = _make_closed_form_last_layer_case_tf(backend)
    nb_params = backend.get_num_params(weights)

    hessian = backend.compute_hessian(model, weights, loss_fn, dataset, nb_params)
    expected_hessian = tf.reduce_mean(hessian_stack, axis=0)

    assert hessian.shape == (nb_params, nb_params)
    assert tf.reduce_all(tf.math.is_finite(hessian))
    assert np.allclose(backend.to_numpy(hessian), backend.to_numpy(tf.transpose(hessian)), atol=1e-5, rtol=1e-5)
    assert_allclose(hessian, expected_hessian)


def test_second_order_ops_reduce_multidim_per_sample_losses(backend):
    """Second-order helpers should collapse multi-dimensional per-sample losses consistently."""
    model = Sequential([Input(shape=(2,)), Dense(2, name='output')])
    _ = model(tf.zeros((1, 2), dtype=tf.float32))
    weights = backend.get_model_weights(model)
    nb_params = backend.get_num_params(weights)

    inputs = tf.constant([[1.0, -0.5]], dtype=tf.float32)
    targets = tf.constant([[0.3, -1.2]], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(1)
    vector = [tf.ones_like(weight) for weight in weights]

    def loss_fn(target, pred):
        return tf.math.squared_difference(target, pred)

    hessian = backend.compute_hessian(model, weights, loss_fn, dataset, nb_params)
    hvp = backend.compute_hvp_single(model, weights, loss_fn, vector, inputs, targets)
    vector_flat = backend.concat([backend.reshape(v, (-1,)) for v in vector], axis=0)
    expected_hvp = backend.reshape(backend.matmul(hessian, backend.reshape(vector_flat, (-1, 1))), (-1,))

    assert hvp.shape == (nb_params,)
    assert tf.reduce_all(tf.math.is_finite(hvp))
    assert_allclose(hvp, expected_hvp)


def test_compute_hvp_single(backend):
    """Test single-sample Hessian-vector product computation."""
    model, weights, inputs, targets, _, loss_fn, _, hessian = _make_closed_form_last_layer_case_tf(backend)
    nb_params = backend.get_num_params(weights)
    vector = [tf.constant([[0.3], [-0.7]], dtype=tf.float32)]

    hvp = backend.compute_hvp_single(model, weights, loss_fn, vector, inputs[:1], targets[:1])
    vector_flat = tf.reshape(vector[0], (-1,))
    expected_hvp = tf.linalg.matvec(hessian[0], vector_flat)

    assert hvp.shape == (nb_params,)
    assert tf.reduce_all(tf.math.is_finite(hvp))
    assert float(tf.linalg.norm(hvp).numpy()) > 0.0
    assert_allclose(hvp, expected_hvp)

def test_compute_hvp_batch(backend):
    """Test batched Hessian-vector product computation."""
    model, weights, inputs, targets, _, loss_fn, _, hessian = _make_closed_form_last_layer_case_tf(backend)
    nb_params = backend.get_num_params(weights)
    vector = [tf.constant([[0.3], [-0.7]], dtype=tf.float32)]

    hvp = backend.compute_hvp_batch(model, weights, loss_fn, vector, inputs, targets)
    vector_flat = tf.reshape(vector[0], (-1,))
    expected_hvp = tf.reduce_sum(tf.einsum('bij,j->bi', hessian, vector_flat), axis=0)

    assert hvp.shape == (nb_params,)
    assert tf.reduce_all(tf.math.is_finite(hvp))
    assert_allclose(hvp, expected_hvp)


def test_compute_hvp_single_is_linear_in_direction(backend):
    """Single-sample HVP should be linear in the direction vector."""
    model, weights, inputs, targets, _, loss_fn, _, _ = _make_closed_form_last_layer_case_tf(backend)
    v1 = [tf.constant([[0.3], [-0.7]], dtype=tf.float32)]
    v2 = [tf.constant([[-0.2], [0.4]], dtype=tf.float32)]
    v_sum = [left + right for left, right in zip(v1, v2)]

    hvp_v1 = backend.compute_hvp_single(model, weights, loss_fn, v1, inputs[:1], targets[:1])
    hvp_v2 = backend.compute_hvp_single(model, weights, loss_fn, v2, inputs[:1], targets[:1])
    hvp_v_sum = backend.compute_hvp_single(model, weights, loss_fn, v_sum, inputs[:1], targets[:1])

    assert_allclose(hvp_v_sum, hvp_v1 + hvp_v2)


def test_compute_hvp_batch_equals_sum_of_singles(backend):
    """Batched HVP should equal the sum of per-sample HVPs."""
    model = Sequential([Input(shape=(2,)), Dense(1, name='output')])
    weights = backend.get_model_weights(model)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    vector = [tf.ones_like(weight) for weight in weights]

    inputs = tf.constant([[1.0, 0.0], [0.5, -1.0]], dtype=tf.float32)
    targets = tf.constant([[1.0], [0.0]], dtype=tf.float32)

    hvp_batch = backend.compute_hvp_batch(model, weights, loss_fn, vector, inputs, targets)
    hvp_single_terms = [
        backend.compute_hvp_single(model, weights, loss_fn, vector, inputs[idx:idx + 1], targets[idx:idx + 1])
        for idx in range(inputs.shape[0])
    ]
    hvp_single_sum = tf.reduce_sum(tf.stack(hvp_single_terms, axis=0), axis=0)

    assert np.allclose(backend.to_numpy(hvp_batch), backend.to_numpy(hvp_single_sum), atol=1e-5, rtol=1e-5)


def test_compute_hvp_single_matches_hessian_vector_product(backend):
    """Single-sample HVP should match explicit Hessian-vector multiplication."""
    model = Sequential([Input(shape=(2,)), Dense(1, name='output')])
    weights = backend.get_model_weights(model)
    nb_params = backend.get_num_params(weights)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    inputs = tf.constant([[1.0, 0.0]], dtype=tf.float32)
    targets = tf.constant([[1.0]], dtype=tf.float32)
    vector = [
        tf.constant([[0.3], [-0.7]], dtype=tf.float32),
        tf.constant([0.5], dtype=tf.float32),
    ]

    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(1)
    hessian = backend.compute_hessian(model, weights, loss_fn, dataset, nb_params)
    hvp = backend.compute_hvp_single(model, weights, loss_fn, vector, inputs, targets)
    vector_flat = backend.concat([backend.reshape(v, (-1,)) for v in vector], axis=0)
    vector_col = backend.reshape(vector_flat, (-1, 1))
    hvp_from_hessian = backend.reshape(backend.matmul(hessian, vector_col), (-1,))

    assert hvp.shape == (nb_params,)
    assert np.allclose(backend.to_numpy(hvp), backend.to_numpy(hvp_from_hessian), atol=1e-5, rtol=1e-5)

def test_compute_output_jacobians(backend):
    """Test output Jacobian computations w.r.t inputs and weights."""
    model = Sequential([Input(shape=(2,)), Dense(1, name='output')])
    inputs = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    weights = backend.get_model_weights(model)

    outputs, jac_inputs = backend.compute_output_jacobian(model, inputs)
    outputs_w, jac_weights = backend.compute_output_jacobian_wrt_weights(model, weights, inputs)

    assert outputs.shape == (2, 1)
    assert outputs_w.shape == (2, 1)
    assert jac_inputs.shape == (2, 1, 2, 2)
    assert len(jac_weights) == len(weights)
    assert jac_weights[0].shape[:2] == (2, 1)
    assert allclose(outputs, outputs_w)
    assert bool(tf.reduce_all(tf.math.is_finite(jac_inputs)).numpy())
    assert all(bool(tf.reduce_all(tf.math.is_finite(jac)).numpy()) for jac in jac_weights)

def test_while_loop(backend):
    """Test while_loop helper with simple integer accumulation."""
    def cond_fn(i, total):
        return i < 3

    def body_fn(i, total):
        return [i + 1, total + i]

    result_i, result_total = backend.while_loop(
        cond_fn,
        body_fn,
        [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)],
        maximum_iterations=10,
    )

    assert int(result_i.numpy()) == 3
    assert int(result_total.numpy()) == 3

def test_random_diag_eig_and_real(backend):
    """Test random normal, diagonal extraction and eigen helpers."""
    random_tensor = backend.random_normal((2, 3), dtype=backend.float32_dtype())
    assert random_tensor.shape == (2, 3)

    matrix = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=tf.float32)
    diag = backend.diag_part(matrix, k=1)
    assert np.array_equal(diag.numpy(), np.array([2.0, 6.0]))

    maindiag = tf.constant([2.0, 3.0], dtype=tf.float32)
    superdiag = tf.constant([1.0], dtype=tf.float32)
    eig_vals, eig_vecs = backend.eigh_tridiagonal(maindiag, superdiag)
    eig_vals_only, eig_vecs_none = backend.eigh_tridiagonal(maindiag, superdiag, eigvals_only=True)

    assert eig_vals.shape == (2,)
    assert eig_vecs.shape == (2, 2)
    assert eig_vals_only.shape == (2,)
    assert eig_vecs_none is None

    eig_input = tf.constant([[0.0, -1.0], [1.0, 0.0]], dtype=tf.float32)
    eigvals, eigvecs = backend.eig(eig_input)
    real_part = backend.real(eigvals)

    assert eigvals.shape == (2,)
    assert eigvecs.shape == (2, 2)
    assert real_part.shape == (2,)


def test_svd_lowrank_shape(backend):
    """svd_lowrank returns tensors with the expected shapes."""
    m, n, rank = 10, 8, 3
    rng = np.random.default_rng(0)
    mat = tf.constant(rng.standard_normal((m, n)), dtype=tf.float32)
    u, s, vh = backend.svd_lowrank(mat, rank)
    assert tuple(u.shape) == (m, rank)
    assert tuple(s.shape) == (rank,)
    assert tuple(vh.shape) == (rank, n)


def test_svd_lowrank_reconstruction(backend):
    """Low-rank product U @ diag(S) @ Vh approximates the original matrix."""
    m, n, rank = 20, 15, 5
    rng = np.random.default_rng(42)
    # Build a rank-5 matrix so full reconstruction should be near-exact
    a = rng.standard_normal((m, rank)).astype(np.float32)
    b = rng.standard_normal((rank, n)).astype(np.float32)
    mat = tf.constant(a @ b)
    u, s, vh = backend.svd_lowrank(mat, rank)
    s_diag = tf.linalg.diag(s)
    reconstructed = u @ s_diag @ vh
    np.testing.assert_allclose(mat.numpy(), reconstructed.numpy(), atol=1e-4)


def test_svd_lowrank_singular_values_descending(backend):
    """Singular values are returned in descending order."""
    m, n, rank = 12, 10, 4
    rng = np.random.default_rng(7)
    mat = tf.constant(rng.standard_normal((m, n)), dtype=tf.float32)
    _, s, _ = backend.svd_lowrank(mat, rank)
    s_np = s.numpy()
    assert np.all(s_np[:-1] >= s_np[1:] - 1e-6), f"Singular values not descending: {s_np}"


def test_svd_lowrank_rank_clamp(backend):
    """Rank is clamped by min(m, n)."""
    m, n = 4, 3
    rank = 10  # larger than both dimensions
    mat = tf.constant(np.eye(m, n, dtype=np.float32))
    u, s, vh = backend.svd_lowrank(mat, rank)
    # TF full SVD returns min(m,n) singular values
    assert u.shape[1] <= min(m, n)


def test_svd_lowrank_orthonormal_columns(backend):
    """U returned by svd_lowrank should have orthonormal columns."""
    m, n, rank = 10, 8, 3
    rng = np.random.default_rng(5)
    mat = tf.constant(rng.standard_normal((m, n)), dtype=tf.float32)
    u, _, _ = backend.svd_lowrank(mat, rank)
    gram = tf.transpose(u) @ u
    np.testing.assert_allclose(gram.numpy(), np.eye(rank), atol=1e-5)


def test_svd_lowrank_vh_orthonormal_rows(backend):
    """Vh returned by svd_lowrank should have orthonormal rows (Vh @ Vh^T ≈ I)."""
    m, n, rank = 10, 8, 3
    rng = np.random.default_rng(13)
    mat = tf.constant(rng.standard_normal((m, n)), dtype=tf.float32)
    _, _, vh = backend.svd_lowrank(mat, rank)
    gram = vh @ tf.transpose(vh)
    np.testing.assert_allclose(gram.numpy(), np.eye(rank), atol=1e-5)


def test_svd_lowrank_tall_matrix(backend):
    """svd_lowrank works correctly for tall-skinny matrices (m >> n)."""
    m, n, rank = 50, 5, 3
    rng = np.random.default_rng(17)
    a = rng.standard_normal((m, rank)).astype(np.float32)
    b = rng.standard_normal((rank, n)).astype(np.float32)
    mat = tf.constant(a @ b)
    u, s, vh = backend.svd_lowrank(mat, rank)
    assert tuple(u.shape) == (m, rank)
    assert tuple(s.shape) == (rank,)
    assert tuple(vh.shape) == (rank, n)
    reconstructed = u @ tf.linalg.diag(s) @ vh
    np.testing.assert_allclose(mat.numpy(), reconstructed.numpy(), atol=1e-4)


def test_svd_lowrank_wide_matrix(backend):
    """svd_lowrank works correctly for wide matrices (m << n)."""
    m, n, rank = 5, 50, 3
    rng = np.random.default_rng(19)
    a = rng.standard_normal((m, rank)).astype(np.float32)
    b = rng.standard_normal((rank, n)).astype(np.float32)
    mat = tf.constant(a @ b)
    u, s, vh = backend.svd_lowrank(mat, rank)
    assert tuple(u.shape) == (m, rank)
    assert tuple(s.shape) == (rank,)
    assert tuple(vh.shape) == (rank, n)
    reconstructed = u @ tf.linalg.diag(s) @ vh
    np.testing.assert_allclose(mat.numpy(), reconstructed.numpy(), atol=1e-4)


def test_svd_lowrank_rank_one(backend):
    """svd_lowrank with rank=1 returns correct shapes and reconstructs a rank-1 matrix."""
    m, n = 8, 6
    rng = np.random.default_rng(23)
    a = rng.standard_normal((m, 1)).astype(np.float32)
    b = rng.standard_normal((1, n)).astype(np.float32)
    mat = tf.constant(a @ b)
    u, s, vh = backend.svd_lowrank(mat, rank=1)
    assert tuple(u.shape) == (m, 1)
    assert tuple(s.shape) == (1,)
    assert tuple(vh.shape) == (1, n)
    reconstructed = u @ tf.linalg.diag(s) @ vh
    np.testing.assert_allclose(mat.numpy(), reconstructed.numpy(), atol=1e-4)


def test_svd_lowrank_zero_matrix(backend):
    """svd_lowrank on an all-zeros matrix returns zero singular values without errors."""
    m, n, rank = 6, 5, 3
    mat = tf.zeros((m, n), dtype=tf.float32)
    u, s, vh = backend.svd_lowrank(mat, rank)
    assert tuple(u.shape) == (m, rank)
    assert tuple(s.shape) == (rank,)
    assert tuple(vh.shape) == (rank, n)
    np.testing.assert_allclose(s.numpy(), np.zeros(rank), atol=1e-6)


def test_svd_lowrank_singular_values_nonnegative(backend):
    """Singular values returned by svd_lowrank are non-negative."""
    m, n, rank = 12, 9, 5
    rng = np.random.default_rng(29)
    mat = tf.constant(rng.standard_normal((m, n)), dtype=tf.float32)
    _, s, _ = backend.svd_lowrank(mat, rank)
    assert np.all(s.numpy() >= 0.0), f"Negative singular values found: {s.numpy()}"


def test_einsum_matmul(backend):
    """einsum reproduces matmul."""
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    result = backend.einsum("ij,jk->ik", a, b)
    expected = a @ b
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_dot(backend):
    """einsum computes a dot product."""
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    result = backend.einsum("i,i->", a, b)
    np.testing.assert_allclose(result.numpy(), np.dot(a.numpy(), b.numpy()))


def test_einsum_three_operand(backend):
    """einsum handles three operands as used in _einsum_low_rank."""
    # equation: "qor,toi,qri->qt"
    # q=2, o=3, r=4, t=5, i=6
    rng = np.random.default_rng(11)
    left = tf.constant(rng.standard_normal((2, 3, 4)), dtype=tf.float32)
    train = tf.constant(rng.standard_normal((5, 3, 6)), dtype=tf.float32)
    right = tf.constant(rng.standard_normal((2, 4, 6)), dtype=tf.float32)
    result = backend.einsum("qor,toi,qri->qt", left, train, right)
    assert result.shape == (2, 5)
    # Verify against numpy reference
    expected = np.einsum("qor,toi,qri->qt", left.numpy(), train.numpy(), right.numpy())
    # float32 multi-operand einsum: TF and NumPy may choose different contraction
    # orders, causing accumulation differences larger than single-op float32 epsilon.
    np.testing.assert_allclose(result.numpy(), expected, atol=5e-3)
