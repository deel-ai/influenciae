# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests to verify that PyTorch and TensorFlow backend implementations produce matching results.
These tests ensure framework-agnostic behavior of the influence function computations.
"""
import numpy as np
import pytest
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential as TFSequential
from tensorflow.keras.losses import MeanSquaredError, Reduction

from ..utils_test import allclose


pytestmark = pytest.mark.requires_both_backends

# Tolerance tiers used in parity checks:
# - 1e-6..1e-5 for deterministic algebraic helpers and eig utilities.
# - 1e-4 (default allclose epsilon) for basic tensor operations.
# - 5e-3 for cross-backend gradient/Jacobian/HVP/Hessian parity after TF/PT
#   parameter-layout remapping in float32.


def _assert_float32_dtype(tf_backend, pt_backend, tf_tensor, pt_tensor):
    """Assert both tensors preserve float32 dtype."""
    assert tf_backend.get_dtype(tf_tensor) == tf_backend.float32_dtype()
    assert pt_backend.get_dtype(pt_tensor) == pt_backend.float32_dtype()


def _tf_to_pt_flat_parameter_index(tf_weights, pt_weights):
    """Build a TF-order to PT-order flat index mapping for aligned model weights."""
    if len(tf_weights) != len(pt_weights):
        raise AssertionError("Mismatched number of watched weights")

    mapping = []
    pt_offset = 0
    total_tf_params = 0

    for tf_weight, pt_weight in zip(tf_weights, pt_weights):
        tf_shape = tuple(int(dim) for dim in tf_weight.shape)
        pt_shape = tuple(int(dim) for dim in pt_weight.shape)

        tf_size = int(np.prod(tf_shape, dtype=np.int64))
        pt_size = int(np.prod(pt_shape, dtype=np.int64))
        if tf_size != pt_size:
            raise AssertionError(f"Weight size mismatch between TF {tf_shape} and PT {pt_shape}")

        if tf_shape == pt_shape:
            mapping.extend(range(pt_offset, pt_offset + tf_size))
        elif len(tf_shape) == 2 and pt_shape == (tf_shape[1], tf_shape[0]):
            rows, cols = tf_shape
            for row in range(rows):
                for col in range(cols):
                    mapping.append(pt_offset + col * rows + row)
        else:
            raise AssertionError(f"Unsupported TF/PT shape pair: {tf_shape} vs {pt_shape}")

        pt_offset += pt_size
        total_tf_params += tf_size

    mapping = np.asarray(mapping, dtype=np.int64)
    if mapping.shape[0] != total_tf_params:
        raise AssertionError("Invalid TF/PT index mapping length")
    return mapping


class SimpleLinearModelTF:
    """Create a simple TensorFlow model for testing."""

    @staticmethod
    def create(input_dim=5, hidden_dim=3, output_dim=2, weights=None):
        """
        Create a simple 2-layer linear model.

        Parameters
        ----------
        input_dim : int
            Input dimension
        hidden_dim : int
            Hidden layer dimension
        output_dim : int
            Output dimension
        weights : dict, optional
            Dictionary with 'w1', 'b1', 'w2', 'b2' numpy arrays for initialization
        """
        model = TFSequential([
            Input(shape=(input_dim,)),
            Dense(hidden_dim, activation=None, name='hidden'),
            Dense(output_dim, activation=None, name='output')
        ])

        if weights is not None:
            model.layers[0].set_weights([weights['w1'], weights['b1']])
            model.layers[1].set_weights([weights['w2'], weights['b2']])

        return model


class SimpleLinearModelPT:
    """Create a simple PyTorch model for testing."""

    @staticmethod
    def create(input_dim=5, hidden_dim=3, output_dim=2, weights=None):
        """
        Create a simple 2-layer linear model.

        Parameters
        ----------
        input_dim : int
            Input dimension
        hidden_dim : int
            Hidden layer dimension
        output_dim : int
            Output dimension
        weights : dict, optional
            Dictionary with 'w1', 'b1', 'w2', 'b2' numpy arrays for initialization
        """
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        if weights is not None:
            with torch.no_grad():
                # PyTorch uses transposed weight matrices compared to TF
                model[0].weight.copy_(torch.from_numpy(weights['w1'].T))
                model[0].bias.copy_(torch.from_numpy(weights['b1']))
                model[1].weight.copy_(torch.from_numpy(weights['w2'].T))
                model[1].bias.copy_(torch.from_numpy(weights['b2']))

        return model


def generate_matching_weights(input_dim=5, hidden_dim=3, output_dim=2, seed=42):
    """Generate random weights that can be used in both frameworks."""
    np.random.seed(seed)
    return {
        'w1': np.random.randn(input_dim, hidden_dim).astype(np.float32),
        'b1': np.random.randn(hidden_dim).astype(np.float32),
        'w2': np.random.randn(hidden_dim, output_dim).astype(np.float32),
        'b2': np.random.randn(output_dim).astype(np.float32),
    }


def generate_test_data(batch_size=4, input_dim=5, output_dim=2, seed=123):
    """Generate test input and target data."""
    np.random.seed(seed)
    inputs = np.random.randn(batch_size, input_dim).astype(np.float32)
    targets = np.random.randn(batch_size, output_dim).astype(np.float32)
    return inputs, targets


def test_detect_tensorflow_model():
    """Test that TensorFlow models are correctly detected."""
    from deel.influenciae.common import detect_framework, Framework

    model = SimpleLinearModelTF.create()
    assert detect_framework(model) == Framework.TENSORFLOW

def test_detect_pytorch_model():
    """Test that PyTorch models are correctly detected."""
    from deel.influenciae.common import detect_framework, Framework

    model = SimpleLinearModelPT.create()
    assert detect_framework(model) == Framework.PYTORCH

def test_detect_invalid_model():
    """Test that invalid models raise ValueError."""
    from deel.influenciae.common import detect_framework

    with pytest.raises(ValueError):
        detect_framework("not a model")

    with pytest.raises(ValueError):
        detect_framework(42)


def test_get_num_params_match():
    """Test that parameter counting matches between frameworks."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_weights = tf_backend.get_model_weights(tf_model)
    pt_weights = pt_backend.get_model_weights(pt_model)

    tf_num_params = tf_backend.get_num_params(tf_weights)
    pt_num_params = pt_backend.get_num_params(pt_weights)

    # Expected: (5*3 + 3) + (3*2 + 2) = 18 + 8 = 26
    expected_params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

    assert tf_num_params == expected_params
    assert pt_num_params == expected_params
    assert tf_num_params == pt_num_params

def test_forward_pass_match():
    """Test that forward pass produces matching outputs."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)
    inputs, _ = generate_test_data(batch_size=4, input_dim=input_dim, output_dim=output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_inputs = tf.constant(inputs)
    pt_inputs = torch.from_numpy(inputs)

    tf_output = tf_backend.forward(tf_model, tf_inputs).numpy()
    pt_output = pt_backend.forward(pt_model, pt_inputs).detach().numpy()

    assert allclose(tf_output, pt_output), \
        f"Forward pass mismatch:\nTF: {tf_output}\nPT: {pt_output}"

def test_loss_computation_match():
    """Test that loss computation produces matching results."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)
    inputs, targets = generate_test_data(batch_size=4, input_dim=input_dim, output_dim=output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    # Create matching loss functions (MSE without reduction)
    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    pt_loss_fn = nn.MSELoss(reduction='none')

    tf_inputs = tf.constant(inputs)
    tf_targets = tf.constant(targets)
    pt_inputs = torch.from_numpy(inputs)
    pt_targets = torch.from_numpy(targets)

    tf_loss = tf_backend.compute_loss(tf_model, tf_loss_fn, tf_inputs, tf_targets).numpy()
    pt_loss = pt_backend.compute_loss(pt_model, pt_loss_fn, pt_inputs, pt_targets).detach().numpy()

    # MSE loss per sample - need to sum over output dimensions for comparison
    # TF MSE returns shape (batch,) while PT returns shape (batch, output_dim)
    if len(pt_loss.shape) > 1:
        pt_loss = pt_loss.mean(axis=-1)

    assert allclose(tf_loss, pt_loss), \
        f"Loss mismatch:\nTF: {tf_loss}\nPT: {pt_loss}"

def test_gradient_computation_match():
    """Test that gradient computation produces matching results."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)
    inputs, targets = generate_test_data(batch_size=4, input_dim=input_dim, output_dim=output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_weights = tf_backend.get_model_weights(tf_model)
    pt_weights = pt_backend.get_model_weights(pt_model)

    # Create matching loss functions
    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    pt_loss_fn = nn.MSELoss(reduction='none')

    # Wrapper to sum PT loss
    def pt_loss_wrapper(pred, target):
        return pt_loss_fn(pred, target).mean(dim=-1)

    tf_inputs = tf.constant(inputs)
    tf_targets = tf.constant(targets)
    pt_inputs = torch.from_numpy(inputs)
    pt_targets = torch.from_numpy(targets)

    tf_grad = tf_backend.compute_gradient(
        tf_model, tf_weights, tf_loss_fn, tf_inputs, tf_targets
    ).numpy()

    pt_grad = pt_backend.compute_gradient(
        pt_model, pt_weights, pt_loss_wrapper, pt_inputs, pt_targets
    ).detach().numpy()

    # Check gradient shapes match
    assert tf_grad.shape == pt_grad.shape, \
        f"Gradient shape mismatch: TF {tf_grad.shape} vs PT {pt_grad.shape}"

    # Compare in TF parameter order; PT kernels are laid out transposed.
    assert np.all(np.isfinite(tf_grad)), "TF gradient has non-finite values"
    assert np.all(np.isfinite(pt_grad)), "PT gradient has non-finite values"
    assert np.linalg.norm(tf_grad) > 0, "TF gradient is zero"
    assert np.linalg.norm(pt_grad) > 0, "PT gradient is zero"
    # First-order parity uses a moderate tolerance in float32.
    assert allclose(np.linalg.norm(tf_grad), np.linalg.norm(pt_grad), epsilon=5e-3)

    tf_to_pt = _tf_to_pt_flat_parameter_index(tf_weights, pt_weights)
    pt_grad_tf_order = pt_grad[tf_to_pt]
    assert allclose(tf_grad, pt_grad_tf_order, epsilon=5e-3)

    tf_abs_sorted = np.sort(np.abs(tf_grad))
    pt_abs_sorted = np.sort(np.abs(pt_grad))
    assert allclose(tf_abs_sorted, pt_abs_sorted, epsilon=5e-3)


def test_influence_model_creation():
    """Test that InfluenceModel can be created for both frameworks."""
    from deel.influenciae.common import InfluenceModel

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    pt_loss_fn = nn.MSELoss(reduction='none')

    tf_influence = InfluenceModel(tf_model, loss_function=tf_loss_fn)
    pt_influence = InfluenceModel(pt_model, loss_function=pt_loss_fn)

    # Check parameter counts match
    assert tf_influence.nb_params == pt_influence.nb_params

def test_influence_model_forward():
    """Test that InfluenceModel forward pass matches."""
    from deel.influenciae.common import InfluenceModel

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)
    inputs, _ = generate_test_data(batch_size=4, input_dim=input_dim, output_dim=output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    pt_loss_fn = nn.MSELoss(reduction='none')

    tf_influence = InfluenceModel(tf_model, loss_function=tf_loss_fn)
    pt_influence = InfluenceModel(pt_model, loss_function=pt_loss_fn)

    tf_output = tf_influence(tf.constant(inputs)).numpy()
    pt_output = pt_influence(torch.from_numpy(inputs)).detach().numpy()

    assert allclose(tf_output, pt_output), \
        f"InfluenceModel forward mismatch:\nTF: {tf_output}\nPT: {pt_output}"

def test_influence_model_batch_loss():
    """Test that InfluenceModel batch_loss matches."""
    from deel.influenciae.common import InfluenceModel

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)
    inputs, targets = generate_test_data(batch_size=4, input_dim=input_dim, output_dim=output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    pt_loss_fn = nn.MSELoss(reduction='none')

    # Wrapper for PT to match TF behavior
    def pt_loss_wrapper(pred, target):
        return pt_loss_fn(pred, target).mean(dim=-1)

    tf_influence = InfluenceModel(tf_model, loss_function=tf_loss_fn)
    pt_influence = InfluenceModel(pt_model, loss_function=pt_loss_wrapper)

    # Create datasets
    tf_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)
    pt_dataset = [(torch.from_numpy(inputs[:2]), torch.from_numpy(targets[:2])),
                  (torch.from_numpy(inputs[2:]), torch.from_numpy(targets[2:]))]

    tf_loss = tf_influence.batch_loss(tf_dataset).numpy()
    pt_loss = pt_influence.backend.to_numpy(pt_influence.batch_loss(pt_dataset))

    assert allclose(tf_loss, pt_loss), \
        f"InfluenceModel batch_loss mismatch:\nTF: {tf_loss}\nPT: {pt_loss}"


def test_influence_model_batch_jacobian_match():
    """Test that InfluenceModel batch_jacobian matches across backends."""
    from deel.influenciae.common import InfluenceModel

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)
    inputs, targets = generate_test_data(batch_size=4, input_dim=input_dim, output_dim=output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    pt_loss_fn = nn.MSELoss(reduction='none')

    def pt_loss_wrapper(pred, target):
        return pt_loss_fn(pred, target).mean(dim=-1)

    tf_influence = InfluenceModel(tf_model, loss_function=tf_loss_fn)
    pt_influence = InfluenceModel(pt_model, loss_function=pt_loss_wrapper)

    tf_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)
    pt_dataset = [
        (torch.from_numpy(inputs[:2]), torch.from_numpy(targets[:2])),
        (torch.from_numpy(inputs[2:]), torch.from_numpy(targets[2:])),
    ]

    tf_jacobian = tf_influence.backend.to_numpy(tf_influence.batch_jacobian(tf_dataset))
    pt_jacobian = pt_influence.backend.to_numpy(pt_influence.batch_jacobian(pt_dataset))

    assert tf_jacobian.shape == pt_jacobian.shape

    tf_to_pt = _tf_to_pt_flat_parameter_index(tf_influence.weights, pt_influence.weights)
    pt_jacobian_tf_order = pt_jacobian[:, tf_to_pt]
    assert allclose(tf_jacobian, pt_jacobian_tf_order, epsilon=5e-3)

def test_influence_model_layer_targeting():
    """Test that layer targeting works consistently."""
    from deel.influenciae.common import InfluenceModel

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    pt_loss_fn = nn.MSELoss(reduction='none')

    # Target first layer only
    tf_influence_layer0 = InfluenceModel(tf_model, start_layer=0, loss_function=tf_loss_fn)
    pt_influence_layer0 = InfluenceModel(pt_model, start_layer=0, loss_function=pt_loss_fn)

    # First layer: hidden_dim weights + hidden_dim bias = input_dim * hidden_dim + hidden_dim
    expected_layer0_params = input_dim * hidden_dim + hidden_dim

    assert tf_influence_layer0.nb_params == expected_layer0_params
    assert pt_influence_layer0.nb_params == expected_layer0_params

    # Target last layer only (default behavior)
    tf_influence_default = InfluenceModel(tf_model, loss_function=tf_loss_fn)
    pt_influence_default = InfluenceModel(pt_model, loss_function=pt_loss_fn)

    # Last layer: output_dim weights + output_dim bias = hidden_dim * output_dim + output_dim
    expected_last_layer_params = hidden_dim * output_dim + output_dim

    assert tf_influence_default.nb_params == expected_last_layer_params
    assert pt_influence_default.nb_params == expected_last_layer_params


def test_concat():
    """Test tensor concatenation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    tf_result = tf_backend.concat([tf.constant(a), tf.constant(b)], axis=0).numpy()
    pt_result = pt_backend.concat([torch.from_numpy(a), torch.from_numpy(b)], axis=0).numpy()

    assert allclose(tf_result, pt_result)

def test_stack():
    """Test tensor stacking."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)

    tf_result = tf_backend.stack([tf.constant(a), tf.constant(b)], axis=0).numpy()
    pt_result = pt_backend.stack([torch.from_numpy(a), torch.from_numpy(b)], axis=0).numpy()

    assert allclose(tf_result, pt_result)

def test_reshape():
    """Test tensor reshape."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    tf_result = tf_backend.reshape(tf.constant(a), (3, 2)).numpy()
    pt_result = pt_backend.reshape(torch.from_numpy(a), (3, 2)).numpy()

    assert allclose(tf_result, pt_result)

def test_reduce_sum():
    """Test reduce sum."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    # Full reduction
    tf_result_full_tensor = tf_backend.reduce_sum(tf.constant(a))
    pt_result_full_tensor = pt_backend.reduce_sum(torch.from_numpy(a))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_full_tensor, pt_result_full_tensor)

    tf_result_full = tf_result_full_tensor.numpy()
    pt_result_full = pt_result_full_tensor.numpy()
    assert allclose(tf_result_full, pt_result_full)

    # Axis reduction
    tf_result_axis_tensor = tf_backend.reduce_sum(tf.constant(a), axis=1)
    pt_result_axis_tensor = pt_backend.reduce_sum(torch.from_numpy(a), axis=1)
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_axis_tensor, pt_result_axis_tensor)

    tf_result_axis = tf_result_axis_tensor.numpy()
    pt_result_axis = pt_result_axis_tensor.numpy()
    assert allclose(tf_result_axis, pt_result_axis)

def test_to_numpy():
    """Test conversion to numpy."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    tf_result = tf_backend.to_numpy(tf.constant(a))
    pt_result = pt_backend.to_numpy(torch.from_numpy(a))

    assert isinstance(tf_result, np.ndarray)
    assert isinstance(pt_result, np.ndarray)
    assert allclose(tf_result, pt_result)

def test_expand_dims():
    """Test expanding tensor dimensions."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1, 2, 3], dtype=np.float32)

    # Expand at axis 0
    tf_result = tf_backend.expand_dims(tf.constant(a), axis=0).numpy()
    pt_result = pt_backend.expand_dims(torch.from_numpy(a), axis=0).numpy()
    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)

    # Expand at axis 1
    tf_result = tf_backend.expand_dims(tf.constant(a), axis=1).numpy()
    pt_result = pt_backend.expand_dims(torch.from_numpy(a), axis=1).numpy()
    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)

def test_squeeze():
    """Test squeezing tensor dimensions."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[[1, 2, 3]]], dtype=np.float32)  # Shape (1, 1, 3)

    # Squeeze all dimensions of size 1
    tf_result = tf_backend.squeeze(tf.constant(a)).numpy()
    pt_result = pt_backend.squeeze(torch.from_numpy(a)).numpy()
    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)

    # Squeeze specific axis
    tf_result = tf_backend.squeeze(tf.constant(a), axis=0).numpy()
    pt_result = pt_backend.squeeze(torch.from_numpy(a), axis=0).numpy()
    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)

def test_transpose():
    """Test tensor transpose."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # Shape (2, 3)

    tf_result = tf_backend.transpose(tf.constant(a)).numpy()
    pt_result = pt_backend.transpose(torch.from_numpy(a)).numpy()
    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)

def test_matmul():
    """Test matrix multiplication."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    tf_result_tensor = tf_backend.matmul(tf.constant(a), tf.constant(b))
    pt_result_tensor = pt_backend.matmul(torch.from_numpy(a), torch.from_numpy(b))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_tensor, pt_result_tensor)

    tf_result = tf_result_tensor.numpy()
    pt_result = pt_result_tensor.numpy()
    assert allclose(tf_result, pt_result)

def test_multiply():
    """Test element-wise multiplication."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    tf_result_tensor = tf_backend.multiply(tf.constant(a), tf.constant(b))
    pt_result_tensor = pt_backend.multiply(torch.from_numpy(a), torch.from_numpy(b))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_tensor, pt_result_tensor)

    tf_result = tf_result_tensor.numpy()
    pt_result = pt_result_tensor.numpy()
    assert allclose(tf_result, pt_result)

def test_abs():
    """Test absolute value."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([-1, 2, -3, 4], dtype=np.float32)

    tf_result_tensor = tf_backend.abs(tf.constant(a))
    pt_result_tensor = pt_backend.abs(torch.from_numpy(a))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_tensor, pt_result_tensor)

    tf_result = tf_result_tensor.numpy()
    pt_result = pt_result_tensor.numpy()
    assert allclose(tf_result, pt_result)

def test_argmax():
    """Test argmax operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 5, 3], [4, 2, 6]], dtype=np.float32)

    # Argmax along axis 1
    tf_result = tf_backend.argmax(tf.constant(a), axis=1).numpy()
    pt_result = pt_backend.argmax(torch.from_numpy(a), axis=1).numpy()
    assert allclose(tf_result, pt_result)

def test_argmin():
    """Test argmin operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 5, 3], [4, 2, 6]], dtype=np.float32)

    # Argmin along axis 1
    tf_result = tf_backend.argmin(tf.constant(a), axis=1).numpy()
    pt_result = pt_backend.argmin(torch.from_numpy(a), axis=1).numpy()
    assert allclose(tf_result, pt_result)

def test_reduce_mean():
    """Test reduce mean."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    # Full reduction
    tf_result_full_tensor = tf_backend.reduce_mean(tf.constant(a))
    pt_result_full_tensor = pt_backend.reduce_mean(torch.from_numpy(a))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_full_tensor, pt_result_full_tensor)

    tf_result_full = tf_result_full_tensor.numpy()
    pt_result_full = pt_result_full_tensor.numpy()
    assert allclose(tf_result_full, pt_result_full)

    # Axis reduction
    tf_result_axis_tensor = tf_backend.reduce_mean(tf.constant(a), axis=1)
    pt_result_axis_tensor = pt_backend.reduce_mean(torch.from_numpy(a), axis=1)
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_axis_tensor, pt_result_axis_tensor)

    tf_result_axis = tf_result_axis_tensor.numpy()
    pt_result_axis = pt_result_axis_tensor.numpy()
    assert allclose(tf_result_axis, pt_result_axis)

    # Axis reduction with keepdims
    tf_result_keepdims_tensor = tf_backend.reduce_mean(tf.constant(a), axis=1, keepdims=True)
    pt_result_keepdims_tensor = pt_backend.reduce_mean(torch.from_numpy(a), axis=1, keepdims=True)
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_keepdims_tensor, pt_result_keepdims_tensor)

    tf_result_keepdims = tf_result_keepdims_tensor.numpy()
    pt_result_keepdims = pt_result_keepdims_tensor.numpy()
    assert tf_result_keepdims.shape == pt_result_keepdims.shape
    assert allclose(tf_result_keepdims, pt_result_keepdims)


def test_zeros():
    """Test creating zeros tensor."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    shape = (3, 4)

    tf_result = tf_backend.zeros(shape).numpy()
    pt_result = pt_backend.zeros(shape).numpy()

    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)

def test_zeros_like():
    """Test creating zeros tensor with same shape as input."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    tf_result = tf_backend.zeros_like(tf.constant(a)).numpy()
    pt_result = pt_backend.zeros_like(torch.from_numpy(a)).numpy()

    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)
    assert np.all(tf_result == 0)
    assert np.all(pt_result == 0)

def test_sqrt():
    """Test square root operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1, 4, 9, 16], dtype=np.float32)

    tf_result_tensor = tf_backend.sqrt(tf.constant(a))
    pt_result_tensor = pt_backend.sqrt(torch.from_numpy(a))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_tensor, pt_result_tensor)

    tf_result = tf_result_tensor.numpy()
    pt_result = pt_result_tensor.numpy()
    assert allclose(tf_result, pt_result)

def test_maximum():
    """Test element-wise maximum."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1, 5, 3], dtype=np.float32)
    b = np.array([2, 4, 4], dtype=np.float32)

    tf_result = tf_backend.maximum(tf.constant(a), tf.constant(b)).numpy()
    pt_result = pt_backend.maximum(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    assert allclose(tf_result, pt_result)

def test_norm():
    """Test norm computation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)

    # Frobenius norm (default, flattened)
    tf_result_tensor = tf_backend.norm(tf.constant(a))
    pt_result_tensor = pt_backend.norm(torch.from_numpy(a))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_tensor, pt_result_tensor)

    tf_result = tf_result_tensor.numpy()
    pt_result = pt_result_tensor.numpy()
    assert allclose(tf_result, pt_result)

    # Norm along axis
    tf_result_axis_tensor = tf_backend.norm(tf.constant(a), axis=1)
    pt_result_axis_tensor = pt_backend.norm(torch.from_numpy(a), axis=1)
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_axis_tensor, pt_result_axis_tensor)

    tf_result_axis = tf_result_axis_tensor.numpy()
    pt_result_axis = pt_result_axis_tensor.numpy()
    assert allclose(tf_result_axis, pt_result_axis)

def test_pinv():
    """Test pseudo-inverse computation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    # Create a simple matrix
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    tf_result_tensor = tf_backend.pinv(tf.constant(a))
    pt_result_tensor = pt_backend.pinv(torch.from_numpy(a))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_tensor, pt_result_tensor)

    tf_result = tf_result_tensor.numpy()
    pt_result = pt_result_tensor.numpy()

    # Verify A @ pinv(A) @ A ≈ A
    tf_verify = np.matmul(np.matmul(a, tf_result), a)
    pt_verify = np.matmul(np.matmul(a, pt_result), a)

    assert allclose(tf_verify, a, epsilon=1e-3)
    assert allclose(pt_verify, a, epsilon=1e-3)

def test_normalize():
    """Test normalization."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([3, 4], dtype=np.float32)  # norm = 5

    tf_result_tensor = tf_backend.normalize(tf.constant(a))
    pt_result_tensor = pt_backend.normalize(torch.from_numpy(a))
    _assert_float32_dtype(tf_backend, pt_backend, tf_result_tensor, pt_result_tensor)

    tf_result = tf_result_tensor.numpy()
    pt_result = pt_result_tensor.numpy()

    # Both should have unit norm
    assert allclose(np.linalg.norm(tf_result), 1.0)
    assert allclose(np.linalg.norm(pt_result), 1.0)
    assert allclose(tf_result, pt_result)


def test_arange():
    """Test arange operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    tf_result = tf_backend.arange(0, 10).numpy()
    pt_result = pt_backend.arange(0, 10).numpy()
    assert allclose(tf_result, pt_result)

def test_tile():
    """Test tile operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)

    tf_result = tf_backend.tile(tf.constant(a), (2, 3)).numpy()
    pt_result = pt_backend.tile(torch.from_numpy(a), (2, 3)).numpy()

    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)

def test_repeat():
    """Test repeat operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1, 2, 3], dtype=np.float32)

    tf_result = tf_backend.repeat(tf.constant(a), 3, axis=0).numpy()
    pt_result = pt_backend.repeat(torch.from_numpy(a), 3, axis=0).numpy()

    assert tf_result.shape == pt_result.shape
    assert allclose(tf_result, pt_result)

def test_sign():
    """Test sign operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([-2, -1, 0, 1, 2], dtype=np.float32)

    tf_result = tf_backend.sign(tf.constant(a)).numpy()
    pt_result = pt_backend.sign(torch.from_numpy(a)).numpy()
    assert allclose(tf_result, pt_result)

def test_pow():
    """Test power operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1, 2, 3, 4], dtype=np.float32)

    tf_result = tf_backend.pow(tf.constant(a), 2).numpy()
    pt_result = pt_backend.pow(torch.from_numpy(a), 2).numpy()
    assert allclose(tf_result, pt_result)

def test_top_k():
    """Test top_k operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float32)

    tf_values, tf_indices = tf_backend.top_k(tf.constant(a), k=3)
    pt_values, pt_indices = pt_backend.top_k(torch.from_numpy(a), k=3)

    tf_values = tf_values.numpy()
    pt_values = pt_values.numpy()
    tf_indices = tf_indices.numpy()
    pt_indices = pt_indices.numpy()

    # Values should match (sorted descending)
    assert allclose(tf_values, pt_values)
    assert np.array_equal(tf_indices, pt_indices)


def test_top_k_invalid_k_raises():
    """Both backends should raise when requesting too many top-k elements."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([3, 1, 4], dtype=np.float32)

    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        tf_backend.top_k(tf.constant(a), k=4)

    with pytest.raises((RuntimeError, ValueError)):
        pt_backend.top_k(torch.from_numpy(a), k=4)


def test_diag_part_eigh_tridiagonal_and_eig_real_parity():
    """Parity checks for diag_part, eigh_tridiagonal, eig and real helpers."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    tf_diag = tf_backend.diag_part(tf.constant(matrix), k=1).numpy()
    pt_diag = pt_backend.diag_part(torch.from_numpy(matrix), k=1).numpy()
    assert np.array_equal(tf_diag, pt_diag)

    maindiag = np.array([2.0, 3.0], dtype=np.float32)
    superdiag = np.array([1.0], dtype=np.float32)

    tf_eig_vals, tf_eig_vecs = tf_backend.eigh_tridiagonal(tf.constant(maindiag), tf.constant(superdiag))
    pt_eig_vals, pt_eig_vecs = pt_backend.eigh_tridiagonal(torch.from_numpy(maindiag), torch.from_numpy(superdiag))

    tf_eig_vals = tf_backend.to_numpy(tf_eig_vals)
    pt_eig_vals = pt_backend.to_numpy(pt_eig_vals)
    tf_eig_vecs = tf_backend.to_numpy(tf_eig_vecs)
    pt_eig_vecs = pt_backend.to_numpy(pt_eig_vecs)

    assert allclose(tf_eig_vals, pt_eig_vals, epsilon=1e-5)
    assert allclose(np.abs(tf_eig_vecs), np.abs(pt_eig_vecs), epsilon=1e-5)

    tf_vals_only, tf_vecs_none = tf_backend.eigh_tridiagonal(
        tf.constant(maindiag), tf.constant(superdiag), eigvals_only=True
    )
    pt_vals_only, pt_vecs_none = pt_backend.eigh_tridiagonal(
        torch.from_numpy(maindiag), torch.from_numpy(superdiag), eigvals_only=True
    )

    assert tf_vecs_none is None
    assert pt_vecs_none is None
    assert allclose(tf_backend.to_numpy(tf_vals_only), pt_backend.to_numpy(pt_vals_only), epsilon=1e-5)

    eig_input = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    tf_eigvals, _ = tf_backend.eig(tf.constant(eig_input))
    pt_eigvals, _ = pt_backend.eig(torch.from_numpy(eig_input))

    tf_real = tf_backend.real(tf_eigvals).numpy()
    pt_real = pt_backend.real(pt_eigvals).numpy()
    assert allclose(np.sort(tf_real), np.sort(pt_real), epsilon=1e-6)

    tf_eigvals_np = tf_backend.to_numpy(tf_eigvals)
    pt_eigvals_np = pt_backend.to_numpy(pt_eigvals)
    assert allclose(
        np.sort(np.abs(np.imag(tf_eigvals_np))),
        np.sort(np.abs(np.imag(pt_eigvals_np))),
        epsilon=1e-6,
    )


def test_constant():
    """Test constant tensor creation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    value = [1.0, 2.0, 3.0]

    tf_result = tf_backend.constant(value).numpy()
    pt_result = pt_backend.constant(value).numpy()
    assert allclose(tf_result, pt_result)

def test_convert_to_tensor():
    """Test conversion to tensor."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    tf_result = tf_backend.convert_to_tensor(a).numpy()
    pt_result = pt_backend.convert_to_tensor(a).numpy()
    assert allclose(tf_result, pt_result)

def test_cast():
    """Test casting tensors."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1.5, 2.7, 3.2], dtype=np.float32)

    # Cast to int32
    tf_result_tensor = tf_backend.cast(tf.constant(a), tf_backend.int32_dtype())
    pt_result_tensor = pt_backend.cast(torch.from_numpy(a), pt_backend.int32_dtype())

    assert tf_backend.get_dtype(tf_result_tensor) == tf_backend.int32_dtype()
    assert pt_backend.get_dtype(pt_result_tensor) == pt_backend.int32_dtype()

    tf_result = tf_result_tensor.numpy()
    pt_result = pt_result_tensor.numpy()
    assert allclose(tf_result, pt_result)

def test_reduce_prod():
    """Test reduce product."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)

    # Full reduction
    tf_result = tf_backend.reduce_prod(tf.constant(a)).numpy()
    pt_result = pt_backend.reduce_prod(torch.from_numpy(a)).numpy()
    assert allclose(tf_result, pt_result)

    # Axis reduction
    tf_result_axis = tf_backend.reduce_prod(tf.constant(a), axis=1).numpy()
    pt_result_axis = pt_backend.reduce_prod(torch.from_numpy(a), axis=1).numpy()
    assert allclose(tf_result_axis, pt_result_axis)


def test_logical_and():
    """Test logical AND operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])

    tf_result = tf_backend.logical_and(tf.constant(a), tf.constant(b)).numpy()
    pt_result = pt_backend.logical_and(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    assert np.array_equal(tf_result, pt_result)

def test_reduce_any():
    """Test reduce any (logical OR) operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[True, False], [False, False]])

    # Full reduction
    tf_result = tf_backend.reduce_any(tf.constant(a)).numpy()
    pt_result = pt_backend.reduce_any(torch.from_numpy(a)).numpy()
    assert tf_result == pt_result

    # Axis reduction
    tf_result_axis = tf_backend.reduce_any(tf.constant(a), axis=1).numpy()
    pt_result_axis = pt_backend.reduce_any(torch.from_numpy(a), axis=1).numpy()
    assert np.array_equal(tf_result_axis, pt_result_axis)


def test_jacobian_shape_match():
    """Test that Jacobian computation produces matching shapes."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    batch_size = 4
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)
    inputs, targets = generate_test_data(batch_size=batch_size, input_dim=input_dim, output_dim=output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_weights = tf_backend.get_model_weights(tf_model)
    pt_weights = pt_backend.get_model_weights(pt_model)

    # Create matching loss functions
    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    pt_loss_fn = nn.MSELoss(reduction='none')

    def pt_loss_wrapper(pred, target):
        return pt_loss_fn(pred, target).mean(dim=-1)

    tf_inputs = tf.constant(inputs)
    tf_targets = tf.constant(targets)
    pt_inputs = torch.from_numpy(inputs)
    pt_targets = torch.from_numpy(targets)

    tf_jacobian = tf_backend.compute_jacobian(
        tf_model, tf_weights, tf_loss_fn, tf_inputs, tf_targets
    ).numpy()

    pt_jacobian = pt_backend.compute_jacobian(
        pt_model, pt_weights, pt_loss_wrapper, pt_inputs, pt_targets
    ).detach().numpy()

    # Check shapes match (batch_size, num_params)
    assert tf_jacobian.shape == pt_jacobian.shape, \
        f"Jacobian shape mismatch: TF {tf_jacobian.shape} vs PT {pt_jacobian.shape}"

    # Verify batch dimension matches
    assert tf_jacobian.shape[0] == batch_size
    assert pt_jacobian.shape[0] == batch_size

    assert np.all(np.isfinite(tf_jacobian)), "TF Jacobian has non-finite values"
    assert np.all(np.isfinite(pt_jacobian)), "PT Jacobian has non-finite values"

    tf_row_norms = np.linalg.norm(tf_jacobian, axis=1)
    pt_row_norms = np.linalg.norm(pt_jacobian, axis=1)
    # First-order parity uses a moderate tolerance in float32.
    assert allclose(tf_row_norms, pt_row_norms, epsilon=5e-3)

    tf_to_pt = _tf_to_pt_flat_parameter_index(tf_weights, pt_weights)
    pt_jacobian_tf_order = pt_jacobian[:, tf_to_pt]
    assert allclose(tf_jacobian, pt_jacobian_tf_order, epsilon=5e-3)

def test_jacobian_non_zero():
    """Test that Jacobians are non-zero."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)
    inputs, targets = generate_test_data(batch_size=4, input_dim=input_dim, output_dim=output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_weights = tf_backend.get_model_weights(tf_model)
    pt_weights = pt_backend.get_model_weights(pt_model)

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    def pt_loss_fn(pred, target):
        return nn.MSELoss(reduction='none')(pred, target).mean(dim=-1)

    tf_jacobian = tf_backend.compute_jacobian(
        tf_model, tf_weights, tf_loss_fn, tf.constant(inputs), tf.constant(targets)
    ).numpy()

    pt_jacobian = pt_backend.compute_jacobian(
        pt_model, pt_weights, pt_loss_fn, torch.from_numpy(inputs), torch.from_numpy(targets)
    ).detach().numpy()

    # Both should have non-zero Jacobians
    assert np.linalg.norm(tf_jacobian) > 0, "TF Jacobian is zero"
    assert np.linalg.norm(pt_jacobian) > 0, "PT Jacobian is zero"

    tf_to_pt = _tf_to_pt_flat_parameter_index(tf_weights, pt_weights)
    pt_jacobian_tf_order = pt_jacobian[:, tf_to_pt]
    assert allclose(tf_jacobian, pt_jacobian_tf_order, epsilon=5e-3)


def test_output_jacobian_wrt_weights_match():
    """Test output Jacobian wrt weights parity across backends."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim, seed=11)
    inputs, _ = generate_test_data(batch_size=3, input_dim=input_dim, output_dim=output_dim, seed=19)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_weights = tf_backend.get_model_weights(tf_model)
    pt_weights = pt_backend.get_model_weights(pt_model)

    tf_outputs, tf_jacobians = tf_backend.compute_output_jacobian_wrt_weights(
        tf_model, tf_weights, tf.constant(inputs)
    )
    pt_outputs, pt_jacobians = pt_backend.compute_output_jacobian_wrt_weights(
        pt_model, pt_weights, torch.from_numpy(inputs)
    )

    assert allclose(tf_backend.to_numpy(tf_outputs), pt_backend.to_numpy(pt_outputs), epsilon=5e-3)
    assert len(tf_jacobians) == len(pt_jacobians)

    for tf_weight, pt_weight, tf_jacobian, pt_jacobian in zip(tf_weights, pt_weights, tf_jacobians, pt_jacobians):
        tf_shape = tuple(int(dim) for dim in tf_weight.shape)
        pt_shape = tuple(int(dim) for dim in pt_weight.shape)

        tf_jacobian_np = tf_backend.to_numpy(tf_jacobian)
        pt_jacobian_np = pt_backend.to_numpy(pt_jacobian)

        if tf_shape == pt_shape:
            pt_jacobian_tf_order = pt_jacobian_np
        elif len(tf_shape) == 2 and pt_shape == (tf_shape[1], tf_shape[0]):
            pt_jacobian_tf_order = np.swapaxes(pt_jacobian_np, -1, -2)
        else:
            raise AssertionError(f"Unsupported TF/PT shape pair: {tf_shape} vs {pt_shape}")

        assert tf_jacobian_np.shape == pt_jacobian_tf_order.shape
        assert allclose(tf_jacobian_np, pt_jacobian_tf_order, epsilon=5e-3)


def test_get_layers():
    """Test getting layers from model."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_layers = tf_backend.get_layers(tf_model)
    pt_layers = pt_backend.get_layers(pt_model)

    # Both should have 2 dense/linear layers
    assert len(tf_layers) == 2
    assert len(pt_layers) == 2

def test_find_last_weight_layer():
    """Test finding last layer with weights."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_last_idx = tf_backend.find_last_weight_layer(tf_model)
    pt_last_idx = pt_backend.find_last_weight_layer(pt_model)

    # Both should return -1 (last layer has weights)
    assert tf_last_idx == -1
    assert pt_last_idx == -1

def test_is_sequential_model():
    """Test checking if model is sequential."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    # Both models created by our helpers are sequential
    assert tf_backend.is_sequential_model(tf_model) == True
    assert pt_backend.is_sequential_model(pt_model) == True


def test_map_fn_simple():
    """Test simple map_fn operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    def double_fn(x):
        return x * 2

    tf_result = tf_backend.map_fn(double_fn, tf.constant(a)).numpy()
    pt_result = pt_backend.map_fn(double_fn, torch.from_numpy(a)).numpy()

    assert allclose(tf_result, pt_result)


def test_copy():
    """Test tensor copy operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1, 2, 3], dtype=np.float32)

    tf_original = tf.constant(a)
    pt_original = torch.from_numpy(a.copy())

    tf_copy = tf_backend.copy(tf_original)
    pt_copy = pt_backend.copy(pt_original)

    # Copies should equal originals
    assert allclose(tf_copy.numpy(), tf_original.numpy())
    assert allclose(pt_copy.numpy(), pt_original.numpy())

def test_clone_variable():
    """Test variable cloning."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_weights = tf_backend.get_model_weights(tf_model)
    pt_weights = pt_backend.get_model_weights(pt_model)

    # Clone first weight
    tf_clone = tf_backend.clone_variable(tf_weights[0])
    pt_clone = pt_backend.clone_variable(pt_weights[0])

    # Clones should match originals
    assert allclose(tf_clone.numpy(), tf_weights[0].numpy())
    assert allclose(pt_clone.numpy(), pt_weights[0].detach().numpy())


def test_boolean_mask():
    """Test boolean mask operation."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    mask = np.array([True, False, True, False, True])

    tf_result = tf_backend.boolean_mask(tf.constant(a), tf.constant(mask)).numpy()
    pt_result = pt_backend.boolean_mask(torch.from_numpy(a), torch.from_numpy(mask)).numpy()

    assert allclose(tf_result, pt_result)


def test_tensor_shape():
    """Test getting tensor shape."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    tf_shape = tf_backend.tensor_shape(tf.constant(a))
    pt_shape = pt_backend.tensor_shape(torch.from_numpy(a))

    assert tf_shape == pt_shape
    assert tf_shape == (2, 3)

def test_tensor_ndim():
    """Test getting tensor number of dimensions."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    # 2D tensor
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert tf_backend.tensor_ndim(tf.constant(a)) == pt_backend.tensor_ndim(torch.from_numpy(a))
    assert tf_backend.tensor_ndim(tf.constant(a)) == 2

    # 3D tensor
    b = np.zeros((2, 3, 4), dtype=np.float32)
    assert tf_backend.tensor_ndim(tf.constant(b)) == pt_backend.tensor_ndim(torch.from_numpy(b))
    assert tf_backend.tensor_ndim(tf.constant(b)) == 3

def test_get_batch_size():
    """Test getting batch size from tensor."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.zeros((8, 5), dtype=np.float32)

    tf_batch = tf_backend.get_batch_size(tf.constant(a))
    pt_batch = pt_backend.get_batch_size(torch.from_numpy(a))

    # For TF, this returns a Tensor, so convert to int
    if hasattr(tf_batch, 'numpy'):
        tf_batch = int(tf_batch.numpy())

    assert tf_batch == pt_batch
    assert tf_batch == 8


def test_gather_along_axis_simple():
    """Test simple gather along axis."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([0, 2])

    tf_result = tf_backend.gather_along_axis(
        tf.constant(a), tf.constant(indices, dtype=tf.int32), axis=0
    ).numpy()
    pt_result = pt_backend.gather_along_axis(
        torch.from_numpy(a), torch.from_numpy(indices).long(), axis=0
    ).numpy()

    assert allclose(tf_result, pt_result)


def test_gather_along_axis_invalid_index_raises():
    """Both backends should raise for out-of-bounds gather indices."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    indices = np.array([0, 3], dtype=np.int32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        with tf.device("/CPU:0"):
            tf_backend.to_numpy(tf_backend.gather_along_axis(tf.constant(a), tf.constant(indices), axis=0))

    with pytest.raises((RuntimeError, IndexError)):
        pt_backend.to_numpy(pt_backend.gather_along_axis(torch.from_numpy(a), torch.from_numpy(indices).long(), axis=0))


def test_dtype_functions():
    """Test dtype functions return appropriate values."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    # Test that float32 dtype works
    tf_float32 = tf_backend.float32_dtype()
    pt_float32 = pt_backend.float32_dtype()

    tf_tensor = tf_backend.zeros((2, 3), dtype=tf_float32)
    pt_tensor = pt_backend.zeros((2, 3), dtype=pt_float32)

    assert tf_backend.get_dtype(tf_tensor) == tf_float32
    assert pt_backend.get_dtype(pt_tensor) == pt_float32

def test_int_dtypes():
    """Test integer dtype functions."""
    from deel.influenciae.common import get_backend, Framework

    tf_backend = get_backend(Framework.TENSORFLOW)
    pt_backend = get_backend(Framework.PYTORCH)

    # Test int32
    tf_int32 = tf_backend.int32_dtype()
    pt_int32 = pt_backend.int32_dtype()

    tf_tensor = tf_backend.constant([1, 2, 3], dtype=tf_int32)
    pt_tensor = pt_backend.constant([1, 2, 3], dtype=pt_int32)

    assert tf_backend.get_dtype(tf_tensor) == tf_int32
    assert pt_backend.get_dtype(pt_tensor) == pt_int32

    # Test int64
    tf_int64 = tf_backend.int64_dtype()
    pt_int64 = pt_backend.int64_dtype()

    tf_tensor_64 = tf_backend.constant([1, 2, 3], dtype=tf_int64)
    pt_tensor_64 = pt_backend.constant([1, 2, 3], dtype=pt_int64)

    assert tf_backend.get_dtype(tf_tensor_64) == tf_int64
    assert pt_backend.get_dtype(pt_tensor_64) == pt_int64


def test_create_sequential_from_layers():
    """Test creating sequential models from layers."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_layers = tf_backend.get_layers(tf_model)
    pt_layers = pt_backend.get_layers(pt_model)

    # Both should successfully create sequential models from layers
    tf_sequential = tf_backend.create_sequential_from_layers(tf_layers)
    pt_sequential = pt_backend.create_sequential_from_layers(pt_layers)

    # Test that new models produce same output
    inputs, _ = generate_test_data(batch_size=4, input_dim=input_dim, output_dim=output_dim, seed=321)

    tf_output = tf_sequential(tf.constant(inputs)).numpy()
    pt_output = pt_sequential(torch.from_numpy(inputs)).detach().numpy()

    assert tf_output.shape == pt_output.shape
    assert allclose(tf_output, pt_output)


def test_hessian_computation_match():
    """Hessian matrices should match across backends up to parameter layout differences."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 3, 2, 1
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim, seed=17)
    inputs, targets = generate_test_data(batch_size=2, input_dim=input_dim, output_dim=output_dim, seed=29)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_weights = tf_backend.get_model_weights(tf_model)
    pt_weights = pt_backend.get_model_weights(pt_model)
    num_params = tf_backend.get_num_params(tf_weights)
    assert pt_backend.get_num_params(pt_weights) == num_params

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    def pt_loss_fn(pred, target):
        return nn.MSELoss(reduction='none')(pred, target).mean(dim=-1)

    tf_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)
    pt_dataset = [(torch.from_numpy(inputs), torch.from_numpy(targets))]

    tf_hessian = tf_backend.to_numpy(
        tf_backend.compute_hessian(tf_model, tf_weights, tf_loss_fn, tf_dataset, num_params)
    )
    pt_hessian = pt_backend.to_numpy(
        pt_backend.compute_hessian(pt_model, pt_weights, pt_loss_fn, pt_dataset, num_params)
    )

    assert np.all(np.isfinite(tf_hessian))
    assert np.all(np.isfinite(pt_hessian))

    tf_to_pt = _tf_to_pt_flat_parameter_index(tf_weights, pt_weights)
    pt_hessian_tf_order = pt_hessian[np.ix_(tf_to_pt, tf_to_pt)]

    assert np.allclose(tf_hessian, tf_hessian.T, atol=1e-5, rtol=1e-5)
    assert np.allclose(pt_hessian_tf_order, pt_hessian_tf_order.T, atol=1e-5, rtol=1e-5)
    # Second-order parity remains stable at 5e-3 with aligned parameter order.
    assert allclose(tf_hessian, pt_hessian_tf_order, epsilon=5e-3)


def test_hvp_produces_output():
    """Test that HVP computations produce valid non-zero outputs in both backends."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 3, 2, 1  # Smaller for faster computation
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim, seed=42)
    inputs, targets = generate_test_data(batch_size=1, input_dim=input_dim, output_dim=output_dim, seed=123)

    # Test TensorFlow HVP
    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    tf_backend = get_backend_for_model(tf_model)
    tf_weights = tf_backend.get_model_weights(tf_model)
    num_params = tf_backend.get_num_params(tf_weights)

    np.random.seed(456)
    reference_direction = [np.random.randn(*tuple(w.shape)).astype(np.float32) for w in tf_weights]
    tf_v = [tf.constant(direction) for direction in reference_direction]

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    tf_inputs = tf.constant(inputs)
    tf_targets = tf.constant(targets)

    tf_hvp = tf_backend.compute_hvp_single(
        tf_model, tf_weights, tf_loss_fn, tf_v, tf_inputs, tf_targets
    ).numpy()

    # Verify HVP produces output and is non-zero
    assert tf_hvp.shape == (num_params,), f"TF HVP shape mismatch: {tf_hvp.shape} != {(num_params,)}"
    assert np.all(np.isfinite(tf_hvp)), "TF HVP has non-finite values"
    assert np.linalg.norm(tf_hvp) > 0, "TF HVP is zero"

    # Test PyTorch HVP
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)
    pt_backend = get_backend_for_model(pt_model)
    pt_weights = pt_backend.get_model_weights(pt_model)
    assert pt_backend.get_num_params(pt_weights) == num_params

    pt_v = []
    for direction, pt_weight in zip(reference_direction, pt_weights):
        pt_shape = tuple(pt_weight.shape)
        if direction.shape == pt_shape:
            pt_v.append(torch.from_numpy(direction.copy()))
            continue
        if direction.ndim == 2 and pt_shape == (direction.shape[1], direction.shape[0]):
            pt_v.append(torch.from_numpy(direction.T.copy()))
            continue
        raise AssertionError(f"Incompatible direction shape {direction.shape} for PT weight shape {pt_shape}")

    def pt_loss_fn(pred, target):
        return nn.MSELoss(reduction='none')(pred, target).mean(dim=-1)

    pt_inputs = torch.from_numpy(inputs)
    pt_targets = torch.from_numpy(targets)

    pt_hvp = pt_backend.compute_hvp_single(
        pt_model, pt_weights, pt_loss_fn, pt_v, pt_inputs, pt_targets
    ).detach().numpy()

    # Verify HVP produces output and is non-zero
    assert pt_hvp.shape == (num_params,), f"PT HVP shape mismatch: {pt_hvp.shape} != {(num_params,)}"
    assert np.all(np.isfinite(pt_hvp)), "PT HVP has non-finite values"
    assert np.linalg.norm(pt_hvp) > 0, "PT HVP is zero"

    tf_to_pt = _tf_to_pt_flat_parameter_index(tf_weights, pt_weights)
    pt_hvp_tf_order = pt_hvp[tf_to_pt]
    # First-order parity uses a moderate tolerance in float32.
    assert allclose(tf_hvp, pt_hvp_tf_order, epsilon=5e-3)

    tf_direction_flat = np.concatenate([direction.reshape(-1) for direction in reference_direction])
    pt_direction_flat = np.concatenate([direction.detach().numpy().reshape(-1) for direction in pt_v])

    tf_directional_curvature = float(np.dot(tf_direction_flat, tf_hvp))
    pt_directional_curvature = float(np.dot(pt_direction_flat, pt_hvp))

    assert np.isfinite(tf_directional_curvature)
    assert np.isfinite(pt_directional_curvature)
    assert allclose(tf_directional_curvature, pt_directional_curvature, epsilon=5e-3)


def test_hvp_batch_match():
    """Batched HVP should match across backends after parameter-order remapping."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 3, 2, 1
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim, seed=101)
    inputs, targets = generate_test_data(batch_size=2, input_dim=input_dim, output_dim=output_dim, seed=202)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    tf_weights = tf_backend.get_model_weights(tf_model)
    pt_weights = pt_backend.get_model_weights(pt_model)
    num_params = tf_backend.get_num_params(tf_weights)
    assert pt_backend.get_num_params(pt_weights) == num_params

    np.random.seed(303)
    reference_direction = [np.random.randn(*tuple(weight.shape)).astype(np.float32) for weight in tf_weights]
    tf_v = [tf.constant(direction) for direction in reference_direction]

    pt_v = []
    for direction, pt_weight in zip(reference_direction, pt_weights):
        pt_shape = tuple(pt_weight.shape)
        if direction.shape == pt_shape:
            pt_v.append(torch.from_numpy(direction.copy()))
            continue
        if direction.ndim == 2 and pt_shape == (direction.shape[1], direction.shape[0]):
            pt_v.append(torch.from_numpy(direction.T.copy()))
            continue
        raise AssertionError(f"Incompatible direction shape {direction.shape} for PT weight shape {pt_shape}")

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)

    def pt_loss_fn(pred, target):
        return nn.MSELoss(reduction='none')(pred, target).mean(dim=-1)

    tf_hvp_batch = tf_backend.to_numpy(
        tf_backend.compute_hvp_batch(tf_model, tf_weights, tf_loss_fn, tf_v, tf.constant(inputs), tf.constant(targets))
    )
    pt_hvp_batch = pt_backend.to_numpy(
        pt_backend.compute_hvp_batch(
            pt_model, pt_weights, pt_loss_fn, pt_v, torch.from_numpy(inputs), torch.from_numpy(targets)
        )
    )

    tf_to_pt = _tf_to_pt_flat_parameter_index(tf_weights, pt_weights)
    pt_hvp_batch_tf_order = pt_hvp_batch[tf_to_pt]
    assert allclose(tf_hvp_batch, pt_hvp_batch_tf_order, epsilon=5e-3)


def test_split_model_invalid_layer_raises():
    """Both backends should raise when split layer cannot be resolved."""
    from deel.influenciae.common import get_backend_for_model

    tf_model = SimpleLinearModelTF.create()
    pt_model = SimpleLinearModelPT.create()

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    with pytest.raises(ValueError):
        tf_backend.split_model(tf_model, "missing_layer")

    with pytest.raises(ValueError):
        pt_backend.split_model(pt_model, "missing_layer")

def test_hvp_linearity():
    """Test that HVP is linear in v: H(v1+v2) = Hv1 + Hv2"""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 3, 2, 1
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim, seed=42)
    inputs, targets = generate_test_data(batch_size=1, input_dim=input_dim, output_dim=output_dim, seed=123)

    # Test with TensorFlow
    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    tf_backend = get_backend_for_model(tf_model)
    tf_weights = tf_backend.get_model_weights(tf_model)
    num_params = tf_backend.get_num_params(tf_weights)

    np.random.seed(456)
    tf_v1 = [tf.constant(np.random.randn(*w.shape).astype(np.float32)) for w in tf_weights]
    tf_v2 = [tf.constant(np.random.randn(*w.shape).astype(np.float32)) for w in tf_weights]
    tf_v_sum = [v1 + v2 for v1, v2 in zip(tf_v1, tf_v2)]

    tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    tf_inputs = tf.constant(inputs)
    tf_targets = tf.constant(targets)

    tf_hvp1 = tf_backend.compute_hvp_single(
        tf_model, tf_weights, tf_loss_fn, tf_v1, tf_inputs, tf_targets
    ).numpy()
    tf_hvp2 = tf_backend.compute_hvp_single(
        tf_model, tf_weights, tf_loss_fn, tf_v2, tf_inputs, tf_targets
    ).numpy()
    tf_hvp_sum = tf_backend.compute_hvp_single(
        tf_model, tf_weights, tf_loss_fn, tf_v_sum, tf_inputs, tf_targets
    ).numpy()

    assert tf_hvp1.shape == (num_params,)
    assert tf_hvp2.shape == (num_params,)
    assert tf_hvp_sum.shape == (num_params,)

    # H(v1+v2) should equal Hv1 + Hv2
    assert allclose(tf_hvp_sum, tf_hvp1 + tf_hvp2, epsilon=1e-4), \
        "TF HVP is not linear"

    # Test with PyTorch
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)
    pt_backend = get_backend_for_model(pt_model)
    pt_weights = pt_backend.get_model_weights(pt_model)
    assert pt_backend.get_num_params(pt_weights) == num_params

    np.random.seed(456)
    pt_v1 = [torch.from_numpy(np.random.randn(*w.shape).astype(np.float32)) for w in pt_weights]
    pt_v2 = [torch.from_numpy(np.random.randn(*w.shape).astype(np.float32)) for w in pt_weights]
    pt_v_sum = [v1 + v2 for v1, v2 in zip(pt_v1, pt_v2)]

    def pt_loss_fn(pred, target):
        return nn.MSELoss(reduction='none')(pred, target).mean(dim=-1)

    pt_inputs = torch.from_numpy(inputs)
    pt_targets = torch.from_numpy(targets)

    pt_hvp1 = pt_backend.compute_hvp_single(
        pt_model, pt_weights, pt_loss_fn, pt_v1, pt_inputs, pt_targets
    ).detach().numpy()
    pt_hvp2 = pt_backend.compute_hvp_single(
        pt_model, pt_weights, pt_loss_fn, pt_v2, pt_inputs, pt_targets
    ).detach().numpy()
    pt_hvp_sum = pt_backend.compute_hvp_single(
        pt_model, pt_weights, pt_loss_fn, pt_v_sum, pt_inputs, pt_targets
    ).detach().numpy()

    assert pt_hvp1.shape == (num_params,)
    assert pt_hvp2.shape == (num_params,)
    assert pt_hvp_sum.shape == (num_params,)

    # H(v1+v2) should equal Hv1 + Hv2
    assert allclose(pt_hvp_sum, pt_hvp1 + pt_hvp2, epsilon=1e-4), \
        "PT HVP is not linear"


def test_get_weights_for_layer_range():
    """Test getting weights for a range of layers."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    # Get weights for first layer only
    tf_weights_layer0 = tf_backend.get_weights_for_layer_range(tf_model, start_layer=0)
    pt_weights_layer0 = pt_backend.get_weights_for_layer_range(pt_model, start_layer=0)

    tf_num_params_0 = tf_backend.get_num_params(tf_weights_layer0)
    pt_num_params_0 = pt_backend.get_num_params(pt_weights_layer0)

    # First layer: input_dim * hidden_dim + hidden_dim
    expected_layer0_params = input_dim * hidden_dim + hidden_dim

    assert tf_num_params_0 == expected_layer0_params
    assert pt_num_params_0 == expected_layer0_params

def test_get_layer_index():
    """Test getting layer index."""
    from deel.influenciae.common import get_backend_for_model

    input_dim, hidden_dim, output_dim = 5, 3, 2
    weights = generate_matching_weights(input_dim, hidden_dim, output_dim)

    tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
    pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)

    tf_backend = get_backend_for_model(tf_model)
    pt_backend = get_backend_for_model(pt_model)

    # Get index for first layer (0)
    tf_idx_0 = tf_backend.get_layer_index(tf_model, 0)
    pt_idx_0 = pt_backend.get_layer_index(pt_model, 0)

    assert tf_idx_0 == 0
    assert pt_idx_0 == 0

    # Get index for last layer (-1)
    tf_idx_last = tf_backend.get_layer_index(tf_model, -1)
    pt_idx_last = pt_backend.get_layer_index(pt_model, -1)

    assert tf_idx_last == 1
    assert pt_idx_last == 1


def test_get_available_frameworks():
    """Test framework discovery includes both installed backends."""
    from deel.influenciae.common import get_available_frameworks, Framework

    frameworks = get_available_frameworks()

    assert Framework.TENSORFLOW in frameworks
    assert Framework.PYTORCH in frameworks

def test_detect_tensor_framework():
    """Test tensor framework detection for TF and PyTorch tensors."""
    from deel.influenciae.common import detect_tensor_framework, Framework

    tf_tensor = tf.constant([1.0, 2.0], dtype=tf.float32)
    pt_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)

    assert detect_tensor_framework(tf_tensor) == Framework.TENSORFLOW
    assert detect_tensor_framework(pt_tensor) == Framework.PYTORCH

def test_get_backend_for_tensor():
    """Test backend lookup from tensor type."""
    from deel.influenciae.common import get_backend_for_tensor, Framework

    tf_backend = get_backend_for_tensor(tf.constant([1.0], dtype=tf.float32))
    pt_backend = get_backend_for_tensor(torch.tensor([1.0], dtype=torch.float32))

    assert tf_backend.framework == Framework.TENSORFLOW
    assert pt_backend.framework == Framework.PYTORCH

def test_detect_dtype_framework():
    """Test dtype framework detection and unknown dtype fallback."""
    from deel.influenciae.common import detect_dtype_framework, Framework

    assert detect_dtype_framework(tf.float32) == Framework.TENSORFLOW
    assert detect_dtype_framework(torch.float32) == Framework.PYTORCH
    assert detect_dtype_framework(np.float32) is None
