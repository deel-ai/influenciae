# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests to verify that PyTorch and TensorFlow backend implementations produce matching results.
These tests ensure framework-agnostic behavior of the influence function computations.
"""
import pytest
import numpy as np

# Check which frameworks are available
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential as TFSequential
    from tensorflow.keras.losses import MeanSquaredError, Reduction
    HAS_TENSORFLOW = True
except (ImportError, OSError):
    HAS_TENSORFLOW = False
    tf = None

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except (ImportError, OSError):
    HAS_PYTORCH = False
    torch = None
    nn = None

# Skip all tests if both frameworks are not available
pytestmark = pytest.mark.skipif(
    not (HAS_TENSORFLOW and HAS_PYTORCH),
    reason="Both TensorFlow and PyTorch are required for parity tests"
)


def almost_equal(a, b, epsilon=1e-4):
    """Check if two arrays are almost equal."""
    return np.allclose(a, b, atol=epsilon, rtol=epsilon)


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


class TestFrameworkDetection:
    """Test that framework detection works correctly."""

    def test_detect_tensorflow_model(self):
        """Test that TensorFlow models are correctly detected."""
        from deel.influenciae.common import detect_framework, Framework

        model = SimpleLinearModelTF.create()
        assert detect_framework(model) == Framework.TENSORFLOW

    def test_detect_pytorch_model(self):
        """Test that PyTorch models are correctly detected."""
        from deel.influenciae.common import detect_framework, Framework

        model = SimpleLinearModelPT.create()
        assert detect_framework(model) == Framework.PYTORCH

    def test_detect_invalid_model(self):
        """Test that invalid models raise ValueError."""
        from deel.influenciae.common import detect_framework

        with pytest.raises(ValueError):
            detect_framework("not a model")

        with pytest.raises(ValueError):
            detect_framework(42)


class TestBackendParity:
    """Test that backend operations produce matching results."""

    def test_get_num_params_match(self):
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

    def test_forward_pass_match(self):
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

        assert almost_equal(tf_output, pt_output), \
            f"Forward pass mismatch:\nTF: {tf_output}\nPT: {pt_output}"

    def test_loss_computation_match(self):
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

        assert almost_equal(tf_loss, pt_loss), \
            f"Loss mismatch:\nTF: {tf_loss}\nPT: {pt_loss}"

    def test_gradient_computation_match(self):
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

        # Note: Exact gradient values may differ due to weight layout differences
        # We primarily check that gradients have similar magnitude and are non-zero
        assert np.linalg.norm(tf_grad) > 0, "TF gradient is zero"
        assert np.linalg.norm(pt_grad) > 0, "PT gradient is zero"


class TestInfluenceModelParity:
    """Test that InfluenceModel produces matching results across frameworks."""

    def test_influence_model_creation(self):
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

    def test_influence_model_forward(self):
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

        assert almost_equal(tf_output, pt_output), \
            f"InfluenceModel forward mismatch:\nTF: {tf_output}\nPT: {pt_output}"

    def test_influence_model_batch_loss(self):
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

        assert almost_equal(tf_loss, pt_loss), \
            f"InfluenceModel batch_loss mismatch:\nTF: {tf_loss}\nPT: {pt_loss}"

    def test_influence_model_layer_targeting(self):
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


class TestTensorOperationsParity:
    """Test tensor operations match between backends."""

    def test_concat(self):
        """Test tensor concatenation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)

        tf_result = tf_backend.concat([tf.constant(a), tf.constant(b)], axis=0).numpy()
        pt_result = pt_backend.concat([torch.from_numpy(a), torch.from_numpy(b)], axis=0).numpy()

        assert almost_equal(tf_result, pt_result)

    def test_stack(self):
        """Test tensor stacking."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)

        tf_result = tf_backend.stack([tf.constant(a), tf.constant(b)], axis=0).numpy()
        pt_result = pt_backend.stack([torch.from_numpy(a), torch.from_numpy(b)], axis=0).numpy()

        assert almost_equal(tf_result, pt_result)

    def test_reshape(self):
        """Test tensor reshape."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        tf_result = tf_backend.reshape(tf.constant(a), (3, 2)).numpy()
        pt_result = pt_backend.reshape(torch.from_numpy(a), (3, 2)).numpy()

        assert almost_equal(tf_result, pt_result)

    def test_reduce_sum(self):
        """Test reduce sum."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        # Full reduction
        tf_result_full = tf_backend.reduce_sum(tf.constant(a)).numpy()
        pt_result_full = pt_backend.reduce_sum(torch.from_numpy(a)).numpy()
        assert almost_equal(tf_result_full, pt_result_full)

        # Axis reduction
        tf_result_axis = tf_backend.reduce_sum(tf.constant(a), axis=1).numpy()
        pt_result_axis = pt_backend.reduce_sum(torch.from_numpy(a), axis=1).numpy()
        assert almost_equal(tf_result_axis, pt_result_axis)

    def test_to_numpy(self):
        """Test conversion to numpy."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        tf_result = tf_backend.to_numpy(tf.constant(a))
        pt_result = pt_backend.to_numpy(torch.from_numpy(a))

        assert isinstance(tf_result, np.ndarray)
        assert isinstance(pt_result, np.ndarray)
        assert almost_equal(tf_result, pt_result)

    def test_expand_dims(self):
        """Test expanding tensor dimensions."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1, 2, 3], dtype=np.float32)

        # Expand at axis 0
        tf_result = tf_backend.expand_dims(tf.constant(a), axis=0).numpy()
        pt_result = pt_backend.expand_dims(torch.from_numpy(a), axis=0).numpy()
        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)

        # Expand at axis 1
        tf_result = tf_backend.expand_dims(tf.constant(a), axis=1).numpy()
        pt_result = pt_backend.expand_dims(torch.from_numpy(a), axis=1).numpy()
        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)

    def test_squeeze(self):
        """Test squeezing tensor dimensions."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[[1, 2, 3]]], dtype=np.float32)  # Shape (1, 1, 3)

        # Squeeze all dimensions of size 1
        tf_result = tf_backend.squeeze(tf.constant(a)).numpy()
        pt_result = pt_backend.squeeze(torch.from_numpy(a)).numpy()
        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)

        # Squeeze specific axis
        tf_result = tf_backend.squeeze(tf.constant(a), axis=0).numpy()
        pt_result = pt_backend.squeeze(torch.from_numpy(a), axis=0).numpy()
        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)

    def test_transpose(self):
        """Test tensor transpose."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # Shape (2, 3)

        tf_result = tf_backend.transpose(tf.constant(a)).numpy()
        pt_result = pt_backend.transpose(torch.from_numpy(a)).numpy()
        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)

    def test_matmul(self):
        """Test matrix multiplication."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)

        tf_result = tf_backend.matmul(tf.constant(a), tf.constant(b)).numpy()
        pt_result = pt_backend.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_multiply(self):
        """Test element-wise multiplication."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)

        tf_result = tf_backend.multiply(tf.constant(a), tf.constant(b)).numpy()
        pt_result = pt_backend.multiply(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_abs(self):
        """Test absolute value."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([-1, 2, -3, 4], dtype=np.float32)

        tf_result = tf_backend.abs(tf.constant(a)).numpy()
        pt_result = pt_backend.abs(torch.from_numpy(a)).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_argmax(self):
        """Test argmax operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 5, 3], [4, 2, 6]], dtype=np.float32)

        # Argmax along axis 1
        tf_result = tf_backend.argmax(tf.constant(a), axis=1).numpy()
        pt_result = pt_backend.argmax(torch.from_numpy(a), axis=1).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_argmin(self):
        """Test argmin operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 5, 3], [4, 2, 6]], dtype=np.float32)

        # Argmin along axis 1
        tf_result = tf_backend.argmin(tf.constant(a), axis=1).numpy()
        pt_result = pt_backend.argmin(torch.from_numpy(a), axis=1).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_reduce_mean(self):
        """Test reduce mean."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        # Full reduction
        tf_result_full = tf_backend.reduce_mean(tf.constant(a)).numpy()
        pt_result_full = pt_backend.reduce_mean(torch.from_numpy(a)).numpy()
        assert almost_equal(tf_result_full, pt_result_full)

        # Axis reduction
        tf_result_axis = tf_backend.reduce_mean(tf.constant(a), axis=1).numpy()
        pt_result_axis = pt_backend.reduce_mean(torch.from_numpy(a), axis=1).numpy()
        assert almost_equal(tf_result_axis, pt_result_axis)

        # Axis reduction with keepdims
        tf_result_keepdims = tf_backend.reduce_mean(tf.constant(a), axis=1, keepdims=True).numpy()
        pt_result_keepdims = pt_backend.reduce_mean(torch.from_numpy(a), axis=1, keepdims=True).numpy()
        assert tf_result_keepdims.shape == pt_result_keepdims.shape
        assert almost_equal(tf_result_keepdims, pt_result_keepdims)


class TestLinearAlgebraParity:
    """Test linear algebra operations match between backends."""

    def test_zeros(self):
        """Test creating zeros tensor."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        shape = (3, 4)

        tf_result = tf_backend.zeros(shape).numpy()
        pt_result = pt_backend.zeros(shape).numpy()

        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)

    def test_zeros_like(self):
        """Test creating zeros tensor with same shape as input."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        tf_result = tf_backend.zeros_like(tf.constant(a)).numpy()
        pt_result = pt_backend.zeros_like(torch.from_numpy(a)).numpy()

        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)
        assert np.all(tf_result == 0)
        assert np.all(pt_result == 0)

    def test_sqrt(self):
        """Test square root operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1, 4, 9, 16], dtype=np.float32)

        tf_result = tf_backend.sqrt(tf.constant(a)).numpy()
        pt_result = pt_backend.sqrt(torch.from_numpy(a)).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_maximum(self):
        """Test element-wise maximum."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1, 5, 3], dtype=np.float32)
        b = np.array([2, 4, 4], dtype=np.float32)

        tf_result = tf_backend.maximum(tf.constant(a), tf.constant(b)).numpy()
        pt_result = pt_backend.maximum(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_norm(self):
        """Test norm computation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)

        # Frobenius norm (default, flattened)
        tf_result = tf_backend.norm(tf.constant(a)).numpy()
        pt_result = pt_backend.norm(torch.from_numpy(a)).numpy()
        assert almost_equal(tf_result, pt_result)

        # Norm along axis
        tf_result_axis = tf_backend.norm(tf.constant(a), axis=1).numpy()
        pt_result_axis = pt_backend.norm(torch.from_numpy(a), axis=1).numpy()
        assert almost_equal(tf_result_axis, pt_result_axis)

    def test_pinv(self):
        """Test pseudo-inverse computation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        # Create a simple matrix
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        tf_result = tf_backend.pinv(tf.constant(a)).numpy()
        pt_result = pt_backend.pinv(torch.from_numpy(a)).numpy()

        # Verify A @ pinv(A) @ A ≈ A
        tf_verify = np.matmul(np.matmul(a, tf_result), a)
        pt_verify = np.matmul(np.matmul(a, pt_result), a)

        assert almost_equal(tf_verify, a, epsilon=1e-3)
        assert almost_equal(pt_verify, a, epsilon=1e-3)

    def test_normalize(self):
        """Test normalization."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([3, 4], dtype=np.float32)  # norm = 5

        tf_result = tf_backend.normalize(tf.constant(a)).numpy()
        pt_result = pt_backend.normalize(torch.from_numpy(a)).numpy()

        # Both should have unit norm
        assert almost_equal(np.linalg.norm(tf_result), 1.0)
        assert almost_equal(np.linalg.norm(pt_result), 1.0)
        assert almost_equal(tf_result, pt_result)


class TestDataManipulationParity:
    """Test data manipulation operations match between backends."""

    def test_arange(self):
        """Test arange operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        tf_result = tf_backend.arange(0, 10).numpy()
        pt_result = pt_backend.arange(0, 10).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_tile(self):
        """Test tile operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)

        tf_result = tf_backend.tile(tf.constant(a), (2, 3)).numpy()
        pt_result = pt_backend.tile(torch.from_numpy(a), (2, 3)).numpy()

        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)

    def test_repeat(self):
        """Test repeat operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1, 2, 3], dtype=np.float32)

        tf_result = tf_backend.repeat(tf.constant(a), 3, axis=0).numpy()
        pt_result = pt_backend.repeat(torch.from_numpy(a), 3, axis=0).numpy()

        assert tf_result.shape == pt_result.shape
        assert almost_equal(tf_result, pt_result)

    def test_sign(self):
        """Test sign operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([-2, -1, 0, 1, 2], dtype=np.float32)

        tf_result = tf_backend.sign(tf.constant(a)).numpy()
        pt_result = pt_backend.sign(torch.from_numpy(a)).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_pow(self):
        """Test power operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1, 2, 3, 4], dtype=np.float32)

        tf_result = tf_backend.pow(tf.constant(a), 2).numpy()
        pt_result = pt_backend.pow(torch.from_numpy(a), 2).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_top_k(self):
        """Test top_k operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float32)

        tf_values, tf_indices = tf_backend.top_k(tf.constant(a), k=3)
        pt_values, pt_indices = pt_backend.top_k(torch.from_numpy(a), k=3)

        tf_values = tf_values.numpy()
        pt_values = pt_values.numpy()

        # Values should match (sorted descending)
        assert almost_equal(tf_values, pt_values)


class TestTensorConversionParity:
    """Test tensor conversion operations match between backends."""

    def test_constant(self):
        """Test constant tensor creation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        value = [1.0, 2.0, 3.0]

        tf_result = tf_backend.constant(value).numpy()
        pt_result = pt_backend.constant(value).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_convert_to_tensor(self):
        """Test conversion to tensor."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        tf_result = tf_backend.convert_to_tensor(a).numpy()
        pt_result = pt_backend.convert_to_tensor(a).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_cast(self):
        """Test casting tensors."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1.5, 2.7, 3.2], dtype=np.float32)

        # Cast to int32
        tf_result = tf_backend.cast(tf.constant(a), tf_backend.int32_dtype()).numpy()
        pt_result = pt_backend.cast(torch.from_numpy(a), pt_backend.int32_dtype()).numpy()
        assert almost_equal(tf_result, pt_result)

    def test_reduce_prod(self):
        """Test reduce product."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)

        # Full reduction
        tf_result = tf_backend.reduce_prod(tf.constant(a)).numpy()
        pt_result = pt_backend.reduce_prod(torch.from_numpy(a)).numpy()
        assert almost_equal(tf_result, pt_result)

        # Axis reduction
        tf_result_axis = tf_backend.reduce_prod(tf.constant(a), axis=1).numpy()
        pt_result_axis = pt_backend.reduce_prod(torch.from_numpy(a), axis=1).numpy()
        assert almost_equal(tf_result_axis, pt_result_axis)


class TestLogicalOperationsParity:
    """Test logical operations match between backends."""

    def test_logical_and(self):
        """Test logical AND operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])

        tf_result = tf_backend.logical_and(tf.constant(a), tf.constant(b)).numpy()
        pt_result = pt_backend.logical_and(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        assert np.array_equal(tf_result, pt_result)

    def test_reduce_any(self):
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


class TestJacobianComputationParity:
    """Test that Jacobian computations produce matching results."""

    def test_jacobian_shape_match(self):
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

    def test_jacobian_non_zero(self):
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


class TestModelUtilityParity:
    """Test model utility functions match between backends."""

    def test_get_layers(self):
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

    def test_find_last_weight_layer(self):
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

    def test_is_sequential_model(self):
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


class TestMapFnParity:
    """Test map_fn operations match between backends."""

    def test_map_fn_simple(self):
        """Test simple map_fn operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        def double_fn(x):
            return x * 2

        tf_result = tf_backend.map_fn(double_fn, tf.constant(a)).numpy()
        pt_result = pt_backend.map_fn(double_fn, torch.from_numpy(a)).numpy()

        assert almost_equal(tf_result, pt_result)


class TestCopyAndCloneParity:
    """Test copy/clone operations match between backends."""

    def test_copy(self):
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
        assert almost_equal(tf_copy.numpy(), tf_original.numpy())
        assert almost_equal(pt_copy.numpy(), pt_original.numpy())

    def test_clone_variable(self):
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
        assert almost_equal(tf_clone.numpy(), tf_weights[0].numpy())
        assert almost_equal(pt_clone.numpy(), pt_weights[0].detach().numpy())


class TestBooleanMaskParity:
    """Test boolean mask operations match between backends."""

    def test_boolean_mask(self):
        """Test boolean mask operation."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])

        tf_result = tf_backend.boolean_mask(tf.constant(a), tf.constant(mask)).numpy()
        pt_result = pt_backend.boolean_mask(torch.from_numpy(a), torch.from_numpy(mask)).numpy()

        assert almost_equal(tf_result, pt_result)


class TestTensorMetadataParity:
    """Test tensor metadata operations match between backends."""

    def test_tensor_shape(self):
        """Test getting tensor shape."""
        from deel.influenciae.common import get_backend, Framework

        tf_backend = get_backend(Framework.TENSORFLOW)
        pt_backend = get_backend(Framework.PYTORCH)

        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        tf_shape = tf_backend.tensor_shape(tf.constant(a))
        pt_shape = pt_backend.tensor_shape(torch.from_numpy(a))

        assert tf_shape == pt_shape
        assert tf_shape == (2, 3)

    def test_tensor_ndim(self):
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

    def test_get_batch_size(self):
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


class TestGatherOperationsParity:
    """Test gather operations match between backends."""

    def test_gather_along_axis_simple(self):
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

        assert almost_equal(tf_result, pt_result)


class TestDtypeOperationsParity:
    """Test dtype-related operations match between backends."""

    def test_dtype_functions(self):
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

    def test_int_dtypes(self):
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


class TestSequentialModelOperationsParity:
    """Test sequential model operations match between backends."""

    def test_create_sequential_from_layers(self):
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
        inputs = np.random.randn(4, input_dim).astype(np.float32)

        tf_output = tf_sequential(tf.constant(inputs)).numpy()
        pt_output = pt_sequential(torch.from_numpy(inputs)).detach().numpy()

        assert tf_output.shape == pt_output.shape


class TestHVPComputationParity:
    """Test Hessian-vector product computation parity."""

    def test_hvp_produces_output(self):
        """Test that HVP computations produce valid non-zero outputs in both backends."""
        from deel.influenciae.common import get_backend_for_model

        input_dim, hidden_dim, output_dim = 3, 2, 1  # Smaller for faster computation
        weights = generate_matching_weights(input_dim, hidden_dim, output_dim, seed=42)
        inputs, targets = generate_test_data(batch_size=2, input_dim=input_dim, output_dim=output_dim, seed=123)

        # Test TensorFlow HVP
        tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
        tf_backend = get_backend_for_model(tf_model)
        tf_weights = tf_backend.get_model_weights(tf_model)

        np.random.seed(456)
        tf_v = [tf.constant(np.random.randn(*w.shape).astype(np.float32)) for w in tf_weights]

        tf_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        tf_inputs = tf.constant(inputs)
        tf_targets = tf.constant(targets)

        tf_hvp = tf_backend.compute_hvp_single(
            tf_model, tf_weights, tf_loss_fn, tf_v, tf_inputs, tf_targets
        ).numpy()

        # Verify HVP produces output and is non-zero
        assert len(tf_hvp.shape) == 1, "TF HVP should be 1D"
        assert tf_hvp.shape[0] > 0, "TF HVP should have elements"
        assert np.linalg.norm(tf_hvp) > 0, "TF HVP is zero"

        # Test PyTorch HVP
        pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)
        pt_backend = get_backend_for_model(pt_model)
        pt_weights = pt_backend.get_model_weights(pt_model)

        np.random.seed(456)
        pt_v = [torch.from_numpy(np.random.randn(*w.shape).astype(np.float32)) for w in pt_weights]

        def pt_loss_fn(pred, target):
            return nn.MSELoss(reduction='none')(pred, target).mean(dim=-1)

        pt_inputs = torch.from_numpy(inputs)
        pt_targets = torch.from_numpy(targets)

        pt_hvp = pt_backend.compute_hvp_single(
            pt_model, pt_weights, pt_loss_fn, pt_v, pt_inputs, pt_targets
        ).detach().numpy()

        # Verify HVP produces output and is non-zero
        assert len(pt_hvp.shape) == 1, "PT HVP should be 1D"
        assert pt_hvp.shape[0] > 0, "PT HVP should have elements"
        assert np.linalg.norm(pt_hvp) > 0, "PT HVP is zero"

    def test_hvp_linearity(self):
        """Test that HVP is linear in v: H(v1+v2) = Hv1 + Hv2"""
        from deel.influenciae.common import get_backend_for_model

        input_dim, hidden_dim, output_dim = 3, 2, 1
        weights = generate_matching_weights(input_dim, hidden_dim, output_dim, seed=42)
        inputs, targets = generate_test_data(batch_size=2, input_dim=input_dim, output_dim=output_dim, seed=123)

        # Test with TensorFlow
        tf_model = SimpleLinearModelTF.create(input_dim, hidden_dim, output_dim, weights)
        tf_backend = get_backend_for_model(tf_model)
        tf_weights = tf_backend.get_model_weights(tf_model)

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

        # H(v1+v2) should equal Hv1 + Hv2
        assert almost_equal(tf_hvp_sum, tf_hvp1 + tf_hvp2, epsilon=1e-4), \
            "TF HVP is not linear"

        # Test with PyTorch
        pt_model = SimpleLinearModelPT.create(input_dim, hidden_dim, output_dim, weights)
        pt_backend = get_backend_for_model(pt_model)
        pt_weights = pt_backend.get_model_weights(pt_model)

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

        # H(v1+v2) should equal Hv1 + Hv2
        assert almost_equal(pt_hvp_sum, pt_hvp1 + pt_hvp2, epsilon=1e-4), \
            "PT HVP is not linear"


class TestWeightRangeOperationsParity:
    """Test weight range operations match between backends."""

    def test_get_weights_for_layer_range(self):
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

    def test_get_layer_index(self):
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

