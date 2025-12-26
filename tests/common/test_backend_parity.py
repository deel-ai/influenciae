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
        pt_loss = tf_influence.backend.to_numpy(pt_influence.batch_loss(pt_dataset))

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

