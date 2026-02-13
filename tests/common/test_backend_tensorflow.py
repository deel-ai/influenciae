# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the TensorFlow backend implementation.
These tests verify the TensorFlow-specific functionality works correctly.
"""
import pytest
import numpy as np

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input, Flatten, ReLU
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, Reduction
    HAS_TENSORFLOW = True
except (ImportError, OSError):
    HAS_TENSORFLOW = False
    tf = None

pytestmark = pytest.mark.skipif(
    not HAS_TENSORFLOW,
    reason="TensorFlow is required for these tests"
)


def almost_equal(a, b, epsilon=1e-4):
    """Check if two arrays are almost equal."""
    if isinstance(a, tf.Tensor):
        a = a.numpy()
    if isinstance(b, tf.Tensor):
        b = b.numpy()
    return np.allclose(a, b, atol=epsilon, rtol=epsilon)


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


class TestTensorFlowBackendBasics:
    """Test basic backend functionality."""

    def test_framework_property(self, backend):
        """Test that framework property returns TENSORFLOW."""
        from deel.influenciae.common import Framework
        assert backend.framework == Framework.TENSORFLOW

    def test_get_model_weights(self, backend, simple_model):
        """Test getting model weights."""
        weights = backend.get_model_weights(simple_model)

        # Should have 4 weight tensors: 2 weights + 2 biases
        assert len(weights) == 4

        # Check shapes (TF uses [in, out] convention)
        assert weights[0].shape == (5, 3)  # First dense weight
        assert weights[1].shape == (3,)    # First dense bias
        assert weights[2].shape == (3, 2)  # Second dense weight
        assert weights[3].shape == (2,)    # Second dense bias

    def test_get_model_weights_specific_layers(self, backend, simple_model):
        """Test getting weights from specific layers."""
        layers = [simple_model.layers[0], simple_model.layers[1]]

        weights = backend.get_model_weights(simple_model, layers)
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
        inputs = tf.random.normal((4, 5))
        output = backend.forward(simple_model, inputs)

        assert output.shape == (4, 2)

    def test_get_layers(self, backend, simple_model):
        """Test getting all layers."""
        layers = backend.get_layers(simple_model)

        # 2 Dense layers (Input is not counted in model.layers for Sequential with Input)
        assert len(layers) == 2

    def test_find_layer_by_name(self, backend, simple_model):
        """Test finding layer by name."""
        idx, layer = backend.find_layer_by_name(simple_model, 'hidden')

        assert idx == 0
        assert layer.name == 'hidden'

    def test_find_layer_by_name_not_found(self, backend, simple_model):
        """Test finding non-existent layer raises error."""
        with pytest.raises(ValueError):
            backend.find_layer_by_name(simple_model, 'nonexistent')


class TestTensorFlowBackendComputation:
    """Test gradient and loss computation."""

    def test_compute_loss(self, backend, simple_model):
        """Test loss computation."""
        inputs = tf.random.normal((4, 5))
        targets = tf.random.normal((4, 2))
        loss_fn = MeanSquaredError(reduction=Reduction.NONE)

        loss = backend.compute_loss(simple_model, loss_fn, inputs, targets)

        # MSE without reduction returns (batch,)
        assert loss.shape == (4,)

    def test_compute_loss_with_sample_weight(self, backend, simple_model):
        """Test loss computation with sample weights."""
        inputs = tf.random.normal((4, 5))
        targets = tf.random.normal((4, 2))
        sample_weight = tf.constant([1.0, 2.0, 0.5, 1.5])
        loss_fn = MeanSquaredError(reduction=Reduction.NONE)

        loss_unweighted = backend.compute_loss(simple_model, loss_fn, inputs, targets)
        loss_weighted = backend.compute_loss(simple_model, loss_fn, inputs, targets, sample_weight)

        # Weighted loss should be different
        assert not np.allclose(loss_unweighted.numpy(), loss_weighted.numpy())

    def test_compute_gradient(self, backend, simple_model):
        """Test gradient computation."""
        inputs = tf.random.normal((4, 5))
        targets = tf.random.normal((4, 2))
        weights = backend.get_model_weights(simple_model)
        loss_fn = MeanSquaredError(reduction=Reduction.NONE)

        gradient = backend.compute_gradient(simple_model, weights, loss_fn, inputs, targets)

        num_params = backend.get_num_params(weights)
        assert gradient.shape == (num_params,)
        assert tf.linalg.norm(gradient) > 0  # Gradient should be non-zero

    def test_compute_jacobian(self, backend, simple_model):
        """Test Jacobian computation."""
        batch_size = 3
        inputs = tf.random.normal((batch_size, 5))
        targets = tf.random.normal((batch_size, 2))
        weights = backend.get_model_weights(simple_model)
        loss_fn = MeanSquaredError(reduction=Reduction.NONE)

        jacobian = backend.compute_jacobian(simple_model, weights, loss_fn, inputs, targets)

        num_params = backend.get_num_params(weights)
        assert jacobian.shape == (batch_size, num_params)


class TestTensorFlowBackendTensorOps:
    """Test tensor operations."""

    def test_concat(self, backend):
        """Test tensor concatenation."""
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[5, 6], [7, 8]])

        result = backend.concat([a, b], axis=0)
        expected = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]])

        assert tf.reduce_all(result == expected)

    def test_stack(self, backend):
        """Test tensor stacking."""
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])

        result = backend.stack([a, b], axis=0)
        expected = tf.constant([[1, 2, 3], [4, 5, 6]])

        assert tf.reduce_all(result == expected)

    def test_reshape(self, backend):
        """Test tensor reshape."""
        a = tf.constant([[1, 2, 3], [4, 5, 6]])

        result = backend.reshape(a, (3, 2))
        assert result.shape == (3, 2)

    def test_map_fn_output_signature(self, backend):
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

    def test_reduce_sum(self, backend):
        """Test reduce sum."""
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])

        result_full = backend.reduce_sum(a)
        assert result_full.numpy() == 10.0

        result_axis0 = backend.reduce_sum(a, axis=0)
        assert tf.reduce_all(result_axis0 == tf.constant([4.0, 6.0]))

        result_axis1 = backend.reduce_sum(a, axis=1)
        assert tf.reduce_all(result_axis1 == tf.constant([3.0, 7.0]))

    def test_to_numpy(self, backend):
        """Test numpy conversion."""
        a = tf.constant([1.0, 2.0, 3.0])

        result = backend.to_numpy(a)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_get_batch_size(self, backend):
        """Test batch size extraction."""
        a = tf.random.normal((8, 5, 3))

        # Note: TF returns a tensor, need to evaluate
        batch_size = backend.get_batch_size(a)
        if isinstance(batch_size, tf.Tensor):
            batch_size = batch_size.numpy()
        assert batch_size == 8

    def test_abs(self, backend):
        """Test absolute value."""
        a = tf.constant([-1.0, 2.0, -3.0])
        result = backend.abs(a)
        expected = tf.constant([1.0, 2.0, 3.0])
        assert tf.reduce_all(result == expected)

    def test_argmax(self, backend):
        """Test argmax."""
        a = tf.constant([[1.0, 3.0, 2.0], [4.0, 1.0, 2.0]])
        result = backend.argmax(a, axis=1)
        expected = tf.constant([1, 0], dtype=tf.int64)
        assert tf.reduce_all(result == expected)

    def test_gather_along_axis(self, backend):
        """Test gather along axis."""
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        indices = tf.constant([2, 0])
        result = backend.gather_along_axis(a, indices, axis=1, batch_dims=1)
        expected = tf.constant([3.0, 4.0])
        assert tf.reduce_all(result == expected)

    def test_get_output_shape(self, backend, simple_model):
        """Test getting model output shape."""
        out_shape = backend.get_output_shape(simple_model)
        assert out_shape == (None, 2)

    def test_split_model(self, backend, simple_model):
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
        assert almost_equal(head_output, original_output)


class TestTensorFlowBackendLayerOperations:
    """Test layer-related operations."""

    def test_find_last_weight_layer(self, backend, simple_model):
        """Test finding last weight layer."""
        idx = backend.find_last_weight_layer(simple_model)

        # Last Dense layer should be at index -1
        assert idx == -1

    def test_find_last_weight_layer_with_trailing_layers(self, backend):
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

    def test_get_layer_index_by_int(self, backend, simple_model):
        """Test layer index lookup by integer."""
        assert backend.get_layer_index(simple_model, 0) == 0
        assert backend.get_layer_index(simple_model, 1) == 1
        assert backend.get_layer_index(simple_model, -1) == 1  # 2 layers, -1 -> 1

    def test_get_layer_index_by_name(self, backend, simple_model):
        """Test layer index lookup by name."""
        idx = backend.get_layer_index(simple_model, 'hidden')
        assert idx == 0

        idx = backend.get_layer_index(simple_model, 'output')
        assert idx == 1

    def test_get_weights_for_layer_range_single_layer(self, backend, simple_model):
        """Test getting weights for a single layer."""
        weights = backend.get_weights_for_layer_range(simple_model, start_layer=0)

        # First Dense layer: weight + bias
        assert len(weights) == 2
        assert weights[0].shape == (5, 3)
        assert weights[1].shape == (3,)

    def test_get_weights_for_layer_range_by_name(self, backend, simple_model):
        """Test getting weights by layer name."""
        weights = backend.get_weights_for_layer_range(simple_model, start_layer='hidden')

        assert len(weights) == 2
        assert weights[0].shape == (5, 3)

    def test_get_weights_for_layer_range_multiple_layers(self, backend, simple_model):
        """Test getting weights for multiple layers."""
        weights = backend.get_weights_for_layer_range(simple_model, start_layer=0, last_layer=1)

        # Both Dense layers
        assert len(weights) == 4

    def test_get_weights_for_layer_range_auto_detect(self, backend, simple_model):
        """Test auto-detecting the last weight layer."""
        weights = backend.get_weights_for_layer_range(simple_model, start_layer=None)

        # Should get second-to-last Dense layer (before logits)
        assert len(weights) == 2

    def test_get_weights_for_layer_range_invalid_range(self, backend, simple_model):
        """Test that invalid layer range raises error."""
        with pytest.raises(AssertionError):
            backend.get_weights_for_layer_range(simple_model, start_layer=1, last_layer=0)


class TestTensorFlowInfluenceModel:
    """Test InfluenceModel with TensorFlow backend."""

    def test_create_influence_model(self, simple_model):
        """Test creating an InfluenceModel."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

        # Should use last layer by default (the output layer)
        expected_params = 3 * 2 + 2  # Output dense layer
        assert influence_model.nb_params == expected_params

    def test_influence_model_with_start_layer(self, simple_model):
        """Test InfluenceModel with specific start layer."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        influence_model = InfluenceModel(simple_model, start_layer=0, loss_function=loss_fn)

        # First dense layer: 5*3 + 3 = 18
        expected_params = 5 * 3 + 3
        assert influence_model.nb_params == expected_params

    def test_influence_model_with_layer_name(self, simple_model):
        """Test InfluenceModel with layer name."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        influence_model = InfluenceModel(simple_model, start_layer='output', loss_function=loss_fn)

        # Output layer: 3*2 + 2 = 8
        expected_params = 3 * 2 + 2
        assert influence_model.nb_params == expected_params

    def test_influence_model_forward(self, simple_model):
        """Test InfluenceModel forward pass."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

        inputs = tf.random.normal((4, 5))
        output = influence_model(inputs)

        assert output.shape == (4, 2)

    def test_influence_model_batch_loss(self, simple_model):
        """Test InfluenceModel batch loss computation."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

        # Create a batched dataset
        inputs = tf.random.normal((4, 5))
        targets = tf.random.normal((4, 2))
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)

        loss = influence_model.batch_loss(dataset)

        assert loss.shape == (4,)

    def test_influence_model_batch_jacobian(self, simple_model):
        """Test InfluenceModel batch Jacobian computation."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

        inputs = tf.random.normal((4, 5))
        targets = tf.random.normal((4, 2))
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)

        jacobian = influence_model.batch_jacobian(dataset)

        assert jacobian.shape == (4, influence_model.nb_params)

    def test_influence_model_batch_gradient(self, simple_model):
        """Test InfluenceModel batch gradient computation."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

        inputs = tf.random.normal((4, 5))
        targets = tf.random.normal((4, 2))
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)

        gradients = influence_model.batch_gradient(dataset)

        # 2 batches, each producing a gradient
        assert gradients.shape == (2, influence_model.nb_params)


class TestTensorFlowLossValidation:
    """Test loss function validation."""

    def test_loss_with_reduction_raises_error(self, simple_model):
        """Test that loss function with reduction raises ValueError."""
        from deel.influenciae.common import InfluenceModel

        loss_fn_sum = MeanSquaredError(reduction=Reduction.SUM)
        loss_fn_mean = MeanSquaredError(reduction=Reduction.AUTO)

        with pytest.raises(ValueError):
            InfluenceModel(simple_model, loss_function=loss_fn_sum)

        with pytest.raises(ValueError):
            InfluenceModel(simple_model, loss_function=loss_fn_mean)

    def test_loss_without_reduction_works(self, simple_model):
        """Test that loss function without reduction works."""
        from deel.influenciae.common import InfluenceModel

        loss_fn = MeanSquaredError(reduction=Reduction.NONE)
        influence_model = InfluenceModel(simple_model, loss_function=loss_fn)

        assert influence_model is not None


class TestTensorFlowBackendDatasetOperations:
    """Test TensorFlow backend dataset helpers."""

    def test_map_dataset(self, backend):
        """Test mapping a function over a batched dataset."""
        x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
        y = tf.constant([[5.0], [6.0], [7.0], [8.0]])
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

        mapped = backend.map_dataset(dataset, lambda a, b: (a + 1.0, b + 2.0), device="CPU:0")
        first_x, first_y = next(iter(mapped))

        assert almost_equal(first_x, tf.constant([[2.0], [3.0]]))
        assert almost_equal(first_y, tf.constant([[7.0], [8.0]]))

    def test_cache_save_load_dataset(self, backend, tmp_path):
        """Test caching, saving and loading datasets."""
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant([1, 2]), tf.constant([3, 4]))).batch(1)
        cached = backend.cache_dataset(dataset)

        save_path = str(tmp_path / "tf_dataset")
        backend.save_dataset(cached, save_path)
        loaded = backend.load_dataset(save_path)

        loaded_values = list(loaded.as_numpy_iterator())
        assert len(loaded_values) == 2
        assert np.array_equal(loaded_values[0][0], np.array([1]))

    def test_get_dataset_batch_size_and_cardinality(self, backend):
        """Test dataset batch size and cardinality helpers."""
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3, 4])).batch(2)

        assert backend.get_dataset_batch_size(dataset) == 2
        assert backend.get_dataset_cardinality(dataset) == 2

    def test_zip_batch_unbatch_take_dataset(self, backend):
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

    def test_create_dataset_from_tensors(self, backend):
        """Test creating a dataset from a tuple of tensors."""
        tensors = (tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0]))
        dataset = backend.create_dataset_from_tensors(tensors, batch_size=8)
        element = next(iter(dataset))

        assert element[0].shape == (1, 2)
        assert element[1].shape == (1, 2)

    def test_shuffle_dataset_and_size(self, backend):
        """Test shuffling and counting dataset elements."""
        x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
        y = tf.constant([[5.0], [6.0], [7.0], [8.0]])
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        shuffled = backend.shuffle_dataset(dataset, buffer_size=4)
        shuffled_batched = backend.batch_dataset(shuffled, 2)
        assert backend.get_dataset_size(shuffled_batched) == 4

    def test_get_dataset_element_spec_and_assert_batched(self, backend):
        """Test element_spec lookup and batch assertion."""
        batched = tf.data.Dataset.from_tensor_slices((tf.constant([1, 2]), tf.constant([3, 4]))).batch(1)
        spec = backend.get_dataset_element_spec(batched)
        assert isinstance(spec, tuple)
        assert spec[0].shape == tf.TensorShape([None])

        backend.assert_batched_dataset(batched)

        unbatched = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2]))
        with pytest.raises(ValueError):
            backend.assert_batched_dataset(unbatched)


class TestTensorFlowBackendAdvancedOperations:
    """Test TensorFlow backend operations not covered elsewhere."""

    def test_ones_ones_like_and_argsort(self, backend):
        """Test ones constructors and argsort ordering."""
        ones = backend.ones((2, 3), dtype=backend.float32_dtype())
        ones_like = backend.ones_like(tf.constant([[0.0, 0.0], [0.0, 0.0]]))
        sorted_idx = backend.argsort(tf.constant([3.0, 1.0, 2.0]))

        assert ones.shape == (2, 3)
        assert tf.reduce_all(ones == 1.0)
        assert tf.reduce_all(ones_like == 1.0)
        assert np.array_equal(sorted_idx.numpy(), np.array([1, 2, 0]))

    def test_assign_variable(self, backend, simple_model):
        """Test in-place variable assignment."""
        variable = backend.get_model_weights(simple_model)[0]
        new_value = tf.zeros_like(variable)

        backend.assign_variable(variable, new_value)
        assert tf.reduce_all(variable == 0.0)

    def test_compute_hessian(self, backend):
        """Test Hessian computation shape and finiteness."""
        model = Sequential([Input(shape=(2,)), Dense(1, name='output')])
        weights = backend.get_model_weights(model)
        nb_params = backend.get_num_params(weights)
        loss_fn = MeanSquaredError(reduction=Reduction.NONE)

        inputs = tf.constant([[1.0, 0.0], [0.5, -1.0]], dtype=tf.float32)
        targets = tf.constant([[1.0], [0.0]], dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)

        hessian = backend.compute_hessian(model, weights, loss_fn, dataset, nb_params)

        assert hessian.shape == (nb_params, nb_params)
        assert tf.reduce_all(tf.math.is_finite(hessian))

    def test_compute_hvp_batch(self, backend):
        """Test batched Hessian-vector product computation."""
        model = Sequential([Input(shape=(2,)), Dense(1, name='output')])
        weights = backend.get_model_weights(model)
        nb_params = backend.get_num_params(weights)
        loss_fn = MeanSquaredError(reduction=Reduction.NONE)

        inputs = tf.constant([[1.0, 0.0], [0.5, -1.0]], dtype=tf.float32)
        targets = tf.constant([[1.0], [0.0]], dtype=tf.float32)
        vector = [tf.ones_like(w) for w in weights]

        hvp = backend.compute_hvp_batch(model, weights, loss_fn, vector, inputs, targets)

        assert hvp.shape == (nb_params,)
        assert tf.reduce_all(tf.math.is_finite(hvp))

    def test_compute_output_jacobians(self, backend):
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

    def test_while_loop(self, backend):
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

    def test_random_diag_eig_and_real(self, backend):
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
