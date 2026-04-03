# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Reduction, CategoricalCrossentropy, BinaryCrossentropy

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVPFactory

from deel.influenciae.rps import RepresenterPointLJE

from ..utils_test import assert_allclose, assert_inheritance, almost_equal, relative_almost_equal


pytestmark = pytest.mark.tensorflow


def test_alpha():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))
    model.add(Dense(4, use_bias=False, dtype=tf.float64))
    loss_function = CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 4), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    target_layer = -1
    influence_model = InfluenceModel(model, start_layer=target_layer, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)

    # Compute alpha using rps_lje
    feature_extractor = Sequential(model.layers[:target_layer])
    feature_maps = feature_extractor(inputs_train)
    alpha = rps_lje._compute_alpha(feature_maps, targets_train)

    # Compute alpha manually
    # First, create the perturbed model
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    perturbed_model = tf.keras.models.clone_model(rps_lje.original_head)
    perturbed_model.build(input_shape=feature_extractor.output_shape)
    perturbed_model.set_weights(rps_lje.original_head.get_weights())
    perturbed_trainable_vars = list(perturbed_model.trainable_variables)

    with tf.GradientTape() as tape:
        logits = perturbed_model(feature_maps)
        loss = loss_function(targets_train, logits)
        loss = rps_lje._ensure_per_sample_loss(loss)
        loss = -tf.reduce_mean(loss)
    grads = tape.gradient(loss, perturbed_trainable_vars)
    optimizer.apply_gradients(zip(grads, perturbed_trainable_vars))

    watched_weights = rps_lje.backend.normalize_weights_to_watch(list(perturbed_model.trainable_weights))

    # Check that manual perturbation matches implementation perturbation.
    impl_weights = rps_lje.backend.normalize_weights_to_watch(list(rps_lje.perturbed_head.trainable_weights))
    assert_allclose(tf.concat(watched_weights, axis=0), tf.concat(impl_weights, axis=0), rtol=2e-5, atol=1e-7)

    # Now, we can compute alpha
    # Start with the second term
    ihvp = rps_lje.ihvp_calculator
    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
        tape.watch(impl_weights)
        logits = rps_lje.perturbed_head(feature_maps)
        loss = loss_function(targets_train, logits)
        loss = rps_lje._ensure_per_sample_loss(loss)
    grads = tape.jacobian(loss, impl_weights)[0]

    grads = tf.multiply(
        grads,
        tf.repeat(
            tf.expand_dims(
                tf.divide(
                    tf.ones_like(feature_maps),
                    tf.cast(tf.shape(feature_maps)[0], feature_maps.dtype) * feature_maps
                    + tf.cast(rps_lje.epsilon, feature_maps.dtype),
                ),
                axis=-1,
            ),
            grads.shape[-1],
            axis=-1,
        ),
    )
    second_term = tf.map_fn(
        lambda v: ihvp._compute_ihvp_single_batch(  # pylint: disable=protected-access
            tf.expand_dims(v, axis=0),
            use_gradient=False,
        ),
        grads,
    )
    second_term = tf.reduce_sum(tf.reshape(second_term, tf.shape(grads)), axis=1)

    # Now, compute the first term
    # first term is weights divided by feature maps
    weights = tf.concat(impl_weights, axis=0)
    first_term = tf.multiply(
        weights,
        tf.repeat(
            tf.expand_dims(
                tf.divide(
                    tf.ones_like(feature_maps),
                    tf.cast(tf.shape(feature_maps)[0], feature_maps.dtype) * feature_maps
                    + tf.cast(rps_lje.epsilon, feature_maps.dtype),
                ),
                axis=-1,
            ),
            weights.shape[-1],
            axis=-1,
        ),
    )
    first_term = tf.reduce_sum(first_term, axis=1)

    # Combine to get alpha_test
    alpha_test = first_term - second_term

    assert alpha.shape == alpha_test.shape
    assert relative_almost_equal(alpha, alpha_test, percent=0.1)  # results tend to contain large numbers, relative makes more sense


def test_alpha_does_not_use_compiled_loss(monkeypatch):
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(3,), dtype=tf.float32))
    model.add(Dense(4, activation='relu', dtype=tf.float32))
    model.add(Dense(2, use_bias=False, dtype=tf.float32))

    def per_sample_mse(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true - y_pred), axis=1)

    _ = model(tf.random.normal((8, 3), dtype=tf.float32))

    inputs_train = tf.random.normal((8, 3), dtype=tf.float32)
    targets_train = tf.random.normal((8, 2), dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(4)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=per_sample_mse)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)

    def _raise_compiled_loss(*_args, **_kwargs):
        raise RuntimeError("compiled_loss should not be called in RPS-LJE alpha computation")

    monkeypatch.setattr(rps_lje.perturbed_head, "compiled_loss", _raise_compiled_loss, raising=False)

    feature_extractor = Sequential(model.layers[:-1])
    feature_maps = feature_extractor(inputs_train)
    alpha = rps_lje._compute_alpha(feature_maps, targets_train)

    assert alpha.shape == (8, 2)


def test_alpha_raises_on_scalar_loss():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(3,), dtype=tf.float32))
    model.add(Dense(4, activation='relu', dtype=tf.float32))
    model.add(Dense(2, use_bias=False, dtype=tf.float32))

    base_loss = CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    _ = model(tf.random.normal((8, 3), dtype=tf.float32))

    inputs_train = tf.random.normal((8, 3), dtype=tf.float32)
    targets_train = tf.random.normal((8, 2), dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(4)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=base_loss)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)

    # Simulate a misconfigured loss that returns a scalar.
    rps_lje.loss_function = lambda y_true, y_pred: tf.reduce_sum(tf.square(y_true - y_pred))

    feature_extractor = Sequential(model.layers[:-1])
    feature_maps = feature_extractor(inputs_train)

    with pytest.raises(ValueError, match="per-sample"):
        rps_lje._compute_alpha(feature_maps, targets_train)


def test_compute_influence_vector():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))
    model.add(Dense(4, use_bias=False, dtype=tf.float64))
    loss_function = CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 4), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    alpha, z_batch = rps_lje._compute_influence_vector((inputs_train, targets_train))

    # Now, compute them manually to check that it is correct
    feature_extractor = Sequential(model.layers[:-1])
    z_batch_test = feature_extractor(inputs_train)
    alpha_test = rps_lje._compute_alpha(z_batch_test, targets_train)

    assert_allclose(z_batch, z_batch_test, rtol=2e-5, atol=1e-7)
    assert_allclose(alpha, alpha_test, rtol=2e-5, atol=1e-7)  # alpha is already tested somewhere else


def test_preprocess_sample_to_evaluate():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))
    model.add(Dense(4, use_bias=False, dtype=tf.float64))
    loss_function = CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 4), dtype=tf.float64)

    inputs_test = tf.random.normal((60, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.random.normal((60, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    pre_evaluate_computed = rps_lje._preprocess_samples((inputs_test, targets_test))

    # Compute the feature maps
    feature_extractor = Sequential(model.layers[:-1])
    feature_maps = feature_extractor(inputs_test)

    # Check that we get the feature maps and the targets
    assert_allclose(pre_evaluate_computed[0], feature_maps, rtol=2e-5, atol=1e-7)
    assert_allclose(pre_evaluate_computed[1], targets_test, rtol=2e-5, atol=1e-7)


def test_compute_influence_value_from_influence_vector_binary():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))
    model.add(Dense(1, use_bias=False, dtype=tf.float64))
    loss_function = BinaryCrossentropy(from_logits=True, reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    # Compute the influence values using RPS-LJE
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    influence_values_computed = rps_lje._compute_influence_value_from_batch((inputs_train, targets_train))

    # Compute the influence values manually
    alpha, z_batch = rps_lje._compute_influence_vector((inputs_train, targets_train))  # already checked in another test
    influence_values = tf.abs(alpha)

    assert almost_equal(influence_values_computed, influence_values, epsilon=1e-3)


def test_compute_influence_value_from_influence_vector_multiclass():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))
    model.add(Dense(4, use_bias=False, dtype=tf.float64))
    loss_function = CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 4), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    # Compute the influence values using RPS-LJE
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    influence_values_computed = rps_lje._compute_influence_value_from_batch((inputs_train, targets_train))

    # Compute the influence values manually
    alpha, z_batch = rps_lje._compute_influence_vector((inputs_train, targets_train))  # already checked in another test
    alpha_i = tf.gather(alpha, tf.argmax(rps_lje.perturbed_head(z_batch), axis=1), axis=1, batch_dims=1)
    influence_values = tf.abs(alpha_i)

    assert relative_almost_equal(influence_values_computed, influence_values, percent=0.05)


def test_compute_pairwise_influence_value_binary():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))
    model.add(Dense(1, use_bias=False, dtype=tf.float64))
    loss_function = BinaryCrossentropy(from_logits=True, reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    v_test = rps_lje._preprocess_samples((inputs_test, targets_test))
    influence_vector = rps_lje._compute_influence_vector((inputs_train, targets_train))
    influence_values_computed = rps_lje._estimate_influence_value_from_influence_vector(v_test, influence_vector)

    # Compute the values manually
    feature_extractor = Sequential(model.layers[:-1])
    alpha_test = influence_vector[0]  # alpha and influence vector are already tested somewhere else
    feature_maps_train = feature_extractor(inputs_train)
    feature_maps_test = feature_extractor(inputs_test)
    influence_values_test = alpha_test * tf.matmul(feature_maps_train, feature_maps_test, transpose_b=True)
    influence_values_test = tf.transpose(influence_values_test)

    assert relative_almost_equal(influence_values_computed, influence_values_test, percent=0.1)


def test_compute_pairwise_influence_value_multiclass():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))
    model.add(Dense(4, use_bias=False, dtype=tf.float64))
    loss_function = CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 4), dtype=tf.float64)

    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.random.normal((50, 4), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    v_test = rps_lje._preprocess_samples((inputs_test, targets_test))
    influence_vector = rps_lje._compute_influence_vector((inputs_train, targets_train))
    influence_values_computed = rps_lje._estimate_influence_value_from_influence_vector(v_test, influence_vector)

    # Compute the values manually
    feature_extractor = Sequential(model.layers[:-1])
    feature_maps_train = feature_extractor(inputs_train)
    feature_maps_test = feature_extractor(inputs_test)
    indices = tf.argmax(rps_lje.original_head(feature_maps_test), axis=1)
    alpha_test = tf.gather(influence_vector[0], indices, axis=1, batch_dims=1)
    # Reshape alpha_test to (n, 1) for proper row-wise broadcasting
    alpha_test = tf.reshape(alpha_test, (-1, 1))
    influence_values_test = alpha_test * tf.matmul(feature_maps_train, feature_maps_test, transpose_b=True)
    influence_values_test = tf.transpose(influence_values_test)

    assert relative_almost_equal(influence_values_computed, influence_values_test, percent=0.1)


def test_inheritance():
    tf.random.set_seed(0)

    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))
    model.add(Dense(1, use_bias=False, dtype=tf.float64))
    loss_function = BinaryCrossentropy(reduction=Reduction.NONE)

    model(tf.random.normal((10, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((10, 1), dtype=tf.float64)

    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)
    test_dataset = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(10)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)

    method = rps_lje

    nb_params = influence_model.nb_params

    assert_inheritance(
        method,
        nb_params,
        train_dataset,
        test_dataset
    )
