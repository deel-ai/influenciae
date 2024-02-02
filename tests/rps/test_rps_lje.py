# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Reduction, CategoricalCrossentropy, BinaryCrossentropy

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ExactIHVPFactory

from deel.influenciae.rps import RepresenterPointLJE

from ..utils_test import assert_inheritance, almost_equal, relative_almost_equal


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
    perturbed_model = Sequential(model.layers[target_layer:])
    perturbed_model.build(input_shape=feature_extractor.output_shape)
    with tf.GradientTape() as tape:
        tape.watch(perturbed_model.weights)
        logits = perturbed_model(feature_maps)
        loss = tf.reduce_mean(-loss_function(targets_train, logits))
    grads = tape.gradient(loss, perturbed_model.weights)
    optimizer.apply_gradients(zip(grads, perturbed_model.weights))

    # Now, we can compute alpha
    # Start with the second term
    dataset_for_hessian = tf.data.Dataset.from_tensor_slices((feature_maps, targets_train)).batch(5)
    ihvp = ExactIHVP(InfluenceModel(perturbed_model, start_layer=0, loss_function=loss_function), dataset_for_hessian)
    with tf.GradientTape() as tape:
        tape.watch(perturbed_model.weights)
        logits = perturbed_model(feature_maps)
        loss = loss_function(targets_train, logits)
    grads = tape.jacobian(loss, perturbed_model.weights)[0]

    # Divide grads by feature maps
    grads_div_feature_maps = []
    for i in range(inputs_train.shape[0]):
        feature_map = tf.reshape(feature_maps[i], (-1, 1)) if len(feature_maps[i].shape) == 1 else feature_maps[i]
        divisor = tf.tile(
            tf.cast(tf.shape(feature_map)[0], feature_map.dtype) * feature_map +
            tf.constant(1e-5, dtype=feature_map.dtype),
            (1, grads.shape[-1])
        )
        grads_div_feature_maps.append(tf.divide(grads[i], divisor))
    grads_div_feature_maps = tf.convert_to_tensor(grads_div_feature_maps)
    second_term = []
    for i in range(inputs_train.shape[0]):
        second_term.append(ihvp._compute_ihvp_single_batch(
            tf.expand_dims(grads_div_feature_maps[i], axis=0), use_gradient=False
        ))
    second_term = tf.convert_to_tensor(second_term)
    second_term = tf.reshape(second_term, grads.shape)
    second_term = tf.reduce_sum(second_term, axis=1)

    # Now, compute the first term
    # first term is weights divided by feature maps
    weights = [w for w in perturbed_model.weights]
    first_term = []
    for i in range(inputs_train.shape[0]):
        feature_map = tf.reshape(feature_maps[i], (-1, 1)) if len(feature_maps[i].shape) == 1 else feature_maps[i]
        divisor = tf.tile(
            tf.cast(tf.shape(feature_map)[0], feature_map.dtype) * feature_map +
            tf.constant(1e-5, dtype=feature_map.dtype),
            (1, grads.shape[-1])
        )
        first_term.append(tf.divide(weights, divisor))
    first_term = tf.convert_to_tensor(first_term)
    first_term = tf.reshape(first_term, grads.shape)
    first_term = tf.reduce_sum(first_term, axis=1)

    # Combine to get alpha_test
    alpha_test = first_term - second_term

    assert alpha.shape == alpha_test.shape
    assert relative_almost_equal(alpha, alpha_test, percent=0.1)  # results tend to contain large numbers, relative makes more sense


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

    assert almost_equal(z_batch, z_batch_test)
    assert almost_equal(alpha, alpha_test)  # alpha is already tested somewhere else


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
    assert almost_equal(pre_evaluate_computed[0], feature_maps)
    assert almost_equal(pre_evaluate_computed[1], targets_test)


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
    influence_values = alpha

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
    influence_values = alpha_i

    assert almost_equal(influence_values_computed, influence_values, epsilon=1e-3)


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
    indices = tf.argmax(rps_lje.perturbed_head(feature_maps_test), axis=1)
    alpha_test = tf.gather(influence_vector[0], indices, axis=1, batch_dims=1)
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
