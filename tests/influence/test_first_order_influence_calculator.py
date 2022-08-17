# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, MeanSquaredError

from deel.influenciae.common import InfluenceModel
from deel.influenciae.influence.inverse_hessian_vector_product import ExactIHVP, ConjugateGradientDescentIHVP
from deel.influenciae.influence.first_order_influence_calculator import FirstOrderInfluenceCalculator

from ..utils import almost_equal, jacobian_ground_truth, hessian_ground_truth


def test_exact_top_k():
    # Test the shapes of the different quantities
    model = Sequential()
    model.add(Input(shape=(5, 5, 3)))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(10))
    model.compile(loss=CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE), optimizer='sgd')

    model(tf.random.normal((2, 5, 5, 3)))

    influence_model = InfluenceModel(model,
                                     loss_function=CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE))

    x_train = tf.random.normal((50, 5, 5, 3))
    y_train = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    """
    class MockIHVP():
        def compute_ihvp(self,dataset):
            result = None
            for b in dataset:
                if result != None:
                    raise Exception()
                else
                    result = b
            return result
    ihvp_calculator = MockIHVP()
    """

    # Check the shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=True)
    top_k = 7
    expected_influences = []
    expected_values = []

    x_test_batch = []
    y_test_batch = []

    for _ in range(9):
        x_test = tf.random.normal((1, 5, 5, 3))
        y_test = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 1)), 10)
        x_test_batch.append(x_test)
        y_test_batch.append(y_test)

        x_test = tf.repeat(x_test, 50, axis=0)
        y_test = tf.repeat(y_test, 50, axis=0)

        test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        influence_values = influence_calculator.compute_influence_values(train_set.batch(5), test_set.batch(5))
        indexes = tf.squeeze(tf.argsort(influence_values, axis=0), axis=-1).numpy()
        k = tf.gather(influence_values, indexes)[::-1][:top_k]
        v = tf.gather(x_train, indexes)[::-1][:top_k]
        expected_influences.append(tf.expand_dims(k, axis=0))
        expected_values.append(tf.expand_dims(v, axis=0))

    expected_influences = tf.concat(expected_influences, axis=0)
    expected_values = tf.concat(expected_values, axis=0)

    x_test_batch = tf.concat(x_test_batch, axis=0)
    y_test_batch = tf.concat(y_test_batch, axis=0)

    computed_influences, computed_values = influence_calculator.top_k((x_test_batch, y_test_batch), train_set.batch(5),
                                                                      top_k)
    expected_influences = tf.squeeze(expected_influences, axis=-1)
    assert tf.reduce_max(tf.abs(computed_influences - expected_influences)) < 1E-6
    assert tf.reduce_max(tf.abs(computed_values - expected_values)) < 1E-6


def test_exact_cnn_shapes_with_normalization_with_normalization():
    # Test the shapes of the different quantities
    model = Sequential()
    model.add(Input(shape=(5, 5, 3)))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(10))
    model.compile(loss=CategoricalCrossentropy(from_logits=False, reduction=Reduction.NONE), optimizer='sgd')
    influence_model = InfluenceModel(model)

    x_train = tf.random.normal((50, 5, 5, 3))
    y_train = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    x_test = tf.random.normal((50, 5, 5, 3))
    y_test = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Check the shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=True)
    influence_values = influence_calculator.compute_influence_values(train_set.batch(5), test_set.batch(5))
    assert influence_values.shape == (50, 1)
    influence = influence_calculator.compute_influence(train_set.batch(5))
    assert influence.shape == (50, 640)
    influence = influence_calculator.compute_influence_group(train_set.batch(5))
    assert influence.shape == (1, 640)
    influence_values = influence_calculator.compute_influence_values_group(train_set.batch(5), test_set.batch(5))
    assert influence_values.shape == (1, 1)


def test_exact_influence_values_group_with_normalization():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=True)
    influence = influence_calculator.compute_influence_values_group(train_set.batch(25), test_set.batch(25))
    assert influence.shape == (1, 1)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    ground_truth_grads_train = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                         for inp, y in zip(inputs_test, targets_test)], axis=1)
    ground_truth_grads_test = tf.reduce_sum(ground_truth_grads_test, axis=1, keepdims=True)

    ihvp = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    ihvp = ihvp / tf.norm(ihvp, axis=0, keepdims=True)

    ground_truth_influence_values_group = tf.matmul(ground_truth_grads_test,
                                                    ihvp,
                                                    transpose_a=True)

    assert almost_equal(influence, ground_truth_influence_values_group, epsilon=1e-2)


def test_exact_influence_values_with_normalization():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=True)
    influence = influence_calculator.compute_influence_values(train_set.batch(25), test_set.batch(25))
    assert influence.shape == (25, 1)  # 25 times a scalar (1, 1)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                         for inp, y in zip(inputs_test, targets_test)], axis=1)

    ihvp = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    ihvp = ihvp / tf.norm(ihvp, axis=0, keepdims=True)

    ground_truth_influence_values = tf.keras.backend.batch_dot(tf.transpose(ground_truth_grads_test),
                                                               tf.transpose(ihvp))

    assert almost_equal(influence, ground_truth_influence_values, epsilon=1e-3)


def test_exact_influence_group():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=True)
    influence_group = influence_calculator.compute_influence_group(train_set.batch(25))
    assert influence_group.shape == (1, 2)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)

    ground_truth_influence_group = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)

    ground_truth_influence_group = ground_truth_influence_group / tf.norm(ground_truth_influence_group, axis=0,
                                                                          keepdims=True)

    assert almost_equal(influence_group, tf.transpose(ground_truth_influence_group), epsilon=1e-3)


def test_exact_influence_with_normalization():
    # Make sure that the influence vector is calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=True)
    influence = influence_calculator.compute_influence(train_set.batch(25))
    assert influence.shape == (25, 2)  # 25 times (2, 1) stacked on the last axis

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_influence = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)
    ground_truth_influence = ground_truth_influence / tf.norm(ground_truth_influence, axis=0, keepdims=True)
    assert almost_equal(influence, tf.transpose(ground_truth_influence), epsilon=1e-3)


def test_exact_influence():
    # Make sure that the influence vector is calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence = influence_calculator.compute_influence(train_set.batch(25))
    assert influence.shape == (25, 2)  # 25 times (2, 1) stacked on the last axis

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_influence = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)
    assert almost_equal(influence, tf.transpose(ground_truth_influence), epsilon=1e-3)


def test_exact_influence_values():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence = influence_calculator.compute_influence_values(train_set.batch(25), test_set.batch(25))
    assert influence.shape == (25, 1)  # 25 times a scalar (1, 1)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                         for inp, y in zip(inputs_test, targets_test)], axis=1)
    ground_truth_influence_values = tf.keras.backend.batch_dot(tf.transpose(ground_truth_grads_test),
                                                               tf.transpose(tf.matmul(ground_truth_inv_hessian,
                                                                                      ground_truth_grads_train)))
    assert almost_equal(influence, ground_truth_influence_values, epsilon=1e-3)


def test_exact_influence_group():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence_group = influence_calculator.compute_influence_group(train_set.batch(25))
    assert influence_group.shape == (1, 2)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_influence_group = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)
    assert almost_equal(influence_group, tf.transpose(ground_truth_influence_group), epsilon=1e-3)


def test_exact_influence_values_group():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence = influence_calculator.compute_influence_values_group(train_set.batch(25), test_set.batch(25))
    assert influence.shape == (1, 1)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    ground_truth_grads_train = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                         for inp, y in zip(inputs_test, targets_test)], axis=1)
    ground_truth_grads_test = tf.reduce_sum(ground_truth_grads_test, axis=1, keepdims=True)
    ground_truth_influence_values_group = tf.matmul(ground_truth_grads_test,
                                                    tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train),
                                                    transpose_a=True)

    assert almost_equal(influence, ground_truth_influence_values_group, epsilon=1e-2)


def test_exact_cnn_shapes():
    # Test the shapes of the different quantities
    model = Sequential()
    model.add(Input(shape=(5, 5, 3)))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(10))
    model.compile(loss=CategoricalCrossentropy(from_logits=False, reduction=Reduction.NONE), optimizer='sgd')
    influence_model = InfluenceModel(model)

    x_train = tf.random.normal((50, 5, 5, 3))
    y_train = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    x_test = tf.random.normal((50, 5, 5, 3))
    y_test = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Check the shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence_values = influence_calculator.compute_influence_values(train_set.batch(5), test_set.batch(5))
    assert influence_values.shape == (50, 1)
    influence = influence_calculator.compute_influence(train_set.batch(5))
    assert influence.shape == (50, 640)
    influence = influence_calculator.compute_influence_group(train_set.batch(5))
    assert influence.shape == (1, 640)
    influence_values = influence_calculator.compute_influence_values_group(train_set.batch(5), test_set.batch(5))
    assert influence_values.shape == (1, 1)


def test_cgd_influence():
    # Make sure that the influence vector is calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence = influence_calculator.compute_influence(train_set.batch(25))
    assert influence.shape == (25, 2)  # 25 times (2, 1) stacked on the last axis

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_influence = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)
    assert almost_equal(influence, tf.transpose(ground_truth_influence), epsilon=1e-2)


def test_cgd_influence_values():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence = influence_calculator.compute_influence_values(train_set.batch(25), test_set.batch(25))
    assert influence.shape == (25, 1)  # 25 times a scalar (1, 1)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                         for inp, y in zip(inputs_test, targets_test)], axis=1)
    ground_truth_influence_values = tf.keras.backend.batch_dot(tf.transpose(ground_truth_grads_test),
                                                               tf.transpose(tf.matmul(ground_truth_inv_hessian,
                                                                                      ground_truth_grads_train)))
    assert almost_equal(influence, ground_truth_influence_values, epsilon=1e-2)


def test_cgd_influence_group():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence_group = influence_calculator.compute_influence_group(train_set.batch(25))
    assert influence_group.shape == (1, 2)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_influence_group = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)
    assert almost_equal(influence_group, tf.transpose(ground_truth_influence_group), epsilon=1e-2)


def test_cgd_influence_values_group():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence = influence_calculator.compute_influence_values_group(train_set.batch(25), test_set.batch(25))
    assert influence.shape == (1, 1)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    ground_truth_grads_train = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                         for inp, y in zip(inputs_test, targets_test)], axis=1)
    ground_truth_grads_test = tf.reduce_sum(ground_truth_grads_test, axis=1, keepdims=True)
    ground_truth_influence_values_group = tf.matmul(ground_truth_grads_test,
                                                    tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train),
                                                    transpose_a=True)
    assert almost_equal(influence, ground_truth_influence_values_group, epsilon=1e-2)


def test_cgd_cnn_shapes():
    # Test the shapes of the different quantities
    model = Sequential()
    model.add(Input(shape=(5, 5, 3)))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(10))
    model.compile(loss=CategoricalCrossentropy(from_logits=False, reduction=Reduction.NONE), optimizer='sgd')
    influence_model = InfluenceModel(model)

    x_train = tf.random.normal((50, 5, 5, 3))
    y_train = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    x_test = tf.random.normal((50, 5, 5, 3))
    y_test = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Check the shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25)
    influence_values = influence_calculator.compute_influence_values(train_set.batch(5), test_set.batch(5))
    assert influence_values.shape == (50, 1)
    influence = influence_calculator.compute_influence(train_set.batch(5))
    assert influence.shape == (50, 640)
    influence = influence_calculator.compute_influence_group(train_set.batch(5))
    assert influence.shape == (1, 640)
    influence_values = influence_calculator.compute_influence_values_group(train_set.batch(5), test_set.batch(5))
    assert influence_values.shape == (1, 1)
