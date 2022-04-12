import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, MeanSquaredError

from deel.influenciae.common import InfluenceModel
from deel.influenciae.influence.inverse_hessian_vector_product import ExactIHVP, ConjugateGradientDescentIHVP
from deel.influenciae.influence.first_order_influence_calculator import FirstOrderInfluenceCalculator

from ..utils import almost_equal, jacobian_ground_truth, hessian_ground_truth


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
                                     tf.transpose(tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)))
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
                                     tf.transpose(tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)))
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
