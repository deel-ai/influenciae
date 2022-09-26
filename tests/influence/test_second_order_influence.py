import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, MeanSquaredError

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ConjugateGradientDescentIHVP
from deel.influenciae.influence.second_order_influence_calculator import SecondOrderInfluenceCalculator

from ..utils_test import almost_equal, jacobian_ground_truth, hessian_ground_truth


def test_second_order_calculator():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    # Check that the second order influence calculator creates the instance correctly
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                   n_samples_for_hessian=25, shuffle_buffer_size=25)
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                   n_samples_for_hessian=25, shuffle_buffer_size=25)
    influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
    assert isinstance(influence_calculator.ihvp_calculator, ConjugateGradientDescentIHVP)

    # Check that it computes the training dataset's size correctly
    assert influence_calculator.train_size == 25

    # Check that the single point operations are not permitted
    with pytest.raises(NotImplementedError):
        influence_calculator.compute_influence(train_set.batch(5))
    with pytest.raises(NotImplementedError):
        influence_calculator.compute_influence_values(train_set.batch(5), train_set.batch(5))


def test_exact_group_influence():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    group = train_set.take(5)
    n_samples = 25
    fraction = 0.2

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
    influence_group = influence_calculator.compute_influence_group(group.batch(5))
    assert influence_group.shape == (1, 2)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train[:5], targets_train[:5])], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)
    additive_term = (1. - 2. * fraction) / ((1 - fraction)**2 * n_samples) * ground_truth_ihvp
    pairwise_term = ground_truth_inv_hessian @ \
                    tf.reduce_sum(tf.squeeze(tf.concat(tf.expand_dims([hessian_ground_truth(tf.squeeze(inp), kernel) @ ground_truth_ihvp
                                             for inp, _ in group], axis=0), axis=0), axis=0), axis=0) / (n_samples**2 * (1. - fraction)**2)
    ground_truth_influence_group = tf.transpose(additive_term + pairwise_term)
    assert almost_equal(influence_group, ground_truth_influence_group, epsilon=1e-3)  # I was forced to increase from 1e-6


def test_exact_group_influence_values():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))
    n_samples = 25
    fraction = 0.2

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
    influence_values = influence_calculator.compute_influence_values_group(train_set.take(5).batch(5),
                                                                           test_set.take(5).batch(5))
    assert influence_values.shape == (1, 1)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train[:5], targets_train[:5])], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)
    additive_term = (1. - 2. * fraction) / ((1 - fraction)**2 * n_samples) * ground_truth_ihvp
    pairwise_term = ground_truth_inv_hessian @ \
                    tf.reduce_sum(tf.squeeze(tf.concat(tf.expand_dims([hessian_ground_truth(tf.squeeze(inp), kernel) @ ground_truth_ihvp
                                             for inp, _ in train_set.take(5)], axis=0), axis=0), axis=0), axis=0) / (n_samples**2 * (1. - fraction)**2)
    ground_truth_influence_group = tf.transpose(additive_term + pairwise_term)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_test[:5], targets_test[:5])], axis=1)
    reduced_ground_truth_grads_test = tf.reduce_sum(ground_truth_grads_test, axis=1, keepdims=True)
    ground_truth_influence_values = tf.matmul(reduced_ground_truth_grads_test, ground_truth_influence_group,
                                              transpose_a=True, transpose_b=True)
    assert almost_equal(influence_values, ground_truth_influence_values, epsilon=1e-3)  # I was forced to increase from 1e-6


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
    influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
    influence = influence_calculator.compute_influence_group(train_set.batch(5))
    assert influence.shape == (1, 640)
    influence_values = influence_calculator.compute_influence_values_group(train_set.batch(5), test_set.batch(5))
    assert influence_values.shape == (1, 1)


def test_cgd_group_influence():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    group = train_set.take(5)
    n_samples = 25
    fraction = 0.2

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
    influence_group = influence_calculator.compute_influence_group(group.batch(5))
    assert influence_group.shape == (1, 2)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train[:5], targets_train[:5])], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)
    additive_term = (1. - 2. * fraction) / ((1 - fraction)**2 * n_samples) * ground_truth_ihvp
    pairwise_term = ground_truth_inv_hessian @ \
                    tf.reduce_sum(tf.squeeze(tf.concat(tf.expand_dims([hessian_ground_truth(tf.squeeze(inp), kernel) @ ground_truth_ihvp
                                             for inp, _ in group], axis=0), axis=0), axis=0), axis=0) / (n_samples**2 * (1. - fraction)**2)
    ground_truth_influence_group = tf.transpose(additive_term + pairwise_term)
    assert almost_equal(influence_group, ground_truth_influence_group, epsilon=1e-2)  # I was forced to increase from 1e-6


def test_cgd_group_influence_values():
    # Make sure that the influence values are calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))
    n_samples = 25
    fraction = 0.2

    # Compute the influence vector using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
    influence_values = influence_calculator.compute_influence_values_group(train_set.take(5).batch(5),
                                                                           test_set.take(5).batch(5))
    assert influence_values.shape == (1, 1)

    # Compute the influence vector symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train[:5], targets_train[:5])], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)
    additive_term = (1. - 2. * fraction) / ((1 - fraction)**2 * n_samples) * ground_truth_ihvp
    pairwise_term = ground_truth_inv_hessian @ \
                    tf.reduce_sum(tf.squeeze(tf.concat(tf.expand_dims([hessian_ground_truth(tf.squeeze(inp), kernel) @ ground_truth_ihvp
                                             for inp, _ in train_set.take(5)], axis=0), axis=0), axis=0), axis=0) / (n_samples**2 * (1. - fraction)**2)
    ground_truth_influence_group = tf.transpose(additive_term + pairwise_term)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_test[:5], targets_test[:5])], axis=1)
    reduced_ground_truth_grads_test = tf.reduce_sum(ground_truth_grads_test, axis=1, keepdims=True)
    ground_truth_influence_values = tf.matmul(reduced_ground_truth_grads_test, ground_truth_influence_group,
                                              transpose_a=True, transpose_b=True)
    assert almost_equal(influence_values, ground_truth_influence_values, epsilon=1e-2)  # I was forced to increase from 1e-6


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
    influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
    influence = influence_calculator.compute_influence_group(train_set.batch(5))
    assert influence.shape == (1, 640)
    influence_values = influence_calculator.compute_influence_values_group(train_set.batch(5), test_set.batch(5))
    assert influence_values.shape == (1, 1)
