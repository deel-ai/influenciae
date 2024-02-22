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

def test__compute_additive_term():
    """ 
    Test the _compute_additive_term method
    """
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    group = train_set.take(5)

    n_samples = 25
    fraction = 0.2

    # Compute the additive term symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train[:5], targets_train[:5])], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)
    coeff_additive_term = (1. - 2. * fraction) / ((1 - fraction)**2 * n_samples)
    gt_additive_term = coeff_additive_term * ground_truth_ihvp

    # Check if the result is the one expected
    calculators = [
        ExactIHVP(influence_model, train_set.batch(5)),
        ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))
    ]

    for ihvp_calculator in calculators:
        influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
        additive_term = influence_calculator._compute_additive_term(group.batch(5))
        assert additive_term.shape == gt_additive_term.shape
        assert almost_equal(gt_additive_term, coeff_additive_term * additive_term, epsilon=1E-3)


def test__compute_pairwise_interactions():
    """
    Test the _compute_pairwise_interactions method
    """
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    group = train_set.take(5)
    n_samples = 25
    fraction = 0.2

    # Compute the pairwise term symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train[:5], targets_train[:5])], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)
    coeff_pairwise = 1. / tf.square((1. - fraction) * n_samples)
    gt_pairwise_term = coeff_pairwise * ground_truth_inv_hessian @ \
                    tf.reduce_sum(tf.squeeze(tf.concat(tf.expand_dims([hessian_ground_truth(tf.squeeze(inp), kernel) @ ground_truth_ihvp
                                             for inp, _ in group], axis=0), axis=0), axis=0), axis=0)

    # Check if the result is the one expected
    calculators = [
        ExactIHVP(influence_model, train_set.batch(5)),
        ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))
    ]

    for ihvp_calculator in calculators:
        influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
        pairwise_term = influence_calculator._compute_pairwise_interactions(group.batch(5))
        pairwise = coeff_pairwise * pairwise_term
        assert pairwise.shape == gt_pairwise_term.shape
        assert almost_equal(gt_pairwise_term, pairwise, epsilon=1E-3)


def test_compute_influence_group():
    """
    Test the compute_influence_group method
    """
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    group = train_set.take(5)

    n_samples = 25
    fraction = 0.2

    # Compute the influence vector symbolically
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

    # Check if the result is the one expected
    calculators = [
        ExactIHVP(influence_model, train_set.batch(5)),
        ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))
    ]

    for ihvp_calculator in calculators:
        influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
        influence_group = influence_calculator.compute_influence_vector_group(group.batch(5))
        assert influence_group.shape == (1, 2)
        assert almost_equal(influence_group, ground_truth_influence_group, epsilon=1e-3)


def test_compute_influence_values_group():
    """
    Test the compute_influence_values_group method
    """
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_test = tf.random.normal((25, 1))

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    n_samples = 25
    fraction = 0.2

    # Compute the influence vector symbolically
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
    ground_truth_self_influence = tf.matmul(reduced_ground_truth_grads, ground_truth_influence_group,
                                            transpose_a=True, transpose_b=True)

    # Check if the result is the one expected
    calculators = [
        ExactIHVP(influence_model, train_set.batch(5)),
        ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))
    ]

    for ihvp_calculator in calculators:
        influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                          n_samples_for_hessian=25,
                                                          shuffle_buffer_size=25)
        influence_group_values = influence_calculator.estimate_influence_values_group(train_set.take(5).batch(5),
                                                                                      test_set.take(5).batch(5))
        assert influence_group_values.shape == (1, 1)
        assert almost_equal(influence_group_values, ground_truth_influence_values, epsilon=1e-3)

        self_influence_group = influence_calculator.estimate_influence_values_group(train_set.take(5).batch(5))
        assert self_influence_group.shape == (1, 1)
        assert almost_equal(self_influence_group, ground_truth_self_influence, epsilon=1e-3)


def test_cnn_shapes():
    """
    Test all methods with a more challenging model
    """
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
    calculators = [
        ExactIHVP(influence_model, train_set.batch(5)),
        ConjugateGradientDescentIHVP(influence_model, -2, train_set.batch(5))
    ]

    for ihvp_calculator in calculators:
        influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                            n_samples_for_hessian=25,
                                                            shuffle_buffer_size=25)
        influence = influence_calculator.compute_influence_vector_group(train_set.batch(5))
        assert influence.shape == (1, 650)
        influence_values = influence_calculator.estimate_influence_values_group(train_set.batch(5), test_set.batch(5))
        assert influence_values.shape == (1, 1)
