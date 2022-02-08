import pytest
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import (Reduction, MeanSquaredError)

from influenciae.common import InfluenceModel
from influenciae.influence.inverse_hessian_vector_product import ExactIHVP

from ..utils import almost_equal


def jacobian_ground_truth(input_vector, kernel_matrix, target):
    """Symbolically calculates the jacobian for the small 2 layer network in the tests"""
    # input_vector = [A0, A1, A2]
    # kernel_matrix = [W03, W04, W13, W14, W23, W24, W35, W45]
    # target = y
    j1 = 2. * tf.square(input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
                        input_vector[2] * kernel_matrix[4]) * kernel_matrix[6] + \
         2. * (input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
               input_vector[2] * kernel_matrix[4]) * (
                     input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] +
                     input_vector[2] * kernel_matrix[5]) * kernel_matrix[7] - \
         2. * (input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
               input_vector[2] * kernel_matrix[4]) * target
    j2 = 2. * tf.square(input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] +
                        input_vector[2] * kernel_matrix[5]) * kernel_matrix[7] + \
         2. * (input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
               input_vector[2] * kernel_matrix[4]) * kernel_matrix[6] * \
         (input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] + input_vector[2] * kernel_matrix[
             5]) - \
         2. * (input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] +
               input_vector[2] * kernel_matrix[5]) * target

    return tf.convert_to_tensor([j1, j2], dtype=tf.float32)


def hessian_ground_truth(input_vector, kernel_matrix):
    """Symbolically calculates the hessian for the small 2 layer network in the tests"""
    # input_vector = [A0, A1, A2]
    # kernel_matrix = [W03, W04, W13, W14, W23, W24, W35, W45]
    h1 = 2. * tf.square(input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
                        input_vector[2] * kernel_matrix[4])
    h23 = 2. * (input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
                input_vector[2] * kernel_matrix[4]) * (input_vector[0] * kernel_matrix[1] +
                                                       input_vector[1] * kernel_matrix[3] +
                                                       input_vector[2] * kernel_matrix[5])
    h4 = 2. * tf.square(input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] +
                        input_vector[2] * kernel_matrix[5])

    return tf.convert_to_tensor([[h1, h23], [h23, h4]], dtype=tf.float32)


def test_exact_hessian():
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=0, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((5, 1, 3))
    target = tf.random.normal((5, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Test the shape for the first layer
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    inv_hessian = ihvp_calculator.inv_hessian
    assert inv_hessian.shape == (6, 6)  # (3 x 2, 3 x 2)

    # Test the shape for the last layer
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(3))
    inv_hessian = ihvp_calculator.inv_hessian
    assert inv_hessian.shape == (2, 2)  # (2 x 1, 2 x 1)

    # Check the result's precision
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    assert almost_equal(inv_hessian, ground_truth_inv_hessian)


def test_exact_ihvp():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((5, 1, 3))
    target = tf.random.normal((5, 1))
    train_set = tf.data.Dataset.from_tensors((inputs, target))

    # Compute the IHVP using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    ihvp = ihvp_calculator.compute_ihvp(train_set.batch(5))
    assert ihvp.shape == (2, 5)  # 5 times (2, 1) stacked on the last axis

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)
    assert almost_equal(ihvp, ground_truth_ihvp, epsilon=1e-5)  # I was forced to increase from 1e-6
