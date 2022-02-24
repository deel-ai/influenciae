import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import (Reduction, MeanSquaredError)

from influenciae.common import InfluenceModel
from influenciae.influence.inverse_hessian_vector_product import ExactIHVP, ConjugateGradientDescentIHVP

from ..utils import almost_equal, jacobian_ground_truth, hessian_ground_truth


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
    assert almost_equal(ihvp, ground_truth_ihvp, epsilon=1e-3)


def test_exact_hvp():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((5, 1, 3))
    target = tf.random.normal((5, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the IHVP using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    hvp = ihvp_calculator.compute_hvp(train_set.batch(5))
    assert hvp.shape == (2, 5)  # 5 times (2, 1) stacked on the last axis

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_hessian = tf.reduce_mean(hessian_list, axis=0)
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_hvp = tf.matmul(ground_truth_hessian, ground_truth_grads)
    assert almost_equal(hvp, ground_truth_hvp, epsilon=1e-4)


def test_cgd_hvp():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((5, 1, 3))
    target = tf.random.normal((5, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the IHVP using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    hvp = ihvp_calculator.compute_hvp(train_set.batch(5))
    assert hvp.shape == (2, 5)  # 5 times (2, 1) stacked on the last axis

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_hessian = tf.reduce_mean(hessian_list, axis=0)
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_hvp = tf.matmul(ground_truth_hessian, ground_truth_grads)
    assert almost_equal(hvp, ground_truth_hvp, epsilon=1e-4)


def test_cgd_ihvp():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((5, 1, 3))
    target = tf.random.normal((5, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the IHVP using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, train_set.batch(5))
    hvp = ihvp_calculator.compute_ihvp(train_set.batch(5))
    assert hvp.shape == (2, 5)  # 5 times (2, 1) stacked on the last axis

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)
    assert almost_equal(hvp, ground_truth_ihvp, epsilon=1e-4)
