# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import (Reduction, MeanSquaredError)

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import compute_hessian_block

from ..utils import almost_equal, hessian_ground_truth

def test_hessian_value_and_shape():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((5, 1, 3))
    target = tf.random.normal((5, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the hessian calculation
    sum_hessians, nb_hessians = compute_hessian_block(
        influence_model, train_set.batch(1), parallel_iter=10
    )
    estimated_hessian = tf.squeeze(sum_hessians/nb_hessians)
    assert estimated_hessian.shape == (2, 2)  # 2 parameters (Dense(2)->Dense(1))

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_hessian = tf.reduce_mean(hessian_list, axis=0)
    assert almost_equal(estimated_hessian, ground_truth_hessian, epsilon=1e-2)

def test_split_trainset():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((5, 1, 3))
    target = tf.random.normal((5, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    block_1 = train_set.take(3)
    block_2 = train_set.skip(3)
    # Compute the hessians calculations
    sum_hessians1, nb_hessians1 = compute_hessian_block(
        influence_model, block_1.batch(1), parallel_iter=1
    )
    assert tf.squeeze(sum_hessians1).shape == (2,2) # 2 parameters (Dense(2)->Dense(1))
    assert nb_hessians1 == 3

    sum_hessians2, nb_hessians2 = compute_hessian_block(
        influence_model, block_2.batch(1), parallel_iter=1
    )
    assert tf.squeeze(sum_hessians2).shape == (2,2) # 2 parameters (Dense(2)->Dense(1))
    assert nb_hessians2 == 2

    nb_hessians = nb_hessians1 + nb_hessians2
    estimated_hessian = tf.squeeze((sum_hessians1 + sum_hessians2)/nb_hessians)
    assert estimated_hessian.shape == (2, 2)  # 2 parameters (Dense(2)->Dense(1))

    # Compute the ground truth hessian
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_hessian = tf.reduce_mean(hessian_list, axis=0)
    assert almost_equal(estimated_hessian, ground_truth_hessian, epsilon=1e-2)
