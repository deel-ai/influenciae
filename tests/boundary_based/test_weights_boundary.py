# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
import numpy as np

from deel.influenciae.boundary_based import WeightsBoundaryCalculator


def test_compute_influence_shape():
    model = Sequential()
    model.add(Input(shape=(3,)))
    model.add(Dense(2, kernel_initializer=tf.constant_initializer([[1, 1, 1], [0, 0, 0]]),
                    bias_initializer=tf.constant_initializer([4.0, 0.0])))

    calculator = WeightsBoundaryCalculator(model)

    inputs_train = tf.random.normal((10, 3))
    targets_train = tf.one_hot(tf.zeros((10,), dtype=tf.int32), 2)
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_computed_score = calculator._compute_influence_values(train_set)

    assert tf.reduce_all(influence_computed_score.shape == (10, 1))


def test_compute_influence_values():
    model = Sequential()
    model.add(Input(shape=(3,)))
    model.add(Dense(2, kernel_initializer=tf.constant_initializer([[1, 1, 1], [0, 0, 0]]),
                    bias_initializer=tf.constant_initializer([4.0, 0.0])))

    calculator = WeightsBoundaryCalculator(model)

    inputs_train = tf.zeros((1, 3))
    targets_train = tf.one_hot(tf.zeros((1,), dtype=tf.int32), 2)
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(1)

    influence_computed_score = calculator._compute_influence_values(train_set)

    # modify the bias term to get equal logits
    influence_computed_expected = tf.convert_to_tensor([[-np.sqrt(2.0) * 2.0]], dtype=tf.float32)

    assert tf.reduce_max(tf.abs(influence_computed_expected - influence_computed_score)) < 1E-6
