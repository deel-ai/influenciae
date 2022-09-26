# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Reduction, MeanSquaredError

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP
from deel.influenciae.rps.rps_lje import RPSLJE

from ..utils_test import assert_inheritance

def test_compute_influence_vector():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    ihvp_calculator = ExactIHVP(influence_model, train_dataset.batch(5))
    rps_lje = RPSLJE(influence_model, ihvp_calculator, target_layer=-1)

    f_train = model_feature(inputs_train)
    g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train
    ihvp = tf.ones((50, 64), dtype=tf.float64) - (tf.matmul(g_train, ihvp_calculator.inv_hessian))

    ihvp_computed = rps_lje.compute_influence_vector((inputs_train, targets_train))
    assert ihvp_computed.shape == (50, 64) # (nb_inputs, nb_params)
    assert tf.reduce_max(tf.abs(ihvp_computed - ihvp)) < 1E-6

def test_preprocess_sample_to_evaluate():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    ihvp_calculator = ExactIHVP(influence_model, train_dataset.batch(5))
    rps_lje = RPSLJE(influence_model, ihvp_calculator, target_layer=-1)

    f_train = model_feature(inputs_train)
    f_train_computed = rps_lje.preprocess_sample_to_evaluate((inputs_train, targets_train))
    assert f_train_computed.shape == (50, 64)
    assert tf.reduce_max(tf.abs(f_train_computed - f_train)) < 1E-6

def test_compute_influence_value_from_influence_vector():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)
    inputs_test = tf.random.normal((60, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.random.normal((60, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    ihvp_calculator = ExactIHVP(influence_model, train_dataset.batch(5))
    rps_lje = RPSLJE(influence_model, ihvp_calculator, target_layer=-1)

    f_train = model_feature(inputs_train)
    g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train
    ihvp = tf.ones((50, 64), dtype=tf.float64) - (tf.matmul(g_train, ihvp_calculator.inv_hessian))

    v_test = rps_lje.preprocess_sample_to_evaluate((inputs_test, targets_test))
    influence_values_expected = tf.matmul(v_test, tf.transpose(ihvp))

    influence_values_computed = rps_lje.compute_influence_value_from_influence_vector(v_test, ihvp)
    assert influence_values_computed.shape == (60, 50)
    assert tf.reduce_max(tf.abs(influence_values_computed - influence_values_expected)) < 1E-6

def test_compute_pairwise_influence_value():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    ihvp_calculator = ExactIHVP(influence_model, train_dataset.batch(5))
    rps_lje = RPSLJE(influence_model, ihvp_calculator, target_layer=-1)

    f_train = model_feature(inputs_train)
    g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train
    ihvp = tf.ones((50, 64), dtype=tf.float64) - (tf.matmul(g_train, ihvp_calculator.inv_hessian))

    expected_values = tf.reduce_sum(tf.multiply(ihvp, f_train), axis=1, keepdims=True)
    computed_values = rps_lje.compute_pairwise_influence_value((inputs_train, targets_train))
    assert computed_values.shape == (50, 1)
    assert tf.reduce_max(tf.abs(computed_values - expected_values)) < 1E-6

def test_inheritance():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    if_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    inputs_train = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((10, 1), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(10)

    ihvp_calculator = ExactIHVP(if_model, train_set)
    rps_lje = RPSLJE(if_model, ihvp_calculator, target_layer=-1)
    method = rps_lje

    nb_params = if_model.nb_params

    assert_inheritance(
        method,
        nb_params,
        train_set,
        test_set
    )
