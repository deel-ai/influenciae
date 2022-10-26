# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Reduction, MeanSquaredError, BinaryCrossentropy

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ExactIHVPFactory

from deel.influenciae.rps import RepresenterPointLJE
from deel.influenciae.influence import FirstOrderInfluenceCalculator

from ..utils_test import assert_inheritance

def test_compute_influence_vector():
    tf.random.set_seed(0)

    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    binary = True
    if binary:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64, activation='sigmoid')])
        loss_function = BinaryCrossentropy(reduction=Reduction.NONE)
    else:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64)])
        loss_function = MeanSquaredError(reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    ihvp_computed = rps_lje._compute_influence_vector((inputs_train, targets_train))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    ihvp_calculator = ExactIHVP(influence_model, train_dataset)
    first_order = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)

    vect = first_order._compute_influence_vector((inputs_train, targets_train))
    weight = model.layers[-1].weights[0]
    vect = tf.reshape(tf.reduce_mean(vect, axis=0), tf.shape(weight))
    weight.assign(weight - vect)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    ihvp_calculator = ExactIHVP(influence_model, train_dataset)
    first_order = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)
    ihvp_expected = first_order._compute_influence_vector((inputs_train, targets_train))

    assert tf.reduce_max(tf.abs((ihvp_computed - ihvp_expected) / ihvp_expected)) < 1E-2


def test_preprocess_sample_to_evaluate():
    tf.random.set_seed(0)

    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    binary = True
    if binary:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64, activation='sigmoid')])
        loss_function = BinaryCrossentropy(reduction=Reduction.NONE)
    else:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64)])
        loss_function = MeanSquaredError(reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    inputs_test = tf.random.normal((60, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.random.normal((60, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    pre_evaluate_computed = rps_lje._preprocess_samples((inputs_test, targets_test))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    ihvp_calculator = ExactIHVP(influence_model, train_dataset)
    first_order = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)

    vect = first_order._compute_influence_vector((inputs_train, targets_train))
    weight = model.layers[-1].weights[0]
    vect = tf.reshape(tf.reduce_mean(vect, axis=0), tf.shape(weight))
    weight.assign(weight - vect)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    ihvp_calculator = ExactIHVP(influence_model, train_dataset)
    first_order = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)
    pre_evaluate_expected = first_order._preprocess_samples((inputs_test, targets_test))

    assert tf.reduce_max(tf.abs(pre_evaluate_computed - pre_evaluate_expected)) < 1E-3


def test_compute_influence_value_from_influence_vector():
    tf.random.set_seed(0)

    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    binary = True
    if binary:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64, activation='sigmoid')])
        loss_function = BinaryCrossentropy(reduction=Reduction.NONE)
    else:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64)])
        loss_function = MeanSquaredError(reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    inputs_test = tf.random.normal((60, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.random.normal((60, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    v_test = rps_lje._preprocess_samples((inputs_test, targets_test))
    influence_vector = rps_lje._compute_influence_vector((inputs_test, targets_test))
    influence_values_computed = rps_lje._estimate_influence_value_from_influence_vector(v_test, influence_vector)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    ihvp_calculator = ExactIHVP(influence_model, train_dataset)
    first_order = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)

    vect = first_order._compute_influence_vector((inputs_train, targets_train))
    weight = model.layers[-1].weights[0]
    vect = tf.reshape(tf.reduce_mean(vect, axis=0), tf.shape(weight))
    weight.assign(weight - vect)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    ihvp_calculator = ExactIHVP(influence_model, train_dataset)
    first_order = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)
    v_test = first_order._preprocess_samples((inputs_test, targets_test))
    influence_vector = first_order._compute_influence_vector((inputs_test, targets_test))
    influence_values_expected = first_order._estimate_influence_value_from_influence_vector(v_test, influence_vector)

    assert tf.reduce_max(
        tf.abs((influence_values_computed - influence_values_expected) / influence_values_expected)) < 1E-2


def test_compute_pairwise_influence_value():
    tf.random.set_seed(0)

    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    binary = True
    if binary:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64, activation='sigmoid')])
        loss_function = BinaryCrossentropy(reduction=Reduction.NONE)
    else:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64)])
        loss_function = MeanSquaredError(reduction=Reduction.NONE)

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)

    inputs_test = tf.random.normal((60, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.random.normal((60, 1), dtype=tf.float64)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_dataset, ExactIHVPFactory(), target_layer=-1)
    influence_values_computed = rps_lje._compute_influence_value_from_batch((inputs_test, targets_test))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    ihvp_calculator = ExactIHVP(influence_model, train_dataset)
    first_order = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)

    vect = first_order._compute_influence_vector((inputs_train, targets_train))
    weight = model.layers[-1].weights[0]
    vect = tf.reshape(tf.reduce_mean(vect, axis=0), tf.shape(weight))
    weight.assign(weight - vect)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    ihvp_calculator = ExactIHVP(influence_model, train_dataset)
    first_order = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)
    influence_values_expected = first_order._compute_influence_value_from_batch((inputs_test, targets_test))

    assert tf.reduce_max(
        tf.abs((influence_values_computed - influence_values_expected) / influence_values_expected)) < 1E-2


def test_inheritance():
    tf.random.set_seed(0)

    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    binary = True
    if binary:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64, activation='sigmoid')])
        loss_function = BinaryCrossentropy(reduction=Reduction.NONE)
    else:
        model = Sequential(
            [model_feature, Dense(1, use_bias=False, dtype=tf.float64)])
        loss_function = MeanSquaredError(reduction=Reduction.NONE)

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
