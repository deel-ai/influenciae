# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Reduction, MeanSquaredError, BinaryCrossentropy

from deel.influenciae.influence import ArnoldiInfluenceCalculator, FirstOrderInfluenceCalculator
from deel.influenciae.common import InfluenceModel
from tests.utils_test import assert_inheritance


def test_inverse_exact_hessian():
    tf.random.set_seed(0)

    dtype = tf.float64

    model = Sequential()
    model.add(Input(shape=(10,), dtype=dtype))
    model.add(Dense(1, dtype=dtype, use_bias=False))

    loss_function = lambda y1, y2, _=None: (y1 - y2) ** 2
    nb_sample = 100
    batch_size = 10
    model(tf.random.normal((nb_sample, 10), dtype=dtype))

    inputs_train = tf.random.normal((nb_sample, 10), dtype=dtype)
    targets_train = tf.random.normal((nb_sample, 1), dtype=dtype)

    hessian = 2 * tf.matmul(tf.expand_dims(inputs_train, axis=2), tf.expand_dims(inputs_train, axis=1))
    hessian = tf.reduce_mean(hessian, axis=0)

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(batch_size)

    k_largest_eig_vals = 10
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    scaling_up = ArnoldiInfluenceCalculator(influence_model, train_dataset, 10,
                                            False, k_largest_eig_vals, dtype=dtype)

    H_inv = tf.matmul(scaling_up.G, tf.matmul(tf.linalg.diag(1.0 / scaling_up.eig_vals), scaling_up.G),
                      transpose_a=True)
    assert tf.reduce_max(tf.abs(tf.cast(H_inv, dtype=tf.float64) - tf.linalg.inv(hessian))).numpy() < 1E-6


def test_exact_influence_values():
    tf.random.set_seed(0)

    dtype = tf.float64

    model = Sequential()
    model.add(Input(shape=(10,), dtype=dtype))
    model.add(Dense(1, dtype=dtype, use_bias=False))

    loss_function = lambda y1, y2, _=None: (y1 - y2) ** 2
    nb_sample = 100
    batch_size = 10
    model(tf.random.normal((nb_sample, 10), dtype=dtype))

    inputs_train = tf.random.normal((nb_sample, 10), dtype=dtype)
    targets_train = tf.random.normal((nb_sample, 1), dtype=dtype)
    train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(batch_size)

    k_largest_eig_vals = 10
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    arnoldi_calculator = ArnoldiInfluenceCalculator(influence_model, train_dataset, 10,
                                                    False, k_largest_eig_vals, dtype=dtype)

    scaling_up_influence_values = arnoldi_calculator._compute_influence_values(train_dataset)

    first_order_influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_dataset)
    first_order_influence_values = first_order_influence_calculator._compute_influence_values(train_dataset)

    assert tf.reduce_max(tf.abs(scaling_up_influence_values - first_order_influence_values)) < 1E-6

    inputs_test = tf.random.normal((nb_sample, 10), dtype=dtype)
    targets_test = tf.random.normal((nb_sample, 1), dtype=dtype)
    test_dataset = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(7)

    scaling_up_influence_values = arnoldi_calculator.estimate_influence_values_in_batches(test_dataset, train_dataset)
    first_order_influence_values = first_order_influence_calculator.estimate_influence_values_in_batches(test_dataset,
                                                                                                         train_dataset)

    for v1, v2 in zip(scaling_up_influence_values, first_order_influence_values):
        for v1_, v2_ in zip(v1[1], v2[1]):
            assert tf.reduce_max(tf.abs(v1_[1] - v2_[1])) < 1E-6


def test_inheritance():
    for hermissian in [False,True]:
        tf.random.set_seed(0)

        dtype = tf.float64

        model_feature = Sequential()
        model_feature.add(Input(shape=(5, 5, 3), dtype=dtype))
        model_feature.add(Conv2D(4, kernel_size=(2, 2),
                                 activation='relu', dtype=dtype))
        model_feature.add(Flatten(dtype=dtype))

        binary = True
        if binary:
            model = Sequential(
                [model_feature, Dense(1, use_bias=False, dtype=dtype, activation='sigmoid')])
            loss_function = BinaryCrossentropy(reduction=Reduction.NONE)
        else:
            model = Sequential(
                [model_feature, Dense(1, use_bias=False, dtype=dtype)])
            loss_function = MeanSquaredError(reduction=Reduction.NONE)

        model(tf.random.normal((10, 5, 5, 3), dtype=dtype))

        inputs_train = tf.random.normal((10, 5, 5, 3), dtype=dtype)
        targets_train = tf.random.normal((10, 1), dtype=dtype)

        inputs_test = tf.random.normal((50, 5, 5, 3), dtype=dtype)
        targets_test = tf.random.normal((50, 1), dtype=dtype)

        train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)
        test_dataset = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(10)

        k_largest_eig_vals = 7
        influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
        scaling_up = ArnoldiInfluenceCalculator(influence_model, train_dataset, 12,
                                                hermissian, k_largest_eig_vals, dtype=dtype)

        assert_inheritance(
            scaling_up,
            k_largest_eig_vals,
            train_dataset,
            test_dataset
        )
