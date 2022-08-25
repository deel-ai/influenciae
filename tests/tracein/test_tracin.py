# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, MeanSquaredError

from deel.influenciae.common import InfluenceModel
from deel.influenciae.tracein.tracin import TracIn


def test_computation_train_tensor_test_dataset():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    lr = 3.0
    if_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((10, 1), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(10)
    computed_values = tracin.compute_influence_values((inputs_train, targets_train), test_set)

    expected_values = []
    for inputs_test, targets_test in test_set:
        f_train = model_feature(inputs_train)
        f_test = model_feature(inputs_test)

        g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train
        g_test = 2 * (tf.reduce_sum(f_test, axis=1, keepdims=True) - targets_test) * f_test

        sum_lr = 3 * lr
        v = tf.reduce_sum(g_train * g_test, axis=1, keepdims=True) * sum_lr
        expected_values.append(v)

    expected_values = tf.concat(expected_values, axis=0)
    assert tf.reduce_max(tf.abs(computed_values - expected_values)) < 1E-6


def test_computation_train_dataset_test_tensor():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    lr = 3.0
    if_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)
    targets_test = tf.random.normal((10, 1), dtype=tf.float64)

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(10)
    computed_values = tracin.compute_influence_values(train_set, (inputs_test, targets_test))

    expected_values = []
    for inputs_train, targets_train in train_set:
        f_train = model_feature(inputs_train)
        f_test = model_feature(inputs_test)

        g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train
        g_test = 2 * (tf.reduce_sum(f_test, axis=1, keepdims=True) - targets_test) * f_test

        sum_lr = 3 * lr
        v = tf.reduce_sum(g_train * g_test, axis=1, keepdims=True) * sum_lr
        expected_values.append(v)

    expected_values = tf.concat(expected_values, axis=0)
    assert tf.reduce_max(tf.abs(computed_values - expected_values)) < 1E-6


def test_computation_train_dataset_test_dataset():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    lr = 3.0
    if_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))
    computed_values = tracin.compute_influence_values(train_set.batch(10), test_set.batch(10))

    f_train = model_feature(inputs_train)
    f_test = model_feature(inputs_test)

    g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train
    g_test = 2 * (tf.reduce_sum(f_test, axis=1, keepdims=True) - targets_test) * f_test

    sum_lr = 3 * lr
    expected_values = tf.reduce_sum(g_train * g_test * sum_lr, axis=1, keepdims=True)

    assert tf.reduce_max(tf.abs(computed_values - expected_values)) < 1E-6


def test_computation_two_models_differents():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model1 = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model1(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))
    if_model1 = InfluenceModel(model1, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    model2 = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.zeros_initializer, use_bias=False, dtype=tf.float64)])

    model2(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))
    if_model2 = InfluenceModel(model2, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    lrs = [3.0, 5.0]
    tracin = TracIn([if_model1, if_model2], lrs)

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    computed_values = tracin.compute_influence_values((inputs_train, targets_train), (inputs_test, targets_test))

    f_train = model_feature(inputs_train)
    f_test = model_feature(inputs_test)

    g_train1 = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train
    g_test1 = 2 * (tf.reduce_sum(f_test, axis=1, keepdims=True) - targets_test) * f_test

    expected_values1 = tf.reduce_sum(g_train1 * g_test1 * lrs[0], axis=1, keepdims=True)

    g_train2 = 2 * (- targets_train) * f_train
    g_test2 = 2 * (- targets_test) * f_test

    expected_values2 = tf.reduce_sum(g_train2 * g_test2 * lrs[1], axis=1, keepdims=True)

    expected_values = expected_values2 + expected_values1

    assert tf.reduce_max(tf.abs(computed_values - expected_values)) < 1E-6


def test_computation_same_model():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    lr = 3.0
    if_model = InfluenceModel(model, target_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((50, 1), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    computed_values = tracin.compute_influence_values((inputs_train, targets_train), (inputs_test, targets_test))

    f_train = model_feature(inputs_train)
    f_test = model_feature(inputs_test)

    g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train
    g_test = 2 * (tf.reduce_sum(f_test, axis=1, keepdims=True) - targets_test) * f_test

    sum_lr = 3 * lr
    expected_values = tf.reduce_sum(g_train * g_test * sum_lr, axis=1, keepdims=True)

    assert tf.reduce_max(tf.abs(computed_values - expected_values)) < 1E-6


def test_cnn_shapes():
    models = []
    for _ in range(3):
        model = Sequential()
        model.add(Input(shape=(5, 5, 3)))
        model.add(Conv2D(4, kernel_size=(2, 2),
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(10))
        model.add(Dense(10))
        model.compile(loss=CategoricalCrossentropy(from_logits=False, reduction=Reduction.NONE), optimizer='sgd')
        influence_model = InfluenceModel(model)
        models.append(influence_model)

    inputs_train = tf.random.normal((50, 5, 5, 3))
    inputs_test = tf.random.normal((50, 5, 5, 3))
    targets_train = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    targets_test = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # Compute the influence vector using auto-diff and check shapes
    tracin = TracIn(models, 2.0)
    influence = tracin.compute_influence_values(train_set.batch(4), test_set.batch(4))
    assert influence.shape == (50, 1)  # 50 times a scalar (1, 1)
