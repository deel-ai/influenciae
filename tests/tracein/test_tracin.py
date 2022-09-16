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

from ..utils import almost_equal, assert_inheritance

def test_compute_influence_vector():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((10, 1), dtype=tf.float64)
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    f_train = model_feature(inputs_train)
    g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train

    expected_inf_vect = tf.concat([g_train * tf.cast(tf.sqrt(lr), tf.float64), g_train * tf.cast(tf.sqrt(2*lr), tf.float64)], axis=1)

    inf_vect = []
    for batch in train_set:
        batched_inf_vec = tracin.compute_influence_vector(batch)
        assert batched_inf_vec.shape == (5, 2*if_model.nb_params) # (batch_size, nb_model * nb_params)
        #TODO: What should be the shape of that (nb_model*batch_size, nb_params) or (batch_size, nb_model*nb_params)
        inf_vect.append(batched_inf_vec)
    inf_vect = tf.concat(inf_vect, axis=0)
    assert almost_equal(expected_inf_vect, inf_vect)

def test_preprocess_sample_to_evaluate():
    """Not needed as it is simply the compute_influence_vector function tested already"""
    pass

def test_compute_influence_value_from_influence_vector():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((10, 1), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(10)

    expected_values = []

    f_train = model_feature(inputs_train)
    g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train

    inf_vect = tf.concat([g_train * tf.cast(tf.sqrt(lr), tf.float64), g_train * tf.cast(tf.sqrt(2*lr), tf.float64)], axis=1)

    for inputs_test, targets_test in test_set:
        
        f_test = model_feature(inputs_test)
        g_test = 2 * (tf.reduce_sum(f_test, axis=1, keepdims=True) - targets_test) * f_test
        test_inf_vect = tf.concat([g_test * tf.cast(tf.sqrt(lr), tf.float64), g_test * tf.cast(tf.sqrt(2*lr), tf.float64)], axis=1)
        v = tf.matmul(test_inf_vect, tf.transpose(inf_vect))

        expected_values.append(v)

    expected_values = tf.concat(expected_values, axis=0)

    inf_vect = []
    for train_batch in train_set:
        batch_inf_vec = tracin.compute_influence_vector(train_batch)
        inf_vect.append(batch_inf_vec)
    inf_vect = tf.concat(inf_vect, axis=0)

    computed_values = []
    for test_batch in test_set:
        preproc_test_batch = tracin.preprocess_sample_to_evaluate(test_batch)
        inf_values = tracin.compute_influence_value_from_influence_vector(preproc_test_batch, inf_vect)
        computed_values.append(inf_values)
    computed_values = tf.concat(computed_values, axis=0)
    assert computed_values.shape == (50, 10)
    assert tf.reduce_max(tf.abs(computed_values - expected_values)) < 1E-6

def test_compute_pairwise_influence_value():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((10, 1), dtype=tf.float64)
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)

    f_train = model_feature(inputs_train)
    g_train = 2 * (tf.reduce_sum(f_train, axis=1, keepdims=True) - targets_train) * f_train

    expected_inf_vect = tf.concat([g_train * tf.cast(tf.sqrt(lr), tf.float64), g_train * tf.cast(tf.sqrt(2*lr), tf.float64)], axis=1)
    expected_pairwise_inf_vect = tf.reduce_sum(expected_inf_vect*expected_inf_vect, axis=1, keepdims=True)

    pairwise_inf = []
    for batch in train_set:
        loc_pairwise_inf = tracin.compute_pairwise_influence_value(batch)
        assert loc_pairwise_inf.shape == (5, 1)
        pairwise_inf.append(loc_pairwise_inf)
    pairwise_inf = tf.concat(pairwise_inf, axis=0)
    assert pairwise_inf.shape == (10, 1)
    assert almost_equal(expected_pairwise_inf_vect, pairwise_inf)

def test_inheritance():
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))

    model = Sequential(
        [model_feature, Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64)])

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])


    inputs_train = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((10, 1), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(10)

    method = tracin
    nb_params = sum([ifmodel.nb_params for ifmodel in tracin.models])

    assert_inheritance(
        method,
        nb_params,
        train_set,
        test_set
    )
