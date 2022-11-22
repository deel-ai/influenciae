# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf

from deel.influenciae.rps.rps_l2 import RepresenterPointL2
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tests.utils_test import assert_inheritance


def test_surrogate_model():
    x_train = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_train = tf.keras.utils.to_categorical(tf.random.categorical(tf.math.log([[0.25, 0.25, 0.25, 0.25]]), 100))
    train_set = tf.data.Dataset.from_tensor_slices((x_train, tf.cast(tf.squeeze(y_train), tf.float32)))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, 4, "same", activation='swish'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(train_set.shuffle(100).batch(32), epochs=40, verbose=0)
    rps_l2 = RepresenterPointL2(model, train_set.batch(32), lambda_regularization=0.1)

    # Check the shapes of the surrogate model
    surrogate_model = rps_l2._create_surrogate_model()
    assert surrogate_model.input_shape == model.layers[-1].input_shape
    assert surrogate_model.output_shape == model.output_shape

    # Train and check that it has learned to predict like the original model
    rps_l2._train_last_layer(100)
    surrogate_model = rps_l2.linear_layer
    original_preds = model(x_train)
    surrogate_preds = surrogate_model(rps_l2.feature_extractor(x_train))
    mean_bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(original_preds, surrogate_preds))
    assert mean_bce < 0.1


def test_gradients():
    x_train = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_train = tf.random.categorical(tf.math.log([[0.25, 0.25, 0.25, 0.25]]), 100)
    y_train = tf.one_hot(y_train, depth=4)
    y_train = tf.cast(tf.squeeze(y_train, axis=0), dtype=tf.float32)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, 4, "valid", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE),
        metrics=['accuracy']
    )
    model.fit(train_set.shuffle(100).batch(32), epochs=40, verbose=0)
    lambda_regularization = 0.01
    rps_l2 = RepresenterPointL2(model, train_set.batch(32), lambda_regularization=lambda_regularization)
    rps_l2._train_last_layer(100)
    surrogate_model = rps_l2.linear_layer

    # Compute gradients symbolically for the cross-entropy and compare to AD
    feature_maps = rps_l2.feature_extractor(x_train)
    preds = surrogate_model(feature_maps)
    # y * log ( logits) + (1-y) * log(1 + logits)
    #

    ground_truth_gradients = tf.matmul(tf.expand_dims(feature_maps, axis=-1),
                                       tf.reshape(tf.nn.softmax(preds, axis=1) - y_train, (y_train.shape[0], 1, -1)))
    print(tf.shape(ground_truth_gradients))
    ground_truth_influence = tf.divide(ground_truth_gradients,
                                       -2. * lambda_regularization * tf.cast(y_train.shape[0], tf.float32))
    print(tf.shape(ground_truth_influence))
    ground_truth_influence = tf.reduce_sum(
        tf.multiply(
            ground_truth_influence,
            tf.repeat(tf.expand_dims(tf.divide(tf.ones_like(feature_maps), feature_maps), axis=-1),
                      ground_truth_influence.shape[-1], axis=-1)
            ),
        axis=1
    )
    print(tf.shape(ground_truth_influence))

    gradients = []
    for x, y in train_set.batch(32):
        z = rps_l2.feature_extractor(x)
        g = rps_l2._compute_alpha(z, y)
        gradients.append(g)
    gradients = tf.concat(gradients, axis=0)

    ###
    print(f'grads = {gradients}')
    print(f'gt grads = {ground_truth_influence}')
    ###
    assert gradients.shape == (100, 4)
    assert tf.reduce_max(tf.abs(ground_truth_influence - gradients)) < 1e-4


def test_influence_values():
    # Test for binary classification first
    x_train = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_train = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 100)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, tf.cast(tf.squeeze(y_train), tf.float32)))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, 5, "valid", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE),
        metrics=['accuracy']
    )
    model.fit(train_set.shuffle(100).batch(32), epochs=40, verbose=0)
    lambda_regularization = 0.1
    rps_l2 = RepresenterPointL2(model, train_set.batch(20), lambda_regularization=lambda_regularization)
    rps_l2._train_last_layer(100)
    surrogate_model = rps_l2.linear_layer

    # Compute influence values symbolically
    feature_maps = rps_l2.feature_extractor(x_train)
    preds = surrogate_model(feature_maps)
    ground_truth_gradients = tf.matmul(feature_maps, tf.nn.sigmoid(preds) - tf.cast(y_train, tf.float32),
                                       transpose_a=True)
    ground_truth_influence = tf.divide(ground_truth_gradients,
                                       -2. * lambda_regularization * tf.cast(y_train.shape[0], tf.float32))
    ground_truth_influence = tf.reduce_sum(
        tf.multiply(
            tf.transpose(ground_truth_influence),
            tf.divide(tf.ones_like(feature_maps), feature_maps)
        ),
        axis=1
    )

    ###
    print(f'gradients shape = {ground_truth_gradients.shape}')
    print(f'influence shape = {ground_truth_influence.shape}')
    ###

    # Compare to the values from AD
    influence = rps_l2._compute_influence_values(train_set.batch(20))
    ###
    print(f'influence final = {influence.shape}')
    ###

    assert tf.reduce_max(tf.abs(ground_truth_influence - tf.squeeze(influence))) < 1e-3

    # Now test for multi-class classification
    y_train = tf.random.categorical(tf.math.log([[0.25, 0.25, 0.25, 0.25]]), 100)
    y_train = tf.one_hot(y_train, depth=4)
    y_train = tf.cast(tf.squeeze(y_train, axis=0), dtype=tf.float32)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, 4, "valid", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE),
        metrics=['accuracy']
    )
    model.fit(train_set.shuffle(100).batch(32), epochs=40, verbose=0)
    lambda_regularization = 0.1
    rps_l2 = RepresenterPointL2(model, train_set.batch(20), lambda_regularization=lambda_regularization)
    rps_l2._train_last_layer(100)
    surrogate_model = rps_l2.linear_layer

    # Compute influence values symbolically
    pred_indices = tf.argmax(model(x_train), axis=1)
    feature_maps = rps_l2.feature_extractor(x_train)
    preds = surrogate_model(feature_maps)
    ground_truth_gradients = tf.matmul(tf.expand_dims(feature_maps, axis=-1),
                                       tf.reshape(tf.expand_dims(tf.nn.softmax(preds, axis=1), axis=1) - tf.expand_dims(
                                           y_train, axis=-1),
                                                  (y_train.shape[0], 1, -1)))
    gradient_indices = [0, 5, 10, 15]
    ground_truth_gradients = tf.gather(ground_truth_gradients, gradient_indices, axis=-1)
    ground_truth_influence = tf.divide(ground_truth_gradients,
                                       -2. * lambda_regularization * tf.cast(y_train.shape[0], tf.float32))
    ground_truth_influence = tf.reduce_sum(
        tf.multiply(
            ground_truth_influence,
            tf.repeat(tf.expand_dims(tf.divide(tf.ones_like(feature_maps), feature_maps), axis=-1),
                      ground_truth_influence.shape[-1], axis=-1)
        ),
        axis=1
    )
    ground_truth_influence = tf.gather(ground_truth_influence, pred_indices, axis=1, batch_dims=1)

    # Compare to the values from AD
    influence = rps_l2._compute_influence_values(train_set.batch(20))

    assert tf.reduce_max(tf.abs(tf.transpose(ground_truth_influence) - influence)) < 1e-3


def test_predict_with_kernel():
    x_train = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_train = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 100)
    x_test = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_test = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 100)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, tf.cast(tf.squeeze(y_train), tf.float32)))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, tf.cast(tf.squeeze(y_test), tf.float32)))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, 5, "valid", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE),
        metrics=['accuracy']
    )
    model.fit(train_set.shuffle(100).batch(32), epochs=40, validation_data=test_set.batch(32), verbose=0)
    lambda_regularization = 10.
    rps_l2 = RepresenterPointL2(model, train_set.batch(20), lambda_regularization=lambda_regularization)
    kernel_preds = []
    for x_ in train_set.batch(20):
        kernel_preds.append(rps_l2.predict_with_kernel(x_))
    kernel_preds = tf.concat(kernel_preds, axis=0)
    # kernel_preds = rps_l2.predict_with_kernel(train_set.map(lambda x, y: x).batch(20))
    model_preds = model.predict(train_set.map(lambda x, y: x).batch(20))
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    assert bce_loss(tf.squeeze(model_preds), kernel_preds) / 100. < 0.1


def test_inheritance():
    model = Sequential()
    model.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', dtype=tf.float64))
    model.add(Flatten(dtype=tf.float64))

    model.add(Dense(1, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE),
        metrics=['accuracy']
    )

    model(tf.random.normal((50, 5, 5, 3), dtype=tf.float64))

    inputs_train = tf.random.normal((10, 5, 5, 3), dtype=tf.float64)
    inputs_test = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.random.normal((10, 1), dtype=tf.float64)
    targets_test = tf.random.normal((50, 1), dtype=tf.float64)

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(10)

    lambda_regularization = 10.
    method = RepresenterPointL2(model, train_set, lambda_regularization=lambda_regularization)

    nb_params = tf.reduce_sum([tf.size(w) for w in model.layers[-1].weights])

    assert_inheritance(
        method,
        nb_params,
        train_set,
        test_set
    )
