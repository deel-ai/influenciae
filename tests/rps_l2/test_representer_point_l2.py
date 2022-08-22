# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf

from deel.influenciae.rps_l2 import RepresenterPointL2
from ..utils import almost_equal


def test_surrogate_model():
    x_train = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_train = tf.keras.utils.to_categorical(tf.random.categorical(tf.math.log([[0.25, 0.25, 0.25, 0.25]]), 100))
    x_test = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_test = tf.keras.utils.to_categorical(tf.random.categorical(tf.math.log([[0.25, 0.25, 0.25, 0.25]]), 100))
    train_set = tf.data.Dataset.from_tensor_slices((x_train, tf.cast(tf.squeeze(y_train), tf.float32)))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, tf.cast(tf.squeeze(y_test), tf.float32)))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, 2, "same", activation='swish'),
        tf.keras.layers.Conv2D(32, 3, 2, "same", activation='swish'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(train_set.shuffle(100).batch(32), epochs=40, validation_data=test_set.batch(32), verbose=0)
    rps_l2 = RepresenterPointL2(model, train_set.batch(32), lambda_regularization=0.1)

    # Check the shapes of the surrogate model
    surrogate_model = rps_l2._create_surrogate_model()
    assert surrogate_model.input_shape == model.layers[-1].input_shape
    assert surrogate_model.output_shape == model.output_shape

    # Train and check that it has learned to predict like the original model
    rps_l2._train_last_layer()
    surrogate_model = rps_l2.linear_layer
    original_preds = model(x_train)
    surrogate_preds = surrogate_model(rps_l2.feature_extractor(x_train))
    mean_mse = tf.reduce_mean(tf.keras.losses.mse(original_preds, surrogate_preds))
    assert mean_mse < 1e-1


def test_gradients():
    x_train = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_train = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 100)
    x_test = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_test = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 100)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, tf.cast(tf.squeeze(y_train), tf.float32)))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, tf.cast(tf.squeeze(y_test), tf.float32)))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, 2, "same", activation='relu'),
        tf.keras.layers.Conv2D(32, 3, 2, "same", activation='relu'),
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
    rps_l2 = RepresenterPointL2(model, train_set.batch(32), lambda_regularization=0.1)
    rps_l2._train_last_layer()
    surrogate_model = rps_l2.linear_layer

    # Compute gradients symbolically for the cross-entropy and compare to AD
    preds = surrogate_model(rps_l2.feature_extractor(x_train))
    ground_truth_gradients = tf.transpose(tf.nn.sigmoid(preds)) - tf.cast(y_train, tf.float32)
    gradients = tf.concat([g for _, _, g in rps_l2._compute_gradients(train_set.batch(16))], axis=0)
    assert gradients.shape == (100,)
    assert almost_equal(ground_truth_gradients, gradients, epsilon=1e-4)


def test_influence_values():
    x_train = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_train = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 100)
    x_test = tf.random.normal((100, 32, 32, 3), dtype=tf.float32)
    y_test = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 100)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, tf.cast(tf.squeeze(y_train), tf.float32)))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, tf.cast(tf.squeeze(y_test), tf.float32)))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, 2, "same", activation='swish'),
        tf.keras.layers.Conv2D(32, 3, 2, "same", activation='swish'),
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
    lambda_regularization = 0.1
    rps_l2 = RepresenterPointL2(model, train_set.batch(20), lambda_regularization=lambda_regularization)
    rps_l2._train_last_layer()
    surrogate_model = rps_l2.linear_layer

    # Compute influence values symbolically
    preds = surrogate_model(rps_l2.feature_extractor(x_train))
    ground_truth_gradients = tf.transpose(tf.nn.sigmoid(preds)) - tf.cast(y_train, tf.float32)
    ground_truth_influence = tf.divide(ground_truth_gradients,
                                       -2. * lambda_regularization * tf.cast(y_train.shape[1], tf.float32))

    # Compare to the values from AD
    influence = rps_l2.compute_influence_values(train_set.batch(20))
    assert almost_equal(ground_truth_influence, influence, epsilon=1e-3)


def test_predict_with_kernel():
    pass
