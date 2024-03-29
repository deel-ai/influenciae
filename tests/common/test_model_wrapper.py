# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import pytest
import itertools
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import (MeanAbsoluteError, CategoricalCrossentropy, Reduction,
                                     MeanSquaredError, CosineSimilarity)

from deel.influenciae.common import BaseInfluenceModel, InfluenceModel

from ..utils_test import generate_model, assert_tensor_equal, almost_equal


def test_loss_reduction():
    # Ensure we raise a proper error when a loss with reduction is passed
    # and we can instantiate with proper loss
    model = generate_model((5, 5, 1), 10)

    # we should only accept loss without reduction
    with pytest.raises(ValueError):
        BaseInfluenceModel(model, loss_function=MeanAbsoluteError(reduction='sum'))
    with pytest.raises(ValueError):
        BaseInfluenceModel(model, loss_function=CategoricalCrossentropy(reduction='sum'))

    BaseInfluenceModel(model, loss_function=MeanAbsoluteError(reduction=Reduction.NONE))
    BaseInfluenceModel(model, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    BaseInfluenceModel(model, loss_function=CategoricalCrossentropy(reduction=Reduction.NONE))
    BaseInfluenceModel(model, loss_function=CosineSimilarity(reduction=Reduction.NONE))

def test_loss_calculation():
    # Ensure the wrapper can properly compute the loss

    # f x: x^2 + x
    dummy_model = Sequential()
    identity_layer = Dense(1, input_shape=(1,), use_bias=False,
                           kernel_initializer=tf.keras.initializers.Identity)

    dummy_model.add(identity_layer)
    dummy_model.add(Lambda(lambda x: x**2 + x))

    # MSE(f(1), 1) = MSE(2, 1) = 1
    # MSE(f(0), 1) = MSE(0, 1) = 1
    # MSE(f(2), -1) = MSE(6, -1) = 7
    x = tf.cast([1, 0, 2], tf.float32)[:, None]
    y = tf.cast([1, 1, -1], tf.float32)[:, None]

    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(3)

    mse_influence_model = BaseInfluenceModel(
        dummy_model, weights_to_watch=identity_layer.weights,
        loss_function=MeanAbsoluteError(
            reduction=Reduction.NONE)
        )
    loss_score = mse_influence_model.batch_loss(ds)

    assert almost_equal(loss_score, [1, 1, 7])

    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(3)

    cce_influence_model = BaseInfluenceModel(
        dummy_model,
        weights_to_watch = identity_layer.weights,
        loss_function = CategoricalCrossentropy(
            reduction=Reduction.NONE
        )
    )
    loss_score = cce_influence_model.batch_loss(ds)

    assert loss_score.shape == (3,)

def test_grad_calculation():
    # Ensure the wrapper can properly compute the gradients

    # l(y_pred, y) = (y - y_pred)^2
    # f(x) = (x*W0)*2
    # grad_x(l(f(x), y)) = 8x^2W0 - 4x*W0*y
    grad_func = lambda x, y: 8*x**2 - 4*x*y

    dummy_model = Sequential()
    identity_layer = Dense(1, input_shape=(1,), use_bias=False,
                           kernel_initializer=tf.keras.initializers.Identity)

    dummy_model.add(identity_layer)
    dummy_model.add(Lambda(lambda x: x*2))

    x = tf.cast([1, 0, 0], tf.float32)[:, None]
    y = tf.cast([10, 0, 0], tf.float32)[:, None]

    mse_influence_model = BaseInfluenceModel(dummy_model, loss_function=MeanSquaredError(
        reduction=Reduction.NONE))

    # using only one batch
    ds1 = tf.data.Dataset.from_tensor_slices((x, y)).batch(len(x))

    gradients1 = mse_influence_model.batch_gradient(ds1)
    real_gradients1 = tf.reduce_sum([grad_func(x, y) for x, y in ds1])

    assert almost_equal(real_gradients1, gradients1)

    # using one element per batch
    ds2 = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)

    gradients2 = mse_influence_model.batch_gradient(ds2)
    real_gradients2 = [grad_func(x, y).numpy()[0] for x, y in ds2]

    assert almost_equal(real_gradients2, gradients2)

def test_jacobian_calculation():
    # Ensure the wrapper can properly compute the jacobians

    # f(x) = (x*W0)*2
    # grad(f, x, y) = 8x^2W0 - 4x*W0*y
    grad_func = lambda x, y: 8 * x ** 2 - 4 * x * y

    dummy_model = Sequential()
    identity_layer = Dense(1, input_shape=(1,), use_bias=False,
                           kernel_initializer=tf.keras.initializers.Identity)

    dummy_model.add(identity_layer)
    dummy_model.add(Lambda(lambda x: x * 2))

    x = tf.cast([1, 0, 0], tf.float32)[:, None]
    y = tf.cast([10, 0, 0], tf.float32)[:, None]

    mse_influence_model = BaseInfluenceModel(dummy_model, loss_function=MeanSquaredError(
        reduction=Reduction.NONE))

    # multiple batchs or one batch should return the same result (one per x)
    ds1 = tf.data.Dataset.from_tensor_slices((x, y)).batch(len(x))
    ds2 = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)

    jacobian1 = mse_influence_model.batch_jacobian(ds1)
    jacobian2 = mse_influence_model.batch_jacobian(ds2)

    real_jacobian = [grad_func(x, y) for x, y in ds1]

    assert almost_equal(real_jacobian, jacobian1)
    assert almost_equal(real_jacobian, jacobian2)

def test_weights_default_targeting():
    # Ensure we target the correct theta / weights by default -- last layer with
    # weights before the logits
    model = Sequential()
    model.add(Input(shape=(5, 5, 1)))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Dense(10))
    model.add(Flatten())
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    # should skip the last flatten layer
    theta = model.layers[-2].weights
    influence_model = InfluenceModel(model)

    for w, theta_w in zip(influence_model.weights, theta):
        assert_tensor_equal(w, theta_w)

    model2 = Sequential()
    model2.add(Input(shape=(5, 5, 1)))
    model2.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model2.add(Flatten())
    model2.add(Dense(20))
    model2.add(Flatten())
    model2.add(Dense(10))
    model2.compile(loss='categorical_crossentropy', optimizer='sgd')

    # default target layer should be the Dense(20)
    theta2 = model2.layers[-3].weights
    influence_model2 = InfluenceModel(model2)

    for w, theta_w in zip(influence_model2.weights, theta2):
        assert_tensor_equal(w, theta_w)

def test_weights_targeting():
    # Ensure we target the right weights when specifying a layer

    model = Sequential()
    model.add(Input(shape=(5, 5, 1)))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', name="target"))
    model.add(Flatten())
    model.add(Dense(10))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    theta = model.get_layer('target').weights
    influence_model = InfluenceModel(model, start_layer='target')

    assert influence_model.weights==theta

def test_targeting_multiple_layers():
    """
    Assert that everything run smoothly when we have multiple layers
    to check on
    """
    layer_0 = Dense(2, kernel_initializer='ones', bias_initializer='zeros', name="d_test_0")
    layer_1 = Dense(2, use_bias=False, name="d_test_1")
    layer_2 = Dense(1, kernel_initializer='ones', bias_initializer='ones', name="d_test_2")
    model = Sequential([Input(shape=(5, 2)), layer_0, layer_1, layer_2])

    ## only start_layer is passed
    # first layer only
    influence_model = InfluenceModel(model, start_layer=0)
    assert influence_model.weights == layer_0.weights

    influence_model = InfluenceModel(model, start_layer="d_test_0")
    assert influence_model.weights == layer_0.weights

    # second layer only
    influence_model = InfluenceModel(model, start_layer=1)
    assert influence_model.weights == layer_1.weights

    influence_model = InfluenceModel(model, start_layer="d_test_1")
    assert influence_model.weights == layer_1.weights

    # last layer only
    influence_model = InfluenceModel(model, start_layer=2)
    assert influence_model.weights == layer_2.weights

    influence_model = InfluenceModel(model, start_layer="d_test_2")
    assert influence_model.weights == layer_2.weights

    with pytest.raises(ValueError):
        influence_model = InfluenceModel(model, start_layer="whatev")

    # ## only last_layer is passed

    # should have last layer & layer before logits (default start)
    influence_model = InfluenceModel(model, last_layer=2)
    theoric_weights = [layer_1.weights, layer_2.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, last_layer="d_test_2")
    theoric_weights = [layer_1.weights, layer_2.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, last_layer=-1)
    theoric_weights = [layer_1.weights, layer_2.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    # should raise an error
    with pytest.raises(AssertionError):
        influence_model = InfluenceModel(model, last_layer=0)
    with pytest.raises(AssertionError):
        influence_model = InfluenceModel(model, last_layer="d_test_0")
    with pytest.raises(AssertionError):
        influence_model = InfluenceModel(model, last_layer=-3)

    ## use of both
    influence_model = InfluenceModel(model, start_layer=0, last_layer=1)
    theoric_weights = [layer_0.weights, layer_1.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)

    influence_model = InfluenceModel(model, start_layer="d_test_0", last_layer=1)
    theoric_weights = [layer_0.weights, layer_1.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)

    influence_model = InfluenceModel(model, start_layer="d_test_0", last_layer="d_test_1")
    theoric_weights = [layer_0.weights, layer_1.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, start_layer=0, last_layer="d_test_1")
    theoric_weights = [layer_0.weights, layer_1.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1)
    theoric_weights = [layer_0.weights, layer_1.weights, layer_2.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, start_layer=0, last_layer="d_test_2")
    theoric_weights = [layer_0.weights, layer_1.weights, layer_2.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, start_layer="d_test_0", last_layer="d_test_2")
    theoric_weights = [layer_0.weights, layer_1.weights, layer_2.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-2)
    theoric_weights = [layer_0.weights, layer_1.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, start_layer=1, last_layer="d_test_2")
    theoric_weights = [layer_1.weights, layer_2.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights

    influence_model = InfluenceModel(model, start_layer="d_test_1", last_layer="d_test_2")
    theoric_weights = [layer_1.weights, layer_2.weights]
    theoric_weights = list(itertools.chain(*theoric_weights))
    assert influence_model.weights == theoric_weights
