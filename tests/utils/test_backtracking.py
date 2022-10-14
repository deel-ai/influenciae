# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError

from deel.influenciae.utils.backtracking_line_search import BacktrackingLineSearch
from ..utils_test import almost_equal


def test_backtracking_line_search():
    # Define a simple least squares problem y = m * x + b + e, e dist normal(0, 1)
    # This is a convex problem, so the optimizer should converge to the global optimum quite easily
    t = tf.linspace(0., 10., 100)
    m = 0.42
    b = 0.66
    y = m * t + b + tf.random.normal((100,), stddev=0.1)

    # Define the linear problem as Ax=b
    inputs = Input(shape=(1,))
    x = Dense(1, use_bias=True)(inputs)
    model = Model(inputs=inputs, outputs=x)
    optimizer = BacktrackingLineSearch(batches_per_epoch=10, scaling_factor=0.1)
    model.compile(optimizer=optimizer, loss=MeanSquaredError())

    # Optimize using backtracking line-search SGD
    epochs = 100
    train_set = tf.data.Dataset.from_tensor_slices((t, y)).shuffle(100)
    loss_fn = MeanSquaredError()
    for e in range(epochs):
        for t_batch, y_batch in train_set.batch(10):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_weights)
                y_pred = model(t_batch)
                loss = loss_fn(y_batch, y_pred)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.step(model, loss, t_batch, y_batch, grads)

    # Get the estimated m and b
    m_estimated = model.weights[0]
    b_estimated = model.weights[1]
    assert almost_equal(m_estimated, tf.cast(m, tf.float32), epsilon=1e-2)
    assert almost_equal(b_estimated, tf.cast(b, tf.float32), epsilon=1e-2)
