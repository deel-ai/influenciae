# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import pytest
import tensorflow as tf

from deel.influenciae.common import BaseInfluenceModel
from deel.influenciae.trackstar.optimizer_state import extract_optimizer_second_moments


pytestmark = pytest.mark.tensorflow


def _model_and_wrapper():
    model = tf.keras.Sequential(
        [tf.keras.Input((2,)), tf.keras.layers.Dense(2), tf.keras.layers.Dense(1)]
    )
    wrapper = BaseInfluenceModel(
        model,
        loss_function=tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        ),
    )
    return model, wrapper


@pytest.mark.parametrize(
    "optimizer_type",
    tuple(
        optimizer_type
        for optimizer_type in (
            tf.keras.optimizers.Adam,
            getattr(tf.keras.optimizers, "AdamW", None),
        )
        if optimizer_type is not None
    ),
)
def test_extracts_tensorflow_raw_second_moments(optimizer_type):
    model, wrapper = _model_and_wrapper()
    optimizer = optimizer_type(learning_rate=0.01, beta_2=0.5)
    gradients = [tf.fill(weight.shape, float(index)) for index, weight in enumerate(model.trainable_weights, 1)]
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    snapshot = extract_optimizer_second_moments(wrapper, optimizer)

    for index, (moment, shape) in enumerate(
        zip(snapshot.parameter_second_moments, wrapper.parameter_layout.parameter_shapes), 1
    ):
        np.testing.assert_allclose(moment, np.full(shape, 0.5 * index ** 2), rtol=1e-6)
        assert not moment.flags.writeable


def test_uninitialized_and_unsupported_tensorflow_optimizer_state_fails():
    model, wrapper = _model_and_wrapper()

    with pytest.raises(RuntimeError, match="not initialized"):
        extract_optimizer_second_moments(wrapper, tf.keras.optimizers.Adam())
    with pytest.raises(TypeError, match="Adam or AdamW"):
        extract_optimizer_second_moments(wrapper, tf.keras.optimizers.SGD())

    assert model.trainable_weights
