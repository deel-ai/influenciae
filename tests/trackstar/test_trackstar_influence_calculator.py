# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import pytest
import tensorflow as tf

from deel.influenciae.common import BaseInfluenceModel
from deel.influenciae.trackstar import (
    FixedMix,
    ProjectionBlock,
    ProjectionTerm,
    TrackStarBuilder,
    TrackStarProjectionPlan,
)


pytestmark = pytest.mark.tensorflow


def _projection_plan(wrapper):
    blocks = []
    for index, shape in enumerate(wrapper.parameter_layout.parameter_shapes):
        output_axis = 1 if len(shape) > 1 else 0
        rows = shape[output_axis] if shape else 1
        columns = int(np.prod(shape)) // rows if shape else 1
        term = ProjectionTerm(
            f"parameter_{index}",
            (index,),
            (shape,),
            (1, 1),
            output_axis=output_axis,
            left_matrix=np.ones((1, rows)),
            right_matrix=np.ones((columns, 1)),
        )
        blocks.append(ProjectionBlock(f"block_{index}", (term,)))
    return TrackStarProjectionPlan.from_layout(wrapper.parameter_layout, blocks)


def test_tensorflow_builder_and_query_pipeline():
    tf.random.set_seed(4)
    model = tf.keras.Sequential(
        [
            tf.keras.Input((2,), dtype=tf.float64),
            tf.keras.layers.Dense(3, use_bias=False, dtype=tf.float64),
            tf.keras.layers.Dense(1, use_bias=False, dtype=tf.float64),
        ]
    )
    wrapper = BaseInfluenceModel(
        model,
        loss_function=tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        ),
    )
    optimizer = tf.keras.optimizers.Adam(0.01)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(model(tf.ones((2, 2), dtype=tf.float64)) ** 2)
    optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_weights), model.trainable_weights))
    train_x = tf.reshape(tf.range(12, dtype=tf.float64), (6, 2)) / 10
    train = tf.data.Dataset.from_tensor_slices((train_x, tf.zeros((6, 1), tf.float64))).batch(2)
    evaluation = tf.data.Dataset.from_tensor_slices(
        (tf.ones((3, 2), tf.float64), tf.zeros((3, 1), tf.float64))
    ).batch(2)
    builder = TrackStarBuilder.from_optimizer(
        wrapper,
        optimizer,
        _projection_plan(wrapper),
        optimizer_epsilon=1e-8,
        mix=FixedMix(0.25),
    )

    calculator = builder.fit(train, evaluation).build()
    query_batch = next(iter(evaluation))
    result = calculator.search_batch(query_batch, 2)

    assert result.scores.shape == (2, 2)
    assert result.ids.shape == (2, 2)
    assert result.payload.shape == (2, 2, 2)
    np.testing.assert_allclose(calculator.self_influence_batch(query_batch), 1.0)
    top_k_batches = list(calculator.top_k(evaluation, 1, return_payload=False))
    np.testing.assert_array_equal(top_k_batches[0].query_ids, [0, 1])
    np.testing.assert_array_equal(top_k_batches[1].query_ids, [2])
    assert top_k_batches[0].neighbors.payload is None
