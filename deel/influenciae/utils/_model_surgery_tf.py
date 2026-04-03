# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""TensorFlow-specific helpers for RPS model surgery."""
from typing import Any, Callable, Tuple, cast

import tensorflow as tf

from ..common.backend import BaseBackend
from ..types import DatasetLike, Model, Tensor
from .backtracking_line_search import BacktrackingLineSearch


def _split_batch_inputs_targets(samples: Tuple[Any, ...]) -> Tuple[Any, Any]:
    """Split a batched sample tuple into inputs and targets."""
    if len(samples) == 2:
        return samples[0], samples[1]

    inputs = samples[:-1]
    if isinstance(inputs, tuple) and len(inputs) == 1:
        inputs = inputs[0]
    return inputs, samples[-1]


def train_surrogate_linear_model_tensorflow(
    backend: BaseBackend,
    surrogate_model: Model,
    feature_extractor: Model,
    original_head: Model,
    train_set: DatasetLike,
    loss_function: Callable,
    scaling_factor: float,
    epochs: int,
    batches_per_epoch: int,
) -> Model:
    """Fit the surrogate linear model with TensorFlow's line-search optimizer."""
    tf_surrogate_model = cast(tf.keras.Model, surrogate_model)
    tf_feature_extractor = cast(tf.keras.Model, feature_extractor)
    tf_original_head = cast(tf.keras.Model, original_head)
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    optimizer = BacktrackingLineSearch(
        batches_per_epoch=batches_per_epoch,
        scaling_factor=scaling_factor,
    )

    tf_surrogate_model.compile(optimizer=optimizer, loss=mse_loss)
    for _ in range(epochs):
        for batch in train_set:
            inputs, _ = _split_batch_inputs_targets(batch)
            z_batch = backend.forward(tf_feature_extractor, inputs)
            y_target = backend.forward(tf_original_head, z_batch)
            with tf.GradientTape() as tape:
                logits = tf_surrogate_model(z_batch, training=True)
                loss = mse_loss(y_target, logits)
                if tf_surrogate_model.losses:
                    regularization_loss = tf.add_n([
                        tf.cast(loss_term, loss.dtype)
                        for loss_term in tf_surrogate_model.losses
                    ])
                    loss = loss + regularization_loss
            gradients = tape.gradient(loss, tf_surrogate_model.trainable_weights)
            optimizer.step(tf_surrogate_model, loss, z_batch, y_target, gradients)

    tf_surrogate_model.compile(optimizer=optimizer, loss=loss_function)
    return tf_surrogate_model


def perturb_head_single_sgd_step_tensorflow(
    backend: BaseBackend,
    perturbed_head: Model,
    feature_extractor: Model,
    feature_dataset: DatasetLike,
    loss_function: Callable,
    learning_rate: float = 1e-4,
) -> Model:
    """Apply one TensorFlow SGD step to the cloned head."""
    tf_perturbed_head = cast(tf.keras.Model, perturbed_head)
    tf_feature_extractor = cast(tf.keras.Model, feature_extractor)
    if not tf_perturbed_head.built and hasattr(tf_feature_extractor, 'output_shape'):
        tf_perturbed_head.build(tf_feature_extractor.output_shape)

    trainable_vars = list(tf_perturbed_head.trainable_variables)
    if not trainable_vars:
        return tf_perturbed_head

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    accum_vars = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in trainable_vars]
    count_dtype = trainable_vars[0].dtype
    total_samples = tf.zeros((), dtype=count_dtype)

    for feature_batch, target_batch in feature_dataset:
        with tf.GradientTape() as tape:
            logits = backend.forward(tf_perturbed_head, feature_batch)
            normalized_targets = backend.normalize_binary_targets(target_batch, logits)
            loss = loss_function(normalized_targets, logits)
            loss = -backend.reduce_mean(backend.ensure_per_sample_loss(loss))
        gradients = tape.gradient(loss, trainable_vars)

        batch_size = backend.cast(backend.get_batch_size(feature_batch), count_dtype)
        total_samples = total_samples + batch_size
        for idx, gradient in enumerate(gradients):
            if gradient is None:
                raise ValueError("Gradient is None while computing perturbed-head update for RPS-LJE")
            accum_vars[idx].assign_add(gradient * tf.cast(batch_size, gradient.dtype))

    mean_grads = [accum_var / tf.cast(total_samples, accum_var.dtype) for accum_var in accum_vars]
    optimizer.apply_gradients(zip(mean_grads, trainable_vars))
    return tf_perturbed_head


def compute_lje_second_term_tensorflow(
    backend: BaseBackend,
    ihvp_calculator: Any,
    scaled_jacobian: Tensor,
) -> Tensor:
    """Compute the TensorFlow-specific IHVP term for RPS-LJE."""
    tf_scaled_jacobian = cast(tf.Tensor, scaled_jacobian)
    second_term = backend.map_fn(
        lambda value: ihvp_calculator._compute_ihvp_single_batch(  # pylint: disable=protected-access
            tf.expand_dims(value, axis=0),
            use_gradient=False,
        ),
        tf_scaled_jacobian,
    )
    return tf.reshape(second_term, tf.shape(tf_scaled_jacobian))
