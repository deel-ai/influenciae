# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
RPS-specific model helpers.

These helpers keep the RPS module backend-agnostic while delegating genuinely
framework-specific training loops to private utility modules.
"""
from typing import Any, Callable, Optional, Tuple

from ..common.backend import BaseBackend, Framework
from ..types import DatasetLike, Model, Tensor


def split_batch_inputs_targets(samples: Tuple[Any, ...]) -> Tuple[Any, Any]:
    """Split a batched sample tuple into inputs and targets."""
    if len(samples) == 2:
        return samples[0], samples[1]

    inputs = samples[:-1]
    if isinstance(inputs, tuple) and len(inputs) == 1:
        inputs = inputs[0]
    return inputs, samples[-1]


def _normalize_shape(shape: Any) -> Tuple[Any, ...]:
    """Normalize backend shape metadata to a tuple."""
    if isinstance(shape, list):
        if not shape:
            raise ValueError("Expected a non-empty shape list")
        shape = shape[0]
    return tuple(shape)


def create_surrogate_linear_model(
    backend: BaseBackend,
    feature_extractor: Model,
    original_head: Model,
    lambda_regularization: float,
) -> Model:
    """Create an L2-regularized surrogate linear model matching the original head."""
    linear_layer = backend.get_layers(original_head)[-1]
    in_features, _ = backend.get_layer_io_features(linear_layer)

    feature_shape = _normalize_shape(backend.get_output_shape(feature_extractor))
    if len(feature_shape) <= 1 or feature_shape[-1] is None:
        input_shape = (in_features,)
    else:
        input_shape = tuple(feature_shape[1:])

    feature_dtype = getattr(feature_extractor, 'compute_dtype', None)
    if feature_dtype is None:
        feature_dtype = getattr(feature_extractor, 'dtype', None)

    output_shape = _normalize_shape(backend.get_output_shape(original_head))
    output_units = output_shape[-1] if len(output_shape) > 1 else 1

    reference_weights = backend.get_model_weights(original_head, [linear_layer])
    reference_weight = reference_weights[0] if reference_weights else None
    return backend.create_linear_model(
        input_shape=input_shape,
        out_features=output_units,
        use_bias=False,
        l2_regularization=lambda_regularization,
        dtype=feature_dtype,
        reference_weight=reference_weight,
    )


def train_surrogate_linear_model(  # pylint: disable=import-outside-toplevel
    backend: BaseBackend,
    surrogate_model: Model,
    feature_extractor: Model,
    original_head: Model,
    train_set: DatasetLike,
    loss_function: Callable,
    scaling_factor: float,
    epochs: int,
) -> Model:
    """Fit the surrogate linear model with backend-specific optimizer details."""
    n_train = backend.get_dataset_size(train_set)
    batch_size = backend.get_dataset_batch_size(train_set)
    batches_per_epoch = max((n_train + batch_size - 1) // batch_size, 1)

    if backend.framework == Framework.TENSORFLOW:
        from ._model_surgery_tf import train_surrogate_linear_model_tensorflow

        return train_surrogate_linear_model_tensorflow(
            backend=backend,
            surrogate_model=surrogate_model,
            feature_extractor=feature_extractor,
            original_head=original_head,
            train_set=train_set,
            loss_function=loss_function,
            scaling_factor=scaling_factor,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
        )

    if backend.framework == Framework.PYTORCH:
        from ._model_surgery_pytorch import train_surrogate_linear_model_pytorch

        return train_surrogate_linear_model_pytorch(
            backend=backend,
            surrogate_model=surrogate_model,
            feature_extractor=feature_extractor,
            original_head=original_head,
            train_set=train_set,
            scaling_factor=scaling_factor,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
        )

    raise ValueError(f"Unsupported backend framework: {backend.framework}")


def _prepare_feature_dataset(
    backend: BaseBackend,
    feature_extractor: Model,
    dataset: DatasetLike,
    n_samples_for_hessian: Optional[int],
    shuffle_buffer_size: int,
) -> DatasetLike:
    """Materialize the feature-extractor outputs used by RPS-LJE Hessian estimation."""
    batch_size = backend.get_dataset_batch_size(dataset)
    dataset_to_collect = dataset

    if n_samples_for_hessian is not None:
        n_batches = max(n_samples_for_hessian // batch_size, 1)
        dataset_to_collect = backend.shuffle_dataset(dataset_to_collect, shuffle_buffer_size)
        dataset_to_collect = backend.take_dataset(dataset_to_collect, n_batches)

    feature_batches = []
    target_batches = []
    for batch in dataset_to_collect:
        inputs, targets = split_batch_inputs_targets(batch)
        feature_batches.append(backend.forward(feature_extractor, inputs))
        target_batches.append(targets)

    if not feature_batches:
        raise ValueError("Dataset used for Hessian estimation is empty")

    features = backend.concat(feature_batches, axis=0)
    targets = backend.concat(target_batches, axis=0)
    return backend.create_dataset_from_tensor_slices((features, targets), batch_size=batch_size)


def perturb_head_single_sgd_step(  # pylint: disable=import-outside-toplevel
    backend: BaseBackend,
    original_head: Model,
    feature_extractor: Model,
    dataset: DatasetLike,
    loss_function: Callable,
    n_samples_for_hessian: Optional[int],
    shuffle_buffer_size: int,
    learning_rate: float = 1e-4,
) -> Tuple[Model, DatasetLike]:
    """Clone a head and apply a single SGD step on the negated mean loss."""
    feature_dataset = _prepare_feature_dataset(
        backend,
        feature_extractor,
        dataset,
        n_samples_for_hessian,
        shuffle_buffer_size,
    )
    perturbed_head = backend.clone_model(original_head)

    if backend.framework == Framework.TENSORFLOW:
        from ._model_surgery_tf import perturb_head_single_sgd_step_tensorflow

        perturbed_head = perturb_head_single_sgd_step_tensorflow(
            backend=backend,
            perturbed_head=perturbed_head,
            feature_extractor=feature_extractor,
            feature_dataset=feature_dataset,
            loss_function=loss_function,
            learning_rate=learning_rate,
        )
    elif backend.framework == Framework.PYTORCH:
        from ._model_surgery_pytorch import perturb_head_single_sgd_step_pytorch

        perturbed_head = perturb_head_single_sgd_step_pytorch(
            backend=backend,
            original_head=original_head,
            perturbed_head=perturbed_head,
            feature_dataset=feature_dataset,
            loss_function=loss_function,
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"Unsupported backend framework: {backend.framework}")

    return perturbed_head, feature_dataset


def _scale_linear_jacobian(
    backend: BaseBackend,
    jacobian: Tensor,
    feature_maps: Tensor,
    epsilon: float,
    batch_scale: Optional[Any] = None,
) -> Tensor:
    """Scale a linear-layer Jacobian by the feature maps in backend-native layout."""
    _, output_axis = backend.get_linear_weight_axes()
    jacobian_dtype = backend.get_dtype(jacobian)
    feature_maps = backend.cast(feature_maps, jacobian_dtype)
    eps = backend.cast(epsilon, jacobian_dtype)
    denominator = feature_maps + eps

    if batch_scale is not None:
        denominator = backend.cast(batch_scale, jacobian_dtype) * feature_maps + eps

    reciprocal = backend.expand_dims(backend.ones_like(feature_maps) / denominator, axis=output_axis + 1)
    output_size = backend.tensor_shape(jacobian)[output_axis + 1]
    return jacobian * backend.repeat(reciprocal, output_size, axis=output_axis + 1)


def _reduce_linear_jacobian(backend: BaseBackend, jacobian: Tensor) -> Tensor:
    """Reduce a backend-native linear Jacobian over its input-feature axis."""
    input_axis, _ = backend.get_linear_weight_axes()
    return backend.reduce_sum(jacobian, axis=input_axis + 1)


def _compute_first_term_from_weight(
    backend: BaseBackend,
    weight: Tensor,
    feature_maps: Tensor,
    epsilon: float,
    batch_scale: Any,
) -> Tensor:
    """Compute the first RPS-LJE term from the perturbed head weight tensor."""
    input_axis, output_axis = backend.get_linear_weight_axes()
    weight_dtype = backend.get_dtype(weight)
    weight = backend.cast(weight, weight_dtype)
    feature_maps = backend.cast(feature_maps, weight_dtype)
    eps = backend.cast(epsilon, weight_dtype)
    denominator = backend.cast(batch_scale, weight_dtype) * feature_maps + eps
    reciprocal = backend.expand_dims(backend.ones_like(feature_maps) / denominator, axis=output_axis + 1)
    output_size = backend.tensor_shape(weight)[output_axis]
    repeated = backend.repeat(reciprocal, output_size, axis=output_axis + 1)
    return backend.reduce_sum(backend.expand_dims(weight, axis=0) * repeated, axis=input_axis + 1)


def compute_l2_alpha(
    backend: BaseBackend,
    linear_layer: Model,
    loss_function: Callable,
    z_batch: Tensor,
    y_batch: Tensor,
    n_train: int,
    lambda_regularization: float,
    epsilon: float = 1e-5,
) -> Tensor:
    """Compute the RPS-L2 alpha coefficients."""
    weights = backend.get_model_weights(linear_layer)
    logits = backend.forward(linear_layer, z_batch)
    normalized_targets = backend.normalize_binary_targets(y_batch, logits)
    jacobian = backend.compute_jacobian(linear_layer, weights, loss_function, z_batch, normalized_targets)
    jacobian = backend.reshape(jacobian, (-1, *backend.tensor_shape(weights[0])))
    jacobian = jacobian / (-2.0 * lambda_regularization * float(n_train) + epsilon)
    jacobian = _scale_linear_jacobian(backend, jacobian, z_batch, epsilon)
    return _reduce_linear_jacobian(backend, jacobian)


def compute_lje_alpha(  # pylint: disable=import-outside-toplevel
    backend: BaseBackend,
    perturbed_head: Model,
    ihvp_calculator: Any,
    loss_function: Callable,
    z_batch: Tensor,
    y_batch: Tensor,
    epsilon: float,
) -> Tensor:
    """Compute the RPS-LJE alpha coefficients."""
    weights = backend.get_model_weights(perturbed_head)
    logits = backend.forward(perturbed_head, z_batch)
    normalized_targets = backend.normalize_binary_targets(y_batch, logits)

    jacobian = backend.compute_jacobian(perturbed_head, weights, loss_function, z_batch, normalized_targets)
    jacobian = backend.reshape(jacobian, (-1, *backend.tensor_shape(weights[0])))

    batch_size = backend.get_batch_size(z_batch)
    scaled_jacobian = _scale_linear_jacobian(backend, jacobian, z_batch, epsilon, batch_scale=batch_size)

    if backend.framework == Framework.TENSORFLOW:
        from ._model_surgery_tf import compute_lje_second_term_tensorflow

        second_term = compute_lje_second_term_tensorflow(
            backend=backend,
            ihvp_calculator=ihvp_calculator,
            scaled_jacobian=scaled_jacobian,
        )
    elif backend.framework == Framework.PYTORCH:
        from ._model_surgery_pytorch import compute_lje_second_term_pytorch

        second_term = compute_lje_second_term_pytorch(
            backend=backend,
            ihvp_calculator=ihvp_calculator,
            scaled_jacobian=scaled_jacobian,
        )
    else:
        raise ValueError(f"Unsupported backend framework: {backend.framework}")

    second_term = _reduce_linear_jacobian(backend, second_term)
    first_term = _compute_first_term_from_weight(backend, weights[0], z_batch, epsilon, batch_size)
    return first_term - second_term
