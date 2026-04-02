# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""PyTorch-specific helpers for RPS model surgery."""
from typing import Any, Callable

import torch
import torch.nn as nn

from ..common import BaseBackend
from ..types import DatasetLike, Model, Tensor
from .backtracking_line_search import BacktrackingLineSearchPyTorch
from .model_surgery import split_batch_inputs_targets


def _move_to_reference_device(obj: Any, reference_parameter: torch.Tensor) -> Any:
    """Move tensors nested in tuples/lists to the reference parameter device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=reference_parameter.device)
    if isinstance(obj, tuple):
        return tuple(_move_to_reference_device(item, reference_parameter) for item in obj)
    if isinstance(obj, list):
        return [_move_to_reference_device(item, reference_parameter) for item in obj]
    return obj


def train_surrogate_linear_model_pytorch(
    backend: BaseBackend,
    surrogate_model: Model,
    feature_extractor: Model,
    original_head: Model,
    train_set: DatasetLike,
    scaling_factor: float,
    epochs: int,
    batches_per_epoch: int,
) -> Model:
    """Fit the surrogate linear model with the PyTorch line-search optimizer."""
    mse_loss = nn.MSELoss(reduction="mean")
    optimizer = BacktrackingLineSearchPyTorch(
        params=surrogate_model.parameters(),
        batches_per_epoch=batches_per_epoch,
        scaling_factor=scaling_factor,
    )

    weight = next(surrogate_model.parameters())
    surrogate_model.train()
    for _ in range(epochs):
        for batch in train_set:
            inputs, _ = split_batch_inputs_targets(batch)
            inputs = _move_to_reference_device(inputs, weight)

            with torch.no_grad():
                z_batch = backend.forward(feature_extractor, inputs)
                y_target = backend.forward(original_head, z_batch)

            optimizer.zero_grad()
            logits = surrogate_model(z_batch)
            mse = mse_loss(logits, y_target)
            loss = mse
            regularization_losses = [
                loss_term.to(device=mse.device, dtype=mse.dtype)
                for loss_term in getattr(surrogate_model, 'losses', [])
            ]
            if regularization_losses:
                loss = loss + torch.stack(regularization_losses).sum()

            if not torch.isfinite(loss):
                continue

            loss.backward()
            gradient = weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_([weight], max_norm=10.0)
            gradients = [
                parameter.grad.clone()
                for parameter in surrogate_model.parameters()
                if parameter.grad is not None
            ]

            def closure(z_batch_local=z_batch, y_target_local=y_target):
                with torch.no_grad():
                    logits_new = surrogate_model(z_batch_local)
                    loss_new = mse_loss(logits_new, y_target_local)
                    regularization_losses_new = [
                        loss_term.to(device=loss_new.device, dtype=loss_new.dtype)
                        for loss_term in getattr(surrogate_model, 'losses', [])
                    ]
                    if regularization_losses_new:
                        loss_new = loss_new + torch.stack(regularization_losses_new).sum()
                    return loss_new

            optimizer.step(
                model=surrogate_model,
                current_loss=loss.detach(),
                x_inputs=z_batch,
                labels=y_target,
                gradients=gradients,
                closure=closure,
            )

    surrogate_model.eval()
    return surrogate_model


def perturb_head_single_sgd_step_pytorch(
    backend: BaseBackend,
    original_head: Model,
    perturbed_head: Model,
    feature_dataset: DatasetLike,
    loss_function: Callable,
    learning_rate: float = 1e-4,
) -> Model:
    """Apply one PyTorch SGD step to the cloned head."""
    reference_parameter = next(original_head.parameters())
    perturbed_head = perturbed_head.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
    perturbed_head.train()

    optimizer = torch.optim.SGD(perturbed_head.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    dtype = next(perturbed_head.parameters()).dtype
    total_loss = torch.tensor(0.0, device=reference_parameter.device, dtype=dtype)
    total_samples = 0

    for feature_batch, target_batch in feature_dataset:
        feature_batch = _move_to_reference_device(feature_batch, reference_parameter)
        target_batch = _move_to_reference_device(target_batch, reference_parameter)
        logits = backend.forward(perturbed_head, feature_batch)
        normalized_targets = backend.normalize_binary_targets(target_batch, logits)
        per_sample_loss = backend.ensure_per_sample_loss(loss_function(logits, normalized_targets))
        total_loss = total_loss - per_sample_loss.sum()
        total_samples += int(backend.get_batch_size(feature_batch))

    (total_loss / float(total_samples)).backward()
    optimizer.step()
    perturbed_head.eval()
    return perturbed_head


def compute_lje_second_term_pytorch(
    backend: BaseBackend,
    ihvp_calculator: Any,
    scaled_jacobian: Tensor,
) -> Tensor:
    """Compute the PyTorch-specific IHVP term for RPS-LJE."""
    batch_size = int(backend.get_batch_size(scaled_jacobian))
    second_term_batches = []
    for idx in range(batch_size):
        # PyTorch linear weights are stored as (out, in), so transpose to match the
        # flattened parameter layout expected by the IHVP calculator.
        grad_flat = scaled_jacobian[idx].permute(1, 0).reshape(1, -1)
        ihvp_result = ihvp_calculator._compute_ihvp_single_batch(  # pylint: disable=protected-access
            (grad_flat,),
            use_gradient=False,
        )
        second_term_batches.append(ihvp_result.squeeze())

    second_term = backend.stack(second_term_batches, axis=0)
    return backend.reshape(second_term, backend.tensor_shape(scaled_jacobian))
