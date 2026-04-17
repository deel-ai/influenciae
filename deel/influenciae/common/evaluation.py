# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Helpers for customizing how evaluation or query batches are represented.

Influence methods compare training samples against a representation of the
evaluation batch. For simple supervised models, the library's default
preprocessing path is often enough. More structured tasks, however, may need to
express an evaluation batch through the gradient of a custom per-sample
objective computed from a packed batch format.

This module defines the public hook used to provide those custom
representations, together with a reusable implementation that turns a batch
into the Jacobian of a user-supplied objective. Object-detection adapters use
these abstractions to evaluate images against detection-specific objectives
without changing the influence core.
"""
from typing import Any, Callable, Optional, Protocol, Tuple, TypeAlias

from ..types import LossFunction, Tensor
from .model_wrappers import BaseInfluenceModel

#: Adapt a raw dataset batch to the ``(model_input, y_true, sample_weight)``
#: triple expected by ``backend.compute_jacobian``. This isolates task-specific
#: batch packing logic from the generic influence implementation.
ProcessBatchForObjectiveTypeAlias: TypeAlias = Callable[
    [Tuple[Any, ...]],
    Tuple[Any, Any, Optional[Any]],
]


class EvaluationRepresentationProvider(Protocol):
    """Callable building the representation used to score an evaluation batch.

    Implement this protocol when the default evaluation preprocessing is not
    expressive enough. Typical examples include structured outputs, such as
    object detection, where a raw batch must first be unpacked and then scored
    through a custom per-sample objective.

    The returned tensor must be batched along the first dimension and aligned
    with the watched parameters of ``model``. For standard first-order
    influence methods this usually means a tensor of shape
    ``(batch_size, nb_params)``.
    """

    def __call__(
        self,
        model: BaseInfluenceModel,
        batch: Tuple[Any, ...],
    ) -> Tensor:
        """Build the representation associated with ``batch``.

        Parameters
        ----------
        model
            Influence-model wrapper exposing the underlying model, watched
            weights, and backend helper methods.
        batch
            Raw batch produced by the evaluation/query dataset.

        Returns
        -------
        Tensor
            Batched representation used by the influence calculator. The first
            dimension must match the batch size.
        """


class ObjectiveEvaluationRepresentationProvider:
    """Build evaluation representations as per-sample gradients of an objective.

    This is the standard adapter when an evaluation batch should be represented
    by differentiating a custom objective instead of relying on the model's
    default preprocessing path. A common use case is object detection, where a
    packed batch must first be unpacked into ``(images, targets,
    sample_weight)`` before computing the gradient of a detection-specific
    objective.

    Parameters
    ----------
    objective
        Loss-like callable evaluated independently for each sample. It must be
        compatible with ``backend.compute_jacobian`` and therefore return one
        value per-sample with no batch reduction.
    process_batch_for_objective_fn
        Callable converting a raw dataset batch to the
        ``(model_input, y_true, sample_weight)`` triple consumed by
        ``objective`` and ``backend.compute_jacobian``.
    """

    def __init__(
        self,
        objective: LossFunction,
        process_batch_for_objective_fn: ProcessBatchForObjectiveTypeAlias,
    ) -> None:
        self.objective = objective
        self.process_batch_for_objective_fn = process_batch_for_objective_fn

    def __call__(
        self,
        model: BaseInfluenceModel,
        batch: Tuple[Any, ...],
    ) -> Tensor:
        """Return the batched objective Jacobian for ``batch``.

        The result is a tensor whose first dimension matches the number of
        samples in ``batch`` and whose remaining dimension spans the watched
        parameters of ``model``.
        """
        model_inp, y_true, sample_weight = self.process_batch_for_objective_fn(batch)
        return model.backend.compute_jacobian(
            model.model,
            model.weights,
            self.objective,
            model_inp,
            y_true,
            sample_weight,
        )
