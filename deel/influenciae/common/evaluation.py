# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Helpers for customizing evaluation/query representations.
"""
from typing import Any, Callable, Optional, Protocol, Tuple, TypeAlias

from ..types import LossFunction, Tensor
from .model_wrappers import BaseInfluenceModel

ProcessBatchForObjectiveTypeAlias: TypeAlias = Callable[
    [Tuple[Any, ...]],
    Tuple[Any, Any, Optional[Any]],
]


class EvaluationRepresentationProvider(Protocol):
    """Protocol for custom query/evaluation representations."""

    def __call__(
        self,
        model: BaseInfluenceModel,
        batch: Tuple[Any, ...],
    ) -> Tensor:
        """Return a representation of shape ``(batch_size, nb_params)``."""


class ObjectiveEvaluationRepresentationProvider:
    """Build a query representation from a per-sample objective."""

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
        model_inp, y_true, sample_weight = self.process_batch_for_objective_fn(batch)
        return model.backend.compute_jacobian(
            model.model,
            model.weights,
            self.objective,
            model_inp,
            y_true,
            sample_weight,
        )
