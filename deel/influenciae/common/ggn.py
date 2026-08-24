# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Matrix-free generalized Gauss-Newton operators."""
from typing import Any, Tuple

from ..types import Tensor
from .model_wrappers import BaseInfluenceModel


class GeneralizedGaussNewtonOperator:
    """Bind a raw batch to a true GGN operator for an influence model."""

    def __init__(
        self,
        model: BaseInfluenceModel,
        batch: Tuple[Any, ...],
        batch_reduction: str = 'mean',
    ):
        if batch_reduction not in ('sum', 'mean'):
            raise ValueError("batch_reduction must be either 'sum' or 'mean'.")
        self.model = model
        self.batch = batch
        self.batch_reduction = batch_reduction

    def matvec(self, vector: Tensor) -> Tensor:
        """Apply the batch GGN to one flat vector of shape ``(P,)``."""
        shape = self.model.backend.tensor_shape(vector)
        expected = self.model.parameter_layout.total_size
        if len(shape) != 1 or (shape[0] is not None and shape[0] != expected):
            raise ValueError(f"GGN vector must have shape ({expected},); got {shape}.")
        tangents = [
            self.model.backend.reshape(vector[entry.flat_slice], entry.shape)
            for entry in self.model.parameter_layout.entries
        ]
        inputs, targets, sample_weight = self.model.process_batch_for_loss_fn(self.batch)
        return self.model.backend.compute_ggn_vector_product(
            self.model.model,
            self.model.weights,
            self.model.loss_function,
            tangents,
            inputs,
            targets,
            sample_weight=sample_weight,
            batch_reduction=self.batch_reduction,
        )

    def matmat(self, vectors: Tensor) -> Tensor:
        """Apply the batch GGN to row-batched vectors of shape ``(R, P)``."""
        shape = self.model.backend.tensor_shape(vectors)
        expected = self.model.parameter_layout.total_size
        if len(shape) != 2 or (shape[1] is not None and shape[1] != expected):
            raise ValueError(f"GGN vectors must have shape (R, {expected}); got {shape}.")
        return self.model.backend.map_fn(self.matvec, vectors)
