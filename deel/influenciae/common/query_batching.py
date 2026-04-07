# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Configuration and helpers for query-batched influence computation.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .backend import BaseBackend
from .kfac_factors import LayerInfo
from ..types import Tensor
from ..utils.partitioning import chunk_items


class PreconditioningMode(Enum):
    """Controls whether the IHVP is applied to training or query gradients."""

    TRAIN = "train"
    QUERY = "query"


@dataclass
class QueryBatchingConfig:
    """Configuration for query-side preconditioning and score computation.

    Parameters
    ----------
    query_gradient_low_rank
        Optional rank used to compress preconditioned query gradients with a
        truncated SVD. ``None`` disables low-rank compression.
    query_gradient_svd_dtype
        Optional dtype used for the SVD compression step. This can be used to
        upcast query gradients before factorization for better numerical
        stability.
    query_gradient_accumulation_steps
        Number of query batches to precondition and merge before scoring them
        against the training set.
    score_data_partitions
        Number of contiguous partitions to split each training batch into while
        computing scores. Larger values reduce peak memory usage at the cost of
        more scoring passes.
    score_module_partitions
        Number of contiguous layer partitions used when scoring per-module
        low-rank query representations. This has no effect for dense query
        representations or for global low-rank compression.
    """

    query_gradient_low_rank: Optional[int] = None
    query_gradient_svd_dtype: Optional[object] = None
    query_gradient_accumulation_steps: int = 1
    score_data_partitions: int = 1
    score_module_partitions: int = 1

    def validate(self) -> None:
        """Validate query batching settings."""
        if self.query_gradient_accumulation_steps <= 0:
            raise ValueError("query_gradient_accumulation_steps must be strictly positive.")
        if self.score_data_partitions <= 0:
            raise ValueError("score_data_partitions must be strictly positive.")
        if self.score_module_partitions <= 0:
            raise ValueError("score_module_partitions must be strictly positive.")
        if self.query_gradient_low_rank is not None and self.query_gradient_low_rank <= 0:
            raise ValueError("query_gradient_low_rank must be strictly positive or None.")


LowRankModuleValue = Union[Tensor, Tuple[Tensor, Tensor]]


class LowRankGradient:
    """Compressed query gradients used during score computation.

    Values are stored per module. A module can either remain full-rank as a tensor
    of shape ``(batch, rows, cols)`` or be represented as a tuple ``(left, right)``
    where ``left`` has shape ``(batch, rows, rank)`` and ``right`` has shape
    ``(batch, rank, cols)``.
    """

    def __init__(self, module_values: Dict[int, LowRankModuleValue], is_global: bool = False):
        self.module_values = module_values
        self.is_global = is_global

    def concat(self, others: Sequence["LowRankGradient"], backend: BaseBackend) -> "LowRankGradient":
        """Concatenate low-rank gradients along the batch dimension."""
        merged: Dict[int, LowRankModuleValue] = {}
        all_grads = [self, *others]
        keys = self.module_values.keys()
        for key in keys:
            sample_value = self.module_values[key]
            if isinstance(sample_value, tuple):
                left_parts = [grad.module_values[key][0] for grad in all_grads]  # type: ignore[index]
                right_parts = [grad.module_values[key][1] for grad in all_grads]  # type: ignore[index]
                merged[key] = (
                    backend.concat(left_parts, axis=0),
                    backend.concat(right_parts, axis=0),
                )
            else:
                merged[key] = backend.concat([grad.module_values[key] for grad in all_grads], axis=0)
        return LowRankGradient(merged, is_global=self.is_global)

    def dot_with_modules(
        self,
        module_tensors: Dict[int, Tensor],
        backend: BaseBackend,
    ) -> Tensor:
        """Compute dot products between compressed query grads and module gradients."""
        if self.is_global:
            value = self.module_values[-1]
            if not isinstance(value, tuple):
                raise ValueError("Global low-rank gradients must store a (left, right) tuple.")
            left, right = value
            train_tensor = module_tensors[-1]
            projected = backend.matmul(right, backend.transpose(train_tensor))
            return backend.matmul(left, projected)

        scores = None
        for layer_idx, value in self.module_values.items():
            train_tensor = module_tensors[layer_idx]
            if isinstance(value, tuple):
                left, right = value
                layer_scores = _einsum_low_rank(left, right, train_tensor, backend)
            else:
                layer_scores = _einsum_full_rank(value, train_tensor, backend)
            scores = layer_scores if scores is None else scores + layer_scores

        if scores is None:
            raise ValueError("LowRankGradient contains no module values.")
        return scores


def make_module_partitions(
    layer_infos: Sequence[LayerInfo],
    num_partitions: int,
) -> List[List[LayerInfo]]:
    """Split layer infos into a fixed number of contiguous partitions."""
    if num_partitions <= 1 or len(layer_infos) == 0:
        return [list(layer_infos)]

    partition_size = max(1, (len(layer_infos) + num_partitions - 1) // num_partitions)
    return chunk_items(layer_infos, partition_size)


def concat_query_representations(
    representations: Sequence[Union[Tensor, LowRankGradient]],
    backend: BaseBackend,
) -> Union[Tensor, LowRankGradient]:
    """Concatenate query representations along the batch dimension."""
    if not representations:
        raise ValueError("Expected at least one query representation to concatenate.")

    first = representations[0]
    if isinstance(first, LowRankGradient):
        return first.concat([rep for rep in representations[1:] if isinstance(rep, LowRankGradient)], backend)
    return backend.concat(list(representations), axis=0)


def _einsum_full_rank(query_tensor: Tensor, train_tensor: Tensor, backend: BaseBackend) -> Tensor:
    """Compute ``sum(query * train)`` across all non-batch dimensions."""
    query_flat = backend.reshape(query_tensor, (backend.get_batch_size(query_tensor), -1))
    train_flat = backend.reshape(train_tensor, (backend.get_batch_size(train_tensor), -1))
    return backend.matmul(query_flat, backend.transpose(train_flat))


def _einsum_low_rank(left: Tensor, right: Tensor, train_tensor: Tensor, backend: BaseBackend) -> Tensor:
    """Compute dot products for a low-rank module representation."""
    return backend.einsum("qor,toi,qri->qt", left, train_tensor, right)
