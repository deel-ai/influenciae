# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Exact bounded-memory inner-product search over representation shards."""
# pylint: disable=too-many-branches
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..common.representation_store import RepresentationStore


@dataclass(frozen=True)
class VectorSearchResult:
    """Scores, stable sample IDs, and optional payloads for a vector search."""

    scores: np.ndarray
    ids: np.ndarray
    payload: Optional[np.ndarray]

    def __iter__(self):
        """Allow tuple unpacking as ``scores, ids, payload``."""
        return iter((self.scores, self.ids, self.payload))


class ExactStreamingVectorIndex:
    """Scan immutable shards while bounding temporary score matrices.

    Ties are resolved by ascending sample ID for both score orders, making the
    result independent of shard size and insertion grouping.
    """

    def __init__(self, store: RepresentationStore, *, max_score_block_elements: int = 1_000_000) -> None:
        if not isinstance(store, RepresentationStore):
            raise TypeError("store must implement RepresentationStore.")
        if isinstance(max_score_block_elements, bool) or not isinstance(max_score_block_elements, int) \
                or max_score_block_elements <= 0:
            raise ValueError("max_score_block_elements must be a strictly positive integer.")
        self.store = store
        self.max_score_block_elements = max_score_block_elements

    def search(
        self,
        queries: Any,
        k: int,
        *,
        order: str = "descending",
        return_payload: bool = True,
    ) -> VectorSearchResult:
        """Return exact inner-product neighbors for each query row."""
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError("k must be a strictly positive integer.")
        if k > self.store.count:
            raise ValueError(f"k={k} exceeds the store count ({self.store.count}).")
        if order not in ("ascending", "descending"):
            raise ValueError("order must be either 'ascending' or 'descending'.")
        query_matrix = np.asarray(queries)
        was_vector = query_matrix.ndim == 1
        if was_vector:
            query_matrix = query_matrix[None, :]
        if query_matrix.ndim != 2 or query_matrix.shape[1] != self.store.vector_dim:
            raise ValueError(
                f"queries must have shape (queries, {self.store.vector_dim}) or ({self.store.vector_dim},)."
            )
        if not np.issubdtype(query_matrix.dtype, np.number) or np.iscomplexobj(query_matrix):
            raise ValueError("queries must have a real numeric dtype.")
        if not np.all(np.isfinite(query_matrix)):
            raise ValueError("queries must contain only finite values.")

        query_count = query_matrix.shape[0]
        score_dtype = np.result_type(query_matrix.dtype, self.store.vector_dtype)
        all_scores = np.empty((query_count, int(k)), dtype=score_dtype)
        all_ids = np.empty((query_count, int(k)), dtype=np.int64)
        query_block_size = min(query_count or 1, self.max_score_block_elements)

        for query_start in range(0, query_count, query_block_size):
            query_block = query_matrix[query_start:query_start + query_block_size]
            rows = query_block.shape[0]
            vector_block_size = max(1, self.max_score_block_elements // max(rows, 1))
            best_scores = np.empty((rows, 0), dtype=score_dtype)
            best_ids = np.empty((rows, 0), dtype=np.int64)
            for shard in self.store.iter_shards():
                if not np.all(np.isfinite(shard.vectors)):
                    raise ValueError("Stored vectors must contain only finite values.")
                for vector_start in range(0, len(shard.ids), vector_block_size):
                    vector_stop = vector_start + vector_block_size
                    ids = np.asarray(shard.ids[vector_start:vector_stop])
                    scores = np.asarray(query_block @ shard.vectors[vector_start:vector_stop].T)
                    candidate_scores = np.concatenate((best_scores, scores), axis=1)
                    candidate_ids = np.concatenate(
                        (best_ids, np.broadcast_to(ids, (rows, ids.size))), axis=1
                    )
                    take = min(int(k), candidate_scores.shape[1])
                    selected = np.empty((rows, take), dtype=np.int64)
                    for row in range(rows):
                        primary = -candidate_scores[row] if order == "descending" else candidate_scores[row]
                        selected[row] = np.lexsort((candidate_ids[row], primary))[:take]
                    best_scores = np.take_along_axis(candidate_scores, selected, axis=1)
                    best_ids = np.take_along_axis(candidate_ids, selected, axis=1)
            all_scores[query_start:query_start + rows] = best_scores
            all_ids[query_start:query_start + rows] = best_ids

        payload = self.store.gather(all_ids) if return_payload else None
        if was_vector:
            return VectorSearchResult(all_scores[0], all_ids[0], None if payload is None else payload[0])
        return VectorSearchResult(all_scores, all_ids, payload)

    query = search


__all__ = ["ExactStreamingVectorIndex", "VectorSearchResult"]
