# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Paper-faithful TrackStar fitting, representation, and exact search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, Union

import numpy as np

from ..common.model_wrappers import BaseInfluenceModel
from ..common.payloads import TrainingPayloadExtractor, default_training_payload_extractor
from ..common.representation_store import MemoryRepresentationStore, RepresentationStore
from .index import ExactStreamingVectorIndex, VectorSearchResult
from .optimizer_state import OptimizerSecondMoments, extract_optimizer_second_moments
from .projected_curvature import (
    AutoMix,
    GramSnapshot,
    MixRule,
    ProjectedCurvature,
    ProjectedGramAccumulator,
    build_projected_curvature,
)
from .projection import TrackStarProjectionPlan


Batch = Tuple[Any, ...]
BatchIdExtractor = Callable[[Batch, int], Any]
StoreFactory = Callable[..., Any]


@dataclass(frozen=True)
class TrackStarTopKBatch:
    """Exact training neighbors for one raw query batch."""

    query_ids: np.ndarray
    query_batch: Batch
    neighbors: VectorSearchResult


def _create_store(
    factory: Optional[StoreFactory],
    vector_dim: int,
    *,
    vector_dtype: Any,
    payload: Optional[np.ndarray],
    stage: str,
) -> Any:
    options = {
        "vector_dim": vector_dim,
        "vector_dtype": vector_dtype,
        "payload_shape": None if payload is None else payload.shape[1:],
        "payload_dtype": None if payload is None else payload.dtype,
        "stage": stage,
    }
    store = MemoryRepresentationStore(**options) if factory is None else factory(**options)
    if not all(hasattr(store, name) for name in ("append", "finalize", "iter_shards")):
        raise TypeError("A TrackStar store factory must return a writable representation store.")
    return store


def _batch_ids(
    batch: Batch,
    batch_size: int,
    next_id: int,
    extractor: Optional[BatchIdExtractor],
) -> np.ndarray:
    values = np.arange(next_id, next_id + batch_size, dtype=np.int64) \
        if extractor is None else np.asarray(extractor(batch, next_id))
    if values.shape != (batch_size,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Batch IDs must be a one-dimensional integer array matching the batch size.")
    cast = values.astype(np.int64)
    if not np.array_equal(values, cast):
        raise ValueError("Batch IDs must be exactly representable as int64.")
    return cast


class TrackStarBuilder:  # pylint: disable=too-many-instance-attributes
    """Fit projected train/evaluation curvature and build a frozen index."""

    def __init__(
        self,
        model: BaseInfluenceModel,
        projection_plan: TrackStarProjectionPlan,
        optimizer_second_moments: OptimizerSecondMoments,
        *,
        optimizer_epsilon: float,
        mix: Union[MixRule, float] = AutoMix(),
        rcond: float = 1e-12,
        negative_tolerance: float = 1e-10,
    ) -> None:
        if projection_plan.parameter_shapes != model.parameter_layout.parameter_shapes:
            raise ValueError("Projection plan does not match the model parameter layout.")
        moments = optimizer_second_moments.parameter_second_moments
        if len(moments) != model.parameter_layout.num_parameters:
            raise ValueError("Optimizer moments do not match the model parameter layout.")
        for index, (moment, shape) in enumerate(
            zip(moments, model.parameter_layout.parameter_shapes)
        ):
            if moment.shape != shape:
                raise ValueError(f"Optimizer second moment {index} must have shape {shape}.")
        if not np.isfinite(optimizer_epsilon) or optimizer_epsilon < 0:
            raise ValueError("optimizer_epsilon must be finite and non-negative.")

        self.model = model
        self.projection_plan = projection_plan
        self.optimizer_second_moments = optimizer_second_moments
        self.optimizer_epsilon = float(optimizer_epsilon)
        self.mix = mix
        self.rcond = float(rcond)
        self.negative_tolerance = float(negative_tolerance)
        self._state = "new"
        self._train_gram: Optional[GramSnapshot] = None
        self._evaluation_gram: Optional[GramSnapshot] = None
        self._curvature: Optional[ProjectedCurvature] = None
        self._projected_train_store: Optional[RepresentationStore] = None

    @classmethod
    def from_optimizer(
        cls,
        model: BaseInfluenceModel,
        optimizer: Any,
        projection_plan: TrackStarProjectionPlan,
        *,
        optimizer_epsilon: float,
        mix: Union[MixRule, float] = AutoMix(),
        rcond: float = 1e-12,
        negative_tolerance: float = 1e-10,
    ) -> "TrackStarBuilder":
        """Create a builder from a strict snapshot of live Adam/AdamW state."""
        return cls(
            model,
            projection_plan,
            extract_optimizer_second_moments(model, optimizer),
            optimizer_epsilon=optimizer_epsilon,
            mix=mix,
            rcond=rcond,
            negative_tolerance=negative_tolerance,
        )

    @property
    def state(self) -> str:
        """Return ``new``, ``fitted``, or ``built``."""
        return self._state

    @property
    def train_gram(self) -> GramSnapshot:
        """Return the fitted training Gram snapshot."""
        if self._train_gram is None:
            raise RuntimeError("Call fit() before accessing train_gram.")
        return self._train_gram

    @property
    def evaluation_gram(self) -> GramSnapshot:
        """Return the fitted evaluation Gram snapshot."""
        if self._evaluation_gram is None:
            raise RuntimeError("Call fit() before accessing evaluation_gram.")
        return self._evaluation_gram

    @property
    def curvature(self) -> ProjectedCurvature:
        """Return the fitted projected curvature."""
        if self._curvature is None:
            raise RuntimeError("Call fit() before accessing curvature.")
        return self._curvature

    @property
    def projected_train_store(self) -> RepresentationStore:
        """Return cached corrected and projected training rows."""
        if self._projected_train_store is None:
            raise RuntimeError("Call fit() before accessing projected_train_store.")
        return self._projected_train_store

    def _project_batch(self, batch: Batch) -> np.ndarray:
        components = self.model.batch_jacobian_components_tensor(batch)
        gradients = tuple(self.model.backend.to_numpy(component) for component in components)
        return self.projection_plan.apply(
            gradients,
            self.optimizer_second_moments,
            optimizer_epsilon=self.optimizer_epsilon,
        )

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(self.model.backend.to_numpy(value))

    def fit(
        self,
        train_dataset: Iterable[Batch],
        evaluation_curvature_dataset: Iterable[Batch],
        *,
        train_id_extractor: Optional[BatchIdExtractor] = None,
        train_payload_extractor: Optional[TrainingPayloadExtractor] = default_training_payload_extractor,
        projected_train_store_factory: Optional[StoreFactory] = None,
    ) -> "TrackStarBuilder":
        """Consume each source once and fit mixed projected-gradient curvature."""
        if self._state != "new":
            raise RuntimeError("fit() may only be called once on a new TrackStarBuilder.")
        train_accumulator = ProjectedGramAccumulator(self.projection_plan.block_slices)
        evaluation_accumulator = ProjectedGramAccumulator(self.projection_plan.block_slices)
        projected_store = None
        next_id = 0

        for batch in train_dataset:
            projected = self._project_batch(batch)
            if projected.shape[0] == 0:
                continue
            train_accumulator.update(projected)
            ids = _batch_ids(batch, projected.shape[0], next_id, train_id_extractor)
            payload = None
            if train_payload_extractor is not None:
                payload = self._to_numpy(train_payload_extractor(batch))
                if payload.shape[:1] != (projected.shape[0],):
                    raise ValueError("Training payload must have the same batch size as projected rows.")
            if projected_store is None:
                projected_store = _create_store(
                    projected_train_store_factory,
                    self.projection_plan.output_size,
                    vector_dtype=projected.dtype,
                    payload=payload,
                    stage="trackstar_projected",
                )
            projected_store.append(ids, projected, payload)
            next_id += projected.shape[0]

        if projected_store is None or train_accumulator.count == 0:
            raise ValueError("Training dataset must contain at least one sample.")
        projected_store.finalize()

        for batch in evaluation_curvature_dataset:
            projected = self._project_batch(batch)
            if projected.shape[0]:
                evaluation_accumulator.update(projected)
        if evaluation_accumulator.count == 0:
            raise ValueError("Evaluation curvature dataset must contain at least one sample.")

        self._train_gram = train_accumulator.snapshot()
        self._evaluation_gram = evaluation_accumulator.snapshot()
        self._curvature = build_projected_curvature(
            self._train_gram,
            self._evaluation_gram,
            self.mix,
            rcond=self.rcond,
            negative_tolerance=self.negative_tolerance,
        )
        self._projected_train_store = projected_store
        self._state = "fitted"
        return self

    def build(
        self,
        *,
        representation_store_factory: Optional[StoreFactory] = None,
        max_score_block_elements: int = 1_000_000,
    ) -> "TrackStarInfluenceCalculator":
        """Condition cached training rows and create an immutable exact index."""
        if self._state != "fitted":
            raise RuntimeError("build() requires a fitted, not-yet-built TrackStarBuilder.")
        source = self.projected_train_store
        destination = None
        for shard in source.iter_shards():
            vectors = self.curvature.apply(shard.vectors, normalize=True)
            if destination is None:
                destination = _create_store(
                    representation_store_factory,
                    source.vector_dim,
                    vector_dtype=vectors.dtype,
                    payload=shard.payload,
                    stage="trackstar_representations",
                )
            destination.append(shard.ids, vectors, shard.payload)
        assert destination is not None
        destination.finalize()
        index = ExactStreamingVectorIndex(
            destination, max_score_block_elements=max_score_block_elements
        )
        self._state = "built"
        return TrackStarInfluenceCalculator(
            self.model,
            self.projection_plan,
            self.optimizer_second_moments,
            self.optimizer_epsilon,
            self.curvature,
            destination,
            index,
        )


class TrackStarInfluenceCalculator:
    """Frozen TrackStar query transform and exact training-vector index."""

    def __init__(
        self,
        model: BaseInfluenceModel,
        projection_plan: TrackStarProjectionPlan,
        optimizer_second_moments: OptimizerSecondMoments,
        optimizer_epsilon: float,
        curvature: ProjectedCurvature,
        train_store: RepresentationStore,
        index: ExactStreamingVectorIndex,
    ) -> None:
        self.model = model
        self.projection_plan = projection_plan
        self.optimizer_second_moments = optimizer_second_moments
        self.optimizer_epsilon = optimizer_epsilon
        self.curvature = curvature
        self.train_store = train_store
        self.index = index

    def represent_batch(self, batch: Batch, *, normalize: bool = True) -> np.ndarray:
        """Return optimizer-corrected, projected, curvature-conditioned gradients."""
        components = self.model.batch_jacobian_components_tensor(batch)
        gradients = tuple(self.model.backend.to_numpy(component) for component in components)
        projected = self.projection_plan.apply(
            gradients,
            self.optimizer_second_moments,
            optimizer_epsilon=self.optimizer_epsilon,
        )
        return self.curvature.apply(projected, normalize=normalize)

    def search_batch(
        self,
        batch: Batch,
        k: int,
        *,
        order: str = "descending",
        return_payload: bool = True,
    ) -> VectorSearchResult:
        """Search exact inner-product neighbors for one query batch."""
        return self.index.search(
            self.represent_batch(batch),
            k,
            order=order,
            return_payload=return_payload,
        )

    def top_k(
        self,
        dataset_to_evaluate: Iterable[Batch],
        k: int,
        *,
        query_id_extractor: Optional[BatchIdExtractor] = None,
        order: str = "descending",
        return_payload: bool = True,
    ) -> Iterator[TrackStarTopKBatch]:
        """Yield exact neighbors while consuming the query iterable once."""
        next_id = 0
        for batch in dataset_to_evaluate:
            representations = self.represent_batch(batch)
            query_ids = _batch_ids(
                batch, representations.shape[0], next_id, query_id_extractor
            )
            neighbors = self.index.search(
                representations, k, order=order, return_payload=return_payload
            )
            yield TrackStarTopKBatch(query_ids, batch, neighbors)
            next_id += representations.shape[0]

    def self_influence_batch(self, batch: Batch) -> np.ndarray:
        """Return squared norms of final normalized TrackStar representations."""
        representations = self.represent_batch(batch)
        return np.sum(representations * representations, axis=1, keepdims=True)


__all__ = ["TrackStarBuilder", "TrackStarInfluenceCalculator", "TrackStarTopKBatch"]
