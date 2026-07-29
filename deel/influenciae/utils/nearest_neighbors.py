# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a Nearest Neighbors interface and a Linear Nearest Neighbors
algorithm. It will prove itself useful for finding the top-k most influential
examples of datasets, as implemented in the influence calculator interface.

This module is backend-agnostic and supports both TensorFlow and PyTorch.

The optional :class:`FaissNearestNeighbors` backend accelerates approximate
nearest-neighbor search over large gradient indexes used by the TrackStar
attribution pipeline. It becomes available when ``influenciae[faiss]`` is
installed.
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union
from warnings import warn

import numpy as np

from .._optional_imports import import_optional_module, is_module_available
from .sorted_dict import BatchSort, ORDER
from ..common.backend import (
    BaseBackend,
    Framework,
    get_backend,
)


def _ensure_reiterable_dataset(dataset: Any, context: str = "dataset") -> Any:
    """Materialize one-pass iterators to preserve multi-pass behavior."""
    try:
        if iter(dataset) is dataset:
            warn(
                f"{context} is a one-pass iterator; materializing it to preserve multi-pass behavior.",
                RuntimeWarning,
                stacklevel=2,
            )
            return list(dataset)
    except TypeError:
        pass
    return dataset


class BaseNearestNeighbors:
    """
    A Nearest Neighbors interface for efficiently searching for specific data-points
    in a dataset. This class is backend-agnostic and supports both TensorFlow and PyTorch.
    """

    @abstractmethod
    def build(
        self,
        dataset: Any,
        dot_product_fun: Callable[[Any, Any], Any],
        k: int,
        query_batch_size: int,
        d_type: Optional[Any] = None,
        payload_dtype: Optional[Any] = None,
        order: ORDER = ORDER.DESCENDING
    ) -> None:
        """
        Builds the nearest neighbors object that will be used to find the k neighbors
        inside the provided dataset.

        Parameters
        ----------
        dataset
            A dataset containing the points which shall be indexed.
            (tf.data.Dataset for TensorFlow, DataLoader or list of tuples for PyTorch)
        dot_product_fun
            The dot product function used to compute the distance between 2 points
        k
            An integer for the amount of samples to search for
        query_batch_size
            An integer for the query's batch size
        d_type
            The dataset's element's data-type. If None, will be inferred.
        payload_dtype
            Optional dtype used for stored payloads when it differs from the score dtype.
        order
            Either descending or ascending for the top or bottom results as per the similarity metric
        """
        raise NotImplementedError()

    @abstractmethod
    def query(self, vector_to_find: Any, batch_size: Optional[int] = None) -> Tuple[Any, Any]:
        """
        Find the k closest points to the provided vector in the object's dataset.

        Parameters
        ----------
        vector_to_find
            A tensor of points to query.
        batch_size
            An integer with the query's batch size

        Returns
        -------
        values_and_samples
            A tuple with the k closest points to the queries and their corresponding distances in the format
            (distances, points)
        """
        raise NotImplementedError()


class LinearNearestNeighbors(BaseNearestNeighbors):
    """
    An implementation of a Linear Nearest Neighbors search algorithm using a given similarity metric and
    doing as much lazy computations as possible for scalability.

    This class is backend-agnostic and supports both TensorFlow and PyTorch.

    Attributes
    ----------
    dataset
        A dataset with the points from which the nearest neighbors will be searched.
    dot_product_fun
        A callable taking in two vectors and returning a notion of similarity between them.
    batched_sorted_dict
        A BatchSort instance that takes care of keeping the top/bottom most similar examples from
        the batch.
    backend
        The backend to use for tensor operations. If None, it will be inferred from the first batch
        or default to TensorFlow if available.
    """

    def __init__(self, backend: Optional[Union[BaseBackend, Framework]] = None):
        """
        Initialize the LinearNearestNeighbors.

        Parameters
        ----------
        backend
            The backend to use for tensor operations. Can be a BaseBackend instance,
            a Framework enum (TENSORFLOW or PYTORCH), or None to infer from data.
        """
        self.dataset: Optional[Any] = None
        self.dot_product_fun: Optional[Callable[[Any, Any], Any]] = None
        self.batched_sorted_dict: Optional[BatchSort] = None
        self._backend_param = backend
        self._backend: Optional[BaseBackend] = None

    @property
    def backend(self) -> BaseBackend:
        """Return the backend used for tensor operations."""
        if self._backend is None:
            # Determine the backend
            if self._backend_param is None:
                # Default to TensorFlow if available, otherwise PyTorch
                try:
                    self._backend = get_backend(Framework.TENSORFLOW)
                except ImportError:
                    self._backend = get_backend(Framework.PYTORCH)
            elif isinstance(self._backend_param, BaseBackend):
                self._backend = self._backend_param
            else:
                # backend_param is a Framework enum
                self._backend = get_backend(self._backend_param)
        return self._backend

    @staticmethod
    def _get_first_element(value: Any) -> Any:
        """Return the first element for tuple/list containers."""
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    def _infer_batch_shape_from_spec(self, element_spec: Any) -> Tuple[int, ...]:
        """Infer batch sample shape from a dataset element specification."""
        first_spec = self._get_first_element(element_spec)
        first_spec = self._get_first_element(first_spec)

        if hasattr(first_spec, "shape"):
            return tuple(first_spec.shape[1:])
        if isinstance(first_spec, dict) and "shape" in first_spec:
            return tuple(first_spec["shape"][1:])
        return ()

    def _infer_batch_dtype_from_spec(self, element_spec: Any) -> Optional[Any]:
        """Infer payload dtype from a dataset element specification."""
        first_spec = self._get_first_element(element_spec)
        first_spec = self._get_first_element(first_spec)
        if hasattr(first_spec, "dtype"):
            return first_spec.dtype
        if isinstance(first_spec, dict) and "dtype" in first_spec:
            return first_spec["dtype"]
        return None

    def _infer_batch_shape_from_dataset(self, dataset: Any) -> Tuple[int, ...]:
        """Infer batch sample shape by peeking at the first dataset element."""
        for item in dataset:
            first_tensor = self._get_first_element(item)
            first_tensor = self._get_first_element(first_tensor)
            return tuple(self.backend.tensor_shape(first_tensor)[1:])
        return ()

    def _infer_batch_dtype_from_dataset(self, dataset: Any) -> Optional[Any]:
        """Infer payload dtype by peeking at the first dataset element."""
        for item in dataset:
            first_tensor = self._get_first_element(item)
            first_tensor = self._get_first_element(first_tensor)
            return getattr(first_tensor, "dtype", None)
        return None

    def _extract_batch_samples_and_ihvp(self, batch_data: Any) -> Tuple[Any, Any]:
        """Extract sample tensors and IHVP values from a dataset batch entry."""
        if isinstance(batch_data, tuple):
            batch_seq = list(batch_data)
        elif isinstance(batch_data, list):
            batch_seq = batch_data
        else:
            batch_seq = None

        if batch_seq is not None:
            if len(batch_seq) >= 2:
                batch, ihvp = batch_seq[0], batch_seq[-1]
            elif len(batch_seq) == 1:
                batch, ihvp = batch_seq[0], batch_seq[0]
            else:
                raise ValueError("Encountered empty batch data while querying nearest neighbors.")
        else:
            batch, ihvp = batch_data, batch_data

        batch_samples = self._get_first_element(batch)
        return batch_samples, ihvp

    def build(
        self,
        dataset: Any,
        dot_product_fun: Callable[[Any, Any], Any],
        k: int,
        query_batch_size: int,
        d_type: Optional[Any] = None,
        payload_dtype: Optional[Any] = None,
        order: ORDER = ORDER.DESCENDING
        ) -> None:
        """
        Builds the linear nearest neighbors object that will be used to find the k neighbors
        inside the provided dataset.

        Parameters
        ----------
        dataset
            A dataset containing the points which shall be indexed.
            (tf.data.Dataset for TensorFlow, DataLoader or list of tuples for PyTorch)
        dot_product_fun
            The dot product function used to compute the distance between 2 points
        k
            An integer for the amount of samples to search for
        query_batch_size
            An integer for the query's batch size
        d_type
            The dataset's element's data-type. If None, will be inferred from the backend's default.
        payload_dtype
            Optional dtype used for stored payloads when it differs from the score dtype.
        order
            Either descending or ascending for the top or bottom results as per the similarity metric
        """
        self.dataset = _ensure_reiterable_dataset(dataset, context="nearest-neighbor dataset")
        self.dot_product_fun = dot_product_fun

        element_spec = self.backend.get_dataset_element_spec(self.dataset)
        batch_shape = self._infer_batch_shape_from_spec(element_spec)
        inferred_payload_dtype = self._infer_batch_dtype_from_spec(element_spec)

        if not batch_shape:
            if self.dataset is None:
                raise ValueError("Nearest neighbors dataset is not initialized.")
            batch_shape = self._infer_batch_shape_from_dataset(self.dataset)

        if inferred_payload_dtype is None:
            inferred_payload_dtype = self._infer_batch_dtype_from_dataset(self.dataset)

        # Use backend default dtype if not provided
        if d_type is None:
            d_type = self.backend.float32_dtype()
        if payload_dtype is None:
            payload_dtype = inferred_payload_dtype if inferred_payload_dtype is not None else d_type

        self.batched_sorted_dict = BatchSort(
            batch_shape,
            (query_batch_size, k),
            dtype=d_type,
            batch_dtype=payload_dtype,
            value_dtype=d_type,
            order=order,
            backend=self.backend
        )

    def _require_built(self) -> Tuple[Any, Callable[[Any, Any], Any], BatchSort]:
        """Ensure the index is initialized before queries."""
        if self.dataset is None or self.dot_product_fun is None or self.batched_sorted_dict is None:
            raise ValueError("Nearest neighbors index is not built. Call 'build(...)' before querying.")
        return self.dataset, self.dot_product_fun, self.batched_sorted_dict

    def query(self, vector_to_find: Any, batch_size: Optional[int] = None) -> Tuple[Any, Any]:
        """
        Find the k closest points to the provided vector in the object's dataset.

        Parameters
        ----------
        vector_to_find
            A tensor of points to query.
        batch_size
            An integer with the query's batch size

        Returns
        -------
        values_and_samples
            A tuple with the k closest points to the queries and their corresponding distances in the format
            (distances, points)
        """
        if batch_size is None:
            batch_size = self.backend.get_batch_size(vector_to_find)

        if self.backend.framework == Framework.TENSORFLOW:
            return self._query_tensorflow(vector_to_find, batch_size)
        return self._query_pytorch(vector_to_find, batch_size)

    def _query_tensorflow(self, vector_to_find: Any, batch_size: int) -> Tuple[Any, Any]:
        """TensorFlow-specific query using reduce for lazy evaluation within graph."""
        tf = import_optional_module("tensorflow", extra="tensorflow")
        dataset, dot_product_fun, batched_sorted_dict = self._require_built()

        k = batched_sorted_dict.k
        order = batched_sorted_dict.order
        batch_shape = tuple(batched_sorted_dict.shape[2:])  # Remove (1, k) prefix

        # Initialize state tensors
        if order == ORDER.DESCENDING:
            init_values = tf.fill((batch_size, k), float('-inf'))
        else:
            init_values = tf.fill((batch_size, k), float('inf'))

        init_samples = tf.zeros((batch_size, k) + batch_shape, dtype=batched_sorted_dict.batch_dtype)

        # Cast to the appropriate dtype
        init_values = tf.cast(init_values, batched_sorted_dict.value_dtype)

        def reduce_func(state, batch_data):
            best_values, best_samples = state
            batch_samples, ihvp = self._extract_batch_samples_and_ihvp(batch_data)

            # Compute influence values
            influence_values = dot_product_fun(vector_to_find, ihvp)

            # Expand batch_samples to match query batch size
            expanded_batch = tf.repeat(
                tf.expand_dims(batch_samples, axis=0),
                batch_size,
                axis=0
            )

            # Concatenate with current best
            current_score = tf.concat([best_values, influence_values], axis=1)
            current_batch = tf.concat([best_samples, expanded_batch], axis=1)

            # Sort and take top k
            descending = order == ORDER.DESCENDING
            if descending:
                indexes = tf.argsort(current_score, axis=1, direction='DESCENDING')
            else:
                indexes = tf.argsort(current_score, axis=1, direction='ASCENDING')
            indexes = indexes[:, :k]

            # Gather top k values and samples
            new_best_values = tf.gather(current_score, indexes, axis=1, batch_dims=1)
            new_best_samples = tf.gather(current_batch, indexes, axis=1, batch_dims=1)

            return (new_best_values, new_best_samples)

        if hasattr(dataset, "reduce"):
            final_values, final_samples = dataset.reduce(
                (init_values, init_samples),
                reduce_func
            )
        else:
            state = (init_values, init_samples)
            for batch_data in dataset:
                state = reduce_func(state, batch_data)
            final_values, final_samples = state

        return final_values, final_samples

    def _query_pytorch(self, vector_to_find: Any, batch_size: int) -> Tuple[Any, Any]:
        """PyTorch-specific query using BatchSort."""
        dataset, dot_product_fun, batched_sorted_dict = self._require_built()
        batched_sorted_dict.reset()

        # Iterate through the dataset
        for batch_data in dataset:
            batch_samples, ihvp = self._extract_batch_samples_and_ihvp(batch_data)

            # Compute influence values using the dot product function
            influence_values = dot_product_fun(vector_to_find, ihvp)

            # Expand batch_samples to match query batch size and add to sorted dict
            expanded_batch = self.backend.repeat(
                self.backend.expand_dims(batch_samples, axis=0),
                batch_size,
                axis=0
            )

            batched_sorted_dict.add_all(expanded_batch, influence_values)

        training_samples, influences_values = batched_sorted_dict.get()

        return influences_values, training_samples


# ---------------------------------------------------------------------------
# Optional FAISS-backed nearest neighbors
# ---------------------------------------------------------------------------


@dataclass
class FaissConfig:
    """Configuration for the FAISS-backed nearest-neighbor index.

    Parameters
    ----------
    factory_string
        FAISS index factory string. Defaults to a flat inner-product index,
        which mirrors ``LinearNearestNeighbors`` semantics but relies on
        FAISS' vectorized implementation.
    metric
        Similarity metric. Either ``"inner_product"`` (default) or ``"l2"``.
    use_gpu
        Attempt to move the FAISS index to GPU when ``faiss-gpu`` is available.
        Silently falls back to CPU otherwise.
    normalize_queries
        When ``True``, L2-normalize both index vectors and queries before
        search. Combined with ``metric="inner_product"`` this yields cosine
        similarity, which matches TrackStar's normalized-gradient scoring.
    train_size
        Optional maximum number of vectors used to train the index when the
        factory string requires it (e.g. ``"IVF"``).
    """

    factory_string: str = "Flat"
    metric: str = "inner_product"
    use_gpu: bool = False
    normalize_queries: bool = False
    train_size: Optional[int] = None


def is_faiss_available() -> bool:
    """Return whether the optional FAISS dependency is installed."""
    return is_module_available("faiss")


class FaissNearestNeighbors(BaseNearestNeighbors):
    """FAISS-accelerated nearest-neighbor index over stacked gradient vectors.

    Unlike :class:`LinearNearestNeighbors`, this backend operates on a materialized
    matrix of index vectors and returns their integer indices. It is designed for
    the TrackStar scoring loop, where per-sample gradients (optionally
    randomly projected) are already available as a dense NumPy matrix.

    Instantiation lazily imports FAISS through
    :func:`~deel.influenciae._optional_imports.import_optional_module` so
    influenciae installs without the extra remain fully functional. Callers who
    want a graceful fallback can use :func:`is_faiss_available` before
    constructing this class.

    Parameters
    ----------
    config
        Optional configuration. Defaults to a flat inner-product index.
    """

    def __init__(self, config: Optional[FaissConfig] = None) -> None:
        self._config = config or FaissConfig()
        if self._config.metric not in ("inner_product", "l2"):
            raise ValueError(
                "FaissConfig.metric must be one of {'inner_product', 'l2'}, "
                f"got {self._config.metric!r}."
            )

        # Lazy-load faiss so importing this module does not require the extra.
        self._faiss = import_optional_module("faiss", extra="faiss")
        self._index: Any = None
        self._payload_indices: Optional[np.ndarray] = None
        self._k: Optional[int] = None
        self._order: ORDER = ORDER.DESCENDING
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # BaseNearestNeighbors implementation
    # ------------------------------------------------------------------

    def build(  # type: ignore[override]
        self,
        dataset: Any,
        dot_product_fun: Optional[Callable[[Any, Any], Any]] = None,
        k: int = 10,
        query_batch_size: Optional[int] = None,
        d_type: Optional[Any] = None,
        payload_dtype: Optional[Any] = None,
        order: ORDER = ORDER.DESCENDING,
    ) -> None:
        """Build the FAISS index from a stacked gradient matrix.

        Parameters
        ----------
        dataset
            Either a 2-D NumPy matrix of shape ``(N, D)`` or a tuple
            ``(indices, gradients)`` where ``gradients`` has shape ``(N, D)``
            and ``indices`` has shape ``(N,)``.
        dot_product_fun
            Ignored; retained for API parity with :class:`BaseNearestNeighbors`.
        k
            Number of neighbors to return on each query.
        query_batch_size
            Ignored; retained for API parity.
        d_type
            Ignored; FAISS uses ``float32`` internally.
        payload_dtype
            Ignored; payloads are always returned as ``int64`` indices.
        order
            Sorting order. Only ``ORDER.DESCENDING`` is meaningful for inner
            product similarity; ``ORDER.ASCENDING`` is honored for L2 metric.
        """
        del dot_product_fun, query_batch_size, d_type, payload_dtype  # unused

        indices, gradients = self._unpack_dataset(dataset)
        if gradients.ndim != 2:
            raise ValueError(
                "FaissNearestNeighbors expects 2-D gradient matrices, got shape "
                f"{tuple(gradients.shape)}."
            )
        if gradients.shape[0] == 0:
            raise ValueError("FaissNearestNeighbors received an empty gradient matrix.")

        gradients_f32 = np.ascontiguousarray(gradients, dtype=np.float32)
        if self._config.normalize_queries:
            gradients_f32 = self._l2_normalize(gradients_f32)

        self._dim = int(gradients_f32.shape[1])
        self._k = int(k)
        self._order = order
        self._payload_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self._payload_indices.shape[0] != gradients_f32.shape[0]:
            raise ValueError(
                "FaissNearestNeighbors payload indices and gradient rows have "
                f"mismatched lengths: {self._payload_indices.shape[0]} vs {gradients_f32.shape[0]}."
            )

        metric_flag = (
            self._faiss.METRIC_INNER_PRODUCT
            if self._config.metric == "inner_product"
            else self._faiss.METRIC_L2
        )
        index = self._faiss.index_factory(self._dim, self._config.factory_string, metric_flag)

        if not index.is_trained:
            training_vectors = gradients_f32
            if self._config.train_size is not None and self._config.train_size < training_vectors.shape[0]:
                rng = np.random.default_rng(0)
                sampled_indices = rng.choice(
                    training_vectors.shape[0],
                    size=int(self._config.train_size),
                    replace=False,
                )
                training_vectors = training_vectors[sampled_indices]
            index.train(training_vectors)

        index.add(gradients_f32)

        if self._config.use_gpu:
            try:
                res = self._faiss.StandardGpuResources()
                index = self._faiss.index_cpu_to_gpu(res, 0, index)
            except (AttributeError, RuntimeError) as exc:
                warn(
                    f"FAISS GPU support requested but unavailable ({exc}); "
                    "continuing with the CPU index.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._index = index

    def query(  # type: ignore[override]
        self,
        vector_to_find: Any,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, payload_indices)`` for each query row."""
        del batch_size  # FAISS handles batching internally

        if self._index is None or self._payload_indices is None or self._k is None:
            raise ValueError(
                "FaissNearestNeighbors index is not built. Call `build(...)` before querying."
            )

        query_matrix = np.asarray(vector_to_find, dtype=np.float32)
        if query_matrix.ndim == 1:
            query_matrix = query_matrix[np.newaxis, :]
        if query_matrix.ndim != 2:
            raise ValueError(
                "FaissNearestNeighbors.query expects 1-D or 2-D queries, got shape "
                f"{tuple(query_matrix.shape)}."
            )
        if query_matrix.shape[1] != self._dim:
            raise ValueError(
                "FaissNearestNeighbors query dimension mismatch: "
                f"expected {self._dim}, got {query_matrix.shape[1]}."
            )
        if self._config.normalize_queries:
            query_matrix = self._l2_normalize(query_matrix)

        query_matrix = np.ascontiguousarray(query_matrix)
        scores, positions = self._index.search(query_matrix, self._k)
        payload = self._payload_indices[positions]

        if self._config.metric == "l2" and self._order == ORDER.DESCENDING:
            # FAISS returns ascending L2 distances by default. Flip to match
            # descending-order calling conventions from BatchSort.
            scores = scores[:, ::-1]
            payload = payload[:, ::-1]

        return scores, payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_dataset(dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, gradients)`` from the accepted dataset formats."""
        if isinstance(dataset, tuple) and len(dataset) == 2:
            indices, gradients = dataset
        elif isinstance(dataset, np.ndarray):
            gradients = dataset
            indices = np.arange(gradients.shape[0], dtype=np.int64)
        else:
            raise TypeError(
                "FaissNearestNeighbors.build accepts either a 2-D numpy array or a "
                "(indices, gradients) tuple; got "
                f"{type(dataset).__name__}."
            )
        return np.asarray(indices), np.asarray(gradients)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Return an L2-row-normalized copy of *matrix*, guarding against zero rows."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero for zero-gradient rows; they normalize to 0.
        safe_norms = np.where(norms > 0, norms, 1.0)
        return matrix / safe_norms
