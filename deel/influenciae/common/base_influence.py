# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module defining the interface for all the different influence calculator classes.

The BaseInfluenceCalculator interface provides implementations to common to all of
their child classes (i.e. all the different techniques to compute a notion of influence).

It provides optimized implementations for some methods following the assumption that
the computation can be written as a matrix-vector product with a matrix that can be
(pre)-computed and remains unchanged throughout the computation.

Evaluation representation hook
------------------------------
The :meth:`BaseInfluenceCalculator._get_evaluation_representation` method acts as
a dispatch point that determines how an evaluation (query) batch is converted into
the tensor representation used for scoring.  By default it falls through to the
subclass-specific :meth:`_preprocess_samples`; when an
:class:`~.evaluation.EvaluationRepresentationProvider` is supplied, it delegates
to the provider instead.  This allows structured tasks -- such as object detection
-- to inject a custom per-sample objective without modifying the core influence
math.
"""
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Tuple
from warnings import warn

from .backend import BaseBackend
from .evaluation import EvaluationRepresentationProvider
from .payloads import TrainingPayloadExtractor, default_training_payload_extractor
from .query_batching import PreconditioningMode, QueryBatchingConfig
from ..utils.nearest_neighbors import LinearNearestNeighbors
from ..utils.sorted_dict import BatchSort, ORDER
from ..types import DType, DatasetLike, Tensor

if TYPE_CHECKING:
    from ..utils.nearest_neighbors import BaseNearestNeighbors


class CACHE(Enum):
    """
    Class for the options of where to cache intermediary results for optimizing computations.
    """
    MEMORY = 0
    DISK = 1
    NO_CACHE = 2


class SelfInfluenceCalculator:
    """
    A basic interface for influence calculators whose influence score computation can't be
    decomposed into an inner product between an "influence vector" reflecting the influence
    of the training points and another vector related to a test point. In particular, it will
    be used for RepresenterPointL2 and [WIP] the techniques based on adversarial attacks.

    Attributes
    ----------
    backend
        The framework-specific backend for operations.
    """

    # Backend is set by subclasses that have access to a model.
    backend: BaseBackend

    @property
    def _backend(self) -> BaseBackend:
        """Return initialized backend instance."""
        if self.backend is None:
            raise ValueError("Backend is not initialized. Instantiate a calculator with a valid model first.")
        return self.backend

    @abstractmethod
    def _compute_influence_value_from_batch(self, train_samples: Tuple[Tensor, ...]) -> Tensor:
        """
        Computes the influence score (self-influence) for a single batch of training samples.

        Parameters
        ----------
        train_samples
            A tensor with a single batch of training sample.

        Returns
        -------
        influence_values
            The influence score of each sample in the batch train_samples.
        """
        raise NotImplementedError()

    def compute_influence_values(self, train_set: DatasetLike, device: Optional[str] = None) -> DatasetLike:
        """
        Compute the influence score for each sample of the provided (full or partial) model's training dataset.

        If only looking for the values, use `_compute_influence_values`,
        which returns a tensor directly.

        Parameters
        ----------
        train_set
            A dataset with the (full or partial) model's training dataset.
        device
            Device where the computation will be executed

        Returns
        -------
        train_set
            A dataset containing the tuple: (batch of training samples, influence score)
        """
        return self._backend.map_dataset(
            train_set,
            lambda *batch_data: (batch_data, self._compute_influence_value_from_batch(batch_data)),
            device
        )

    def _compute_influence_values(self, train_set: DatasetLike, device: Optional[str] = None) -> Optional[Tensor]:
        """
        Compute the influence score for each sample of the provided (full or partial) model's training dataset.
        This version returns a tensor instead of a dataset.
        For internal use only.

        Parameters
        ----------
        train_set
            A dataset with the (full or partial) model's training dataset.
        Returns
        -------
        influence score
            A tensor with the sample's influence scores.
        """
        influences_values = self.compute_influence_values(train_set, device)

        # Extract just the influence values
        inf_val_list = []
        for item in influences_values:
            if isinstance(item, (list, tuple)):
                # item is (batch_data, inf_val)
                inf_val_list.append(item[-1])
            else:
                inf_val_list.append(item)

        if inf_val_list:
            return self._backend.concat(inf_val_list, axis=0)
        return None

    def compute_top_k_from_training_dataset(
            self,
            train_set: DatasetLike,
            k: int,
            order: ORDER = ORDER.DESCENDING
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the k most influential data-points of the model's training dataset by computing
        Cook's distance for each point individually.

        Parameters
        ----------
        train_set
            A dataset containing the points on which the model was trained.
        k
            An integer with the number of most important samples we wish to keep
        order
            Either ORDER.DESCENDING or ORDER.ASCENDING depending on if we wish to find the top-k or
            bottom-k samples, respectively.

        Returns
        -------
        training_samples, influences_values
            A tuple of tensor.
            - training_samples: A tensor containing the k most influential samples of the training dataset for the model
            provided.
            - influences_values: The influence score corresponding to these k most influential samples.
        """
        self._backend.assert_batched_dataset(train_set)

        # Get element spec for BatchSort initialization
        elt_spec = self._backend.get_dataset_element_spec(train_set)
        if isinstance(elt_spec, (list, tuple)):
            first_spec = elt_spec[0]
        else:
            first_spec = elt_spec

        # Get shape and dtype from spec
        shape = None
        dtype = None
        if hasattr(first_spec, 'shape'):
            shape = first_spec.shape[1:]  # Remove batch dimension
            dtype = first_spec.dtype
        elif isinstance(first_spec, dict):
            shape = first_spec['shape'][1:]
            dtype = first_spec['dtype']

        # Fallback: get from first batch if not available from spec
        if shape is None:
            for batch in train_set:
                first_tensor = batch[0] if isinstance(batch, (list, tuple)) else batch
                shape = self._backend.tensor_shape(first_tensor)[1:]
                dtype = first_tensor.dtype
                break

        if shape is None:
            raise ValueError("Could not determine tensor shape from dataset")

        batch_sorted_dict = BatchSort(shape, (1, k), dtype=dtype, order=order)

        for batch in train_set:
            influence_values = self._compute_influence_value_from_batch(batch)
            if self._backend.tensor_ndim(influence_values) == 1:
                influence_values = self._backend.expand_dims(influence_values, axis=-1)

            batch_input = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_sorted_dict.add_all(
                self._backend.expand_dims(batch_input, axis=0),
                self._backend.transpose(influence_values)
            )

        best_samples, best_values = batch_sorted_dict.get()
        # best_values is already a tensor of shape (1, k), just squeeze the first dimension
        influence_values = self._backend.squeeze(best_values, axis=0)
        # best_samples is of shape (1, k, ...), squeeze the first dimension
        training_samples = self._backend.squeeze(best_samples, axis=0)

        return training_samples, influence_values

    def _save_dataset(self, dataset: DatasetLike, load_or_save_path: str) -> None:
        """
        Save a dataset in the appropriate format for the backend.

        Parameters
        ----------
        dataset
            The dataset to save
        load_or_save_path
            The path to save the dataset
        """
        self._backend.save_dataset(dataset, load_or_save_path)

    def _load_dataset(self, dataset_path: str) -> DatasetLike:
        """
        Loads a dataset from the specified path.

        Parameters
        ----------
        dataset_path
            The path pointing to the file from which to load the dataset

        Returns
        -------
        dataset
            The target dataset
        """
        return self._backend.load_dataset(dataset_path)


class BaseInfluenceCalculator(SelfInfluenceCalculator):
    """
    The base implementation of an interface for all the influence calculators in the library.
    All the methods included in deel-influenciae implement these basic functions, with
    each their own notion of influence of a data-point on the model.

    As the computation of the influence scores can be written as an inner product, we apply some
    optimizations that allow us to scale to large datasets.

    Please note that for some of the classes implementing this interface, the notion of influence
    vector does not necessarily have the meaning of being the delta of the weights of the model
    after the perturbation of the training dataset.
    """

    @abstractmethod
    def _preprocess_samples(self, samples: Tuple[Any, ...]) -> Any:
        """
        Preprocess one evaluation batch into the representation used for scoring.

        Parameters
        ----------
        samples
            A single batch of samples to evaluate.

        Returns
        -------
        preprocessed_samples
            Backend-specific representation consumed by
            :meth:`_estimate_influence_value_from_influence_vector`. For
            first-order calculators this is typically the per-sample Jacobian
            of the loss with respect to the watched weights.
        """
        raise NotImplementedError()

    @abstractmethod
    def _compute_influence_vector(self, train_samples: Tuple[Any, ...]) -> Any:
        """
        Computes the influence vector (i.e. the delta of model's weights after a perturbation on the training
        dataset) for a single batch of training samples.

        Parameters
        ----------
        train_samples
            A tuple with the batch of training samples (with their labels).

        Returns
        -------
        influence_vector
            A tensor with the influence vector for each individual point.
        """
        raise NotImplementedError()

    def compute_influence_vector(
            self,
            train_set: DatasetLike,
            save_influence_vector_ds_path: Optional[str] = None,
            device: Optional[str] = None
    ) -> DatasetLike:
        """
        Compute the influence vector for each sample of the provided (full or partial) model's training dataset.

        Parameters
        ----------
        train_set
            A dataset with the (full or partial) model's training dataset.
        save_influence_vector_ds_path
            The path to save or load the influence vector of the training dataset. If specified,
            load the dataset if it has already been computed, otherwise, compute the influence vector and
            then save it in the specified path.
        device
            Device where the computation will be executed

        Returns
        -------
        inf_vect_ds
            A dataset containing the tuple: (batch of training samples, influence vector)
        """
        inf_vect_ds = self._backend.map_dataset(
            train_set,
            lambda *batch: (batch, self._compute_influence_vector(batch)),
            device
        )

        if save_influence_vector_ds_path is not None:
            # Explicit cache boundary: we save and return the same computed dataset.
            inf_vect_ds = self._backend.cache_dataset(inf_vect_ds)
            inf_vect_only_ds = self._backend.map_dataset(
                inf_vect_ds,
                lambda *item: item[-1],
                device
            )
            unbatched_inf_vect = self._backend.unbatch_dataset(inf_vect_only_ds)
            self._save_dataset(unbatched_inf_vect, save_influence_vector_ds_path)

        return inf_vect_ds

    @staticmethod
    def _resolve_preconditioning_mode(preconditioning_mode: Optional[Any]) -> Any:
        """Resolve the default preconditioning mode lazily."""
        from .query_batching import PreconditioningMode

        if preconditioning_mode is None:
            return PreconditioningMode.TRAIN
        return preconditioning_mode

    def _get_evaluation_representation(
            self,
            samples: Tuple[Any, ...],
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
    ) -> Any:
        """
        Return the representation used to score an evaluation batch.

        Parameters
        ----------
        samples
            A single batch of samples to evaluate.
        evaluation_representation_provider
            Optional callable mapping ``(self.model, samples)`` to the
            representation used for scoring. When ``None``, falls back to
            ``_preprocess_samples``.

        Returns
        -------
        representation
            The tensor or structure used to score the evaluation batch.
        """
        if evaluation_representation_provider is None:
            return self._preprocess_samples(samples)
        return evaluation_representation_provider(self.model, samples)

    def _extract_training_payload(
            self,
            train_batch: Tuple[Any, ...],
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
    ) -> Tensor:
        """
        Extract the payload returned alongside influence scores.

        Parameters
        ----------
        train_batch
            A single batch from the training dataset.
        training_payload_extractor
            Optional callable used to extract the payload returned alongside
            influence scores. When ``None``,
            ``default_training_payload_extractor`` is used.

        Returns
        -------
        payload
            The extracted training payload.
        """
        extractor = default_training_payload_extractor if training_payload_extractor is None else training_payload_extractor
        return extractor(train_batch)

    def _estimate_influence_values_query_mode(
            self,
            dataset_to_evaluate: DatasetLike,
            train_set: DatasetLike,
            config: Optional[Any] = None,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None,
    ) -> DatasetLike:
        """
        Query-side adapter for batched influence-value computation.

        Parameters
        ----------
        dataset_to_evaluate
            Dataset containing the samples to score.
        train_set
            Dataset containing the training samples against which influence is
            computed.
        config
            Optional configuration controlling query-side batching and
            preconditioning.
        evaluation_representation_provider
            Optional callable mapping ``(self.model, batch)`` to the
            representation used when scoring evaluation batches. When
            ``None``, the default preprocessing path is used.
        training_payload_extractor
            Optional callable used to extract the payload returned alongside
            influence scores. When ``None``,
            ``default_training_payload_extractor`` is used.
        device
            Device where the computation will be executed.

        Raises
        ------
        NotImplementedError
            If the calculator does not implement query-side preconditioning.

        Returns
        -------
        influence_value_dataset
            A dataset-like object matching
            :meth:`estimate_influence_values_in_batches`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support query-side preconditioning."
        )

    def _top_k_query_mode(
            self,
            dataset_to_evaluate: DatasetLike,
            train_set: DatasetLike,
            k: int = 5,
            order: ORDER = ORDER.DESCENDING,
            d_type: Optional[DType] = None,
            payload_dtype: Optional[DType] = None,
            config: Optional[Any] = None,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None,
    ) -> DatasetLike:
        """
        Query-side adapter for batched top-k computation.

        Parameters
        ----------
        dataset_to_evaluate
            Dataset containing the samples to score.
        train_set
            Dataset containing the training samples from which to retrieve the
            most influential entries.
        k
            Number of most influential training samples to retain.
        order
            Either ``ORDER.DESCENDING`` or ``ORDER.ASCENDING`` depending on
            whether the most or least influential samples are requested.
        d_type
            Data-type of the influence scores. If ``None``, it is inferred.
        payload_dtype
            Data-type used to store extracted training payloads in the sorted
            structure. If ``None``, it is inferred.
        config
            Optional configuration controlling query-side batching and
            preconditioning.
        evaluation_representation_provider
            Optional callable mapping ``(self.model, batch)`` to the
            representation used when scoring evaluation batches. When
            ``None``, the default preprocessing path is used.
        training_payload_extractor
            Optional callable used to extract the payload returned alongside
            top-k influence scores. When ``None``,
            ``default_training_payload_extractor`` is used.
        device
            Device where the computation will be executed.

        Raises
        ------
        NotImplementedError
            If the calculator does not implement query-side preconditioning.

        Returns
        -------
        top_k_dataset
            A dataset-like object matching :meth:`top_k`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support query-side top-k computation."
        )

    def estimate_influence_values_in_batches(
            self,
            dataset_to_evaluate: DatasetLike,
            train_set: DatasetLike,
            influence_vector_in_cache: CACHE = CACHE.MEMORY,
            load_influence_vector_path: Optional[str] = None,
            save_influence_vector_path: Optional[str] = None,
            save_influence_value_path: Optional[str] = None,
            preconditioning_mode: Optional["PreconditioningMode"] = None,
            query_batching_config: Optional["QueryBatchingConfig"] = None,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None
    ) -> DatasetLike:
        """
        Estimates the influence that each point in the provided training dataset has on each of the test points.
        This can provide some insights as to what makes the model predict a certain way for the given test points,
        and thus presents data-centric explanations.

        Parameters
        ----------
        dataset_to_evaluate
            A dataset containing the test samples for which to compute the effect of removing each of the provided
            training points (individually).
        train_set
            A dataset containing the model's training dataset (partial or full).
        influence_vector_in_cache
            An enum indicating if intermediary values are to be cached (either in memory or on the disk) or not.
            Options include CACHE.MEMORY (0) for caching in memory, CACHE.DISK (1) for the disk and CACHE.NO_CACHE (2)
            for no optimization.
        load_influence_vector_path
            The path to load the influence vectors (if they have already been calculated).
        save_influence_vector_path
            The path to save the computed influence vector.
        save_influence_value_path
            The path to save the computed influence values.
        preconditioning_mode
            Whether to precondition training gradients (default) or query gradients.
        query_batching_config
            Optional configuration used only when
            ``preconditioning_mode=PreconditioningMode.QUERY``.
        evaluation_representation_provider
            Optional callable mapping ``(self.model, batch)`` to a custom
            representation used for scoring evaluation samples. When ``None``,
            the default ``_preprocess_samples`` path is used.
        training_payload_extractor
            Optional callable used to extract the payload returned alongside
            influence scores for each training batch. When ``None``,
            ``default_training_payload_extractor`` is used.
        device
            Device where the computation will be executed.

        Raises
        ------
        ValueError
            If query-side preconditioning is combined with train-side
            influence-vector caching arguments, or if
            ``query_batching_config`` is provided while
            ``preconditioning_mode`` is not
            ``PreconditioningMode.QUERY``.

        Returns
        -------
        influence_value_dataset
            A dataset containing the tuple: (samples_to_evaluate, dataset).

            - samples_to_evaluate: The batch of sample to evaluate.
            - dataset: Dataset containing tuples of batch of the training dataset and their influence score.
        """
        preconditioning_mode = self._resolve_preconditioning_mode(preconditioning_mode)
        if preconditioning_mode == PreconditioningMode.QUERY:
            if influence_vector_in_cache != CACHE.MEMORY:
                raise ValueError(
                    "influence_vector_in_cache is only supported with preconditioning_mode=TRAIN."
                )
            if load_influence_vector_path is not None:
                raise ValueError(
                    "load_influence_vector_path is only supported with preconditioning_mode=TRAIN."
                )
            if save_influence_vector_path is not None:
                raise ValueError(
                    "save_influence_vector_path is only supported with preconditioning_mode=TRAIN."
                )

            influence_value_dataset = self._estimate_influence_values_query_mode(
                dataset_to_evaluate,
                train_set,
                config=query_batching_config,
                evaluation_representation_provider=evaluation_representation_provider,
                training_payload_extractor=training_payload_extractor,
                device=device,
            )

            if save_influence_value_path is not None:
                for batch_idx, (_, samples_inf_val_dataset) in enumerate(influence_value_dataset):
                    self._save_dataset(samples_inf_val_dataset, f"{save_influence_value_path}/batch_{batch_idx:06d}")

            return influence_value_dataset

        if query_batching_config is not None:
            raise ValueError(
                "query_batching_config is only supported with preconditioning_mode=QUERY."
            )

        if influence_vector_in_cache == CACHE.NO_CACHE and load_influence_vector_path is None:
            warn("Warning: The computation is not efficient, thinks to use cache or disk save")

        if influence_vector_in_cache == CACHE.MEMORY:
            load_influence_vector_path = None

        if load_influence_vector_path is not None and influence_vector_in_cache == CACHE.DISK:
            inf_vect_ds = self._load_dataset(load_influence_vector_path)
            batch_size = self._backend.get_dataset_batch_size(train_set)
            inf_vect_ds = self._backend.zip_datasets(
                train_set,
                self._backend.batch_dataset(inf_vect_ds, batch_size)
            )
        else:
            inf_vect_ds = self.compute_influence_vector(train_set, save_influence_vector_path, device)

        if influence_vector_in_cache == CACHE.MEMORY:
            inf_vect_ds = self._backend.cache_dataset(inf_vect_ds)

        influence_value_dataset = self._backend.map_dataset(
            dataset_to_evaluate,
            lambda *batch_evaluate: self._estimate_inf_values_with_inf_vect_dataset(
                inf_vect_ds,
                batch_evaluate,
                evaluation_representation_provider=evaluation_representation_provider,
                training_payload_extractor=training_payload_extractor,
                device=device,
            ),
            device
        )

        if save_influence_value_path is not None:
            for batch_idx, (_, samples_inf_val_dataset) in enumerate(influence_value_dataset):
                self._save_dataset(samples_inf_val_dataset, f"{save_influence_value_path}/batch_{batch_idx:06d}")

        return influence_value_dataset

    def top_k(  # pylint: disable=R0913
            self,
            dataset_to_evaluate: DatasetLike,
            train_set: DatasetLike,
            k: int = 5,
            nearest_neighbors: Optional["BaseNearestNeighbors"] = None,
            influence_vector_in_cache: CACHE = CACHE.MEMORY,
            load_influence_vector_ds_path: Optional[str] = None,
            save_influence_vector_ds_path: Optional[str] = None,
            save_top_k_ds_path: Optional[str] = None,
            order: ORDER = ORDER.DESCENDING,
            d_type: Optional[DType] = None,
            payload_dtype: Optional[DType] = None,
            preconditioning_mode: Optional["PreconditioningMode"] = None,
            query_batching_config: Optional["QueryBatchingConfig"] = None,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None
    ) -> DatasetLike:
        """
        Find the top-k most influential training samples for each evaluation sample.

        Parameters
        ----------
        dataset_to_evaluate
            Dataset containing the samples to compare against the training
            dataset.
        train_set
            The dataset used to train the model.
        k
            Number of most influential training samples to retain.
        nearest_neighbors
            The nearest-neighbor method. The default method is a linear search.
        influence_vector_in_cache
            An enum indicating if intermediary values are to be cached (either in memory or on the disk) or not.
            Options include CACHE.MEMORY (0) for caching in memory, CACHE.DISK (1) for the disk and CACHE.NO_CACHE (2)
            for no optimization.
        load_influence_vector_ds_path
            The path to load the influence vectors (if they have already been calculated).
        save_influence_vector_ds_path
            The path to save the computed influence vector.
        save_top_k_ds_path
            The path to save the result of the computation of the top-k elements
        order
            Either ORDER.DESCENDING or ORDER.ASCENDING depending on if we wish to find the top-k or
            bottom-k samples, respectively.
        d_type
            The data-type of the tensors. If None, will be inferred.
        payload_dtype
            Data-type used to store extracted training payloads in the sorted
            structure. If ``None``, it is inferred.
        preconditioning_mode
            Whether to precondition training gradients (default) or query gradients.
        query_batching_config
            Optional configuration used only when
            ``preconditioning_mode=PreconditioningMode.QUERY``.
        evaluation_representation_provider
            Optional callable mapping ``(self.model, batch)`` to a custom
            representation used for scoring evaluation samples. When ``None``,
            the default ``_preprocess_samples`` path is used.
        training_payload_extractor
            Optional callable used to extract the payload returned alongside
            top-k influence scores for each training batch. When ``None``,
            ``default_training_payload_extractor`` is used.
        device
            Device where the computation will be executed.

        Raises
        ------
        ValueError
            If query-side preconditioning is combined with train-side
            influence-vector caching arguments, or if
            ``query_batching_config`` is provided while
            ``preconditioning_mode`` is not
            ``PreconditioningMode.QUERY``.

        Returns
        -------
        top_k_dataset
            A dataset containing the tuple (samples_to_evaluate, influence_values, training_samples).

            - samples_to_evaluate: Top-k samples to evaluate.
            - influence_values: Top-k influence values for each sample to evaluate.
            - training_samples: Top-k training sample for each sample to evaluate.
        """
        preconditioning_mode = self._resolve_preconditioning_mode(preconditioning_mode)
        if preconditioning_mode == PreconditioningMode.QUERY:
            if influence_vector_in_cache != CACHE.MEMORY:
                raise ValueError(
                    "influence_vector_in_cache is only supported with preconditioning_mode=TRAIN."
                )
            if load_influence_vector_ds_path is not None:
                raise ValueError(
                    "load_influence_vector_ds_path is only supported with preconditioning_mode=TRAIN."
                )
            if save_influence_vector_ds_path is not None:
                raise ValueError(
                    "save_influence_vector_ds_path is only supported with preconditioning_mode=TRAIN."
                )

            top_k_dataset = self._top_k_query_mode(
                dataset_to_evaluate,
                train_set,
                k=k,
                order=order,
                d_type=d_type,
                payload_dtype=payload_dtype,
                config=query_batching_config,
                evaluation_representation_provider=evaluation_representation_provider,
                training_payload_extractor=training_payload_extractor,
                device=device,
            )

            if save_top_k_ds_path is not None:
                self._save_dataset(top_k_dataset, save_top_k_ds_path)

            return top_k_dataset

        if query_batching_config is not None:
            raise ValueError(
                "query_batching_config is only supported with preconditioning_mode=QUERY."
            )

        if influence_vector_in_cache == CACHE.NO_CACHE and load_influence_vector_ds_path is None:
            warn("Warning: The computation is not efficient thinks to use cache or disk save")

        # Create nearest_neighbors with the correct backend if not provided
        if nearest_neighbors is None:
            nearest_neighbors = LinearNearestNeighbors(backend=self._backend)

        if influence_vector_in_cache == CACHE.MEMORY:
            load_influence_vector_ds_path = None

        if load_influence_vector_ds_path is not None and influence_vector_in_cache == CACHE.DISK:
            inf_vect_ds = self._load_dataset(load_influence_vector_ds_path)
            batch_size = self._backend.get_dataset_batch_size(train_set)
            inf_vect_ds = self._backend.zip_datasets(
                train_set,
                self._backend.batch_dataset(inf_vect_ds, batch_size)
            )
        else:
            inf_vect_ds = self.compute_influence_vector(train_set, save_influence_vector_ds_path, device)

        if influence_vector_in_cache == CACHE.MEMORY:
            inf_vect_ds = self._backend.cache_dataset(inf_vect_ds)

        nn_dataset = self._backend.map_dataset(
            inf_vect_ds,
            lambda *item: (
                self._extract_training_payload(item[0], training_payload_extractor),
                item[-1],
            ),
            device,
        )

        batch_size_eval = self._backend.get_dataset_batch_size(dataset_to_evaluate)

        # Infer dtype if not provided
        if d_type is None:
            # Get dtype from first batch of inf_vect_ds
            for item in inf_vect_ds:
                if isinstance(item, (list, tuple)):
                    d_type = item[-1].dtype
                else:
                    d_type = item.dtype
                break

        nearest_neighbors.build(
            nn_dataset,
            self._estimate_influence_value_from_influence_vector,
            k,
            query_batch_size=batch_size_eval,
            d_type=d_type,
            payload_dtype=payload_dtype,
            order=order,
        )

        top_k_dataset = self._backend.map_dataset(
            dataset_to_evaluate,
            lambda *batch_evaluate: self._top_k_with_inf_vect_dataset_train(
                batch_evaluate,
                nearest_neighbors,
                batch_size_eval,
                evaluation_representation_provider=evaluation_representation_provider,
                device=device,
            )
        )

        if save_top_k_ds_path is not None:
            self._save_dataset(top_k_dataset, save_top_k_ds_path)

        return top_k_dataset

    def _estimate_inf_values_with_inf_vect_dataset(
            self,
            inf_vect_dataset: DatasetLike,
            samples_to_evaluate: Tuple[Any, ...],
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None
    ) -> Tuple[Tuple[Any, ...], DatasetLike]:
        """
        Internal function to optimize computations when the influence vectors have already been calculated.

        Estimates the influence score between the samples we wish to evaluate and the set of influence
        vectors from the training dataset.

        Parameters
        ----------
        inf_vect_dataset
            A dataset with the influence vectors computed using some of the model's training data-points.
        samples_to_evaluate
            A tensor containing a single batch of samples of which we wish to estimate the influence of
             leaving out the training points corresponding to the influence vectors.
        evaluation_representation_provider
            Optional callable mapping ``(self.model, batch)`` to a custom
            representation used for scoring evaluation samples. When ``None``,
            the default ``_preprocess_samples`` path is used.
        training_payload_extractor
            Optional callable used to extract the payload returned alongside
            influence scores for each training batch. When ``None``,
            ``default_training_payload_extractor`` is used.
        device
            Device where the computation will be executed
        Returns
        -------
        output
            A tuple containing the original evaluation batch and a dataset of
            ``(training_payload, influence_scores)`` pairs.
        """
        preproc_samples_to_evaluate = self._get_evaluation_representation(
            samples_to_evaluate,
            evaluation_representation_provider,
        )
        samples_inf_val_dataset = self._backend.map_dataset(
            inf_vect_dataset,
            lambda *batch: (
                batch[:-1][0] if training_payload_extractor is None
                else self._extract_training_payload(batch[:-1][0], training_payload_extractor),
                self._estimate_influence_values_from_influence_vector(
                    samples_to_evaluate,
                    batch[-1],
                    preproc_samples_to_evaluate
                )
            ),
            device
        )
        return samples_to_evaluate, samples_inf_val_dataset

    def _top_k_with_inf_vect_dataset_train(
            self,
            sample_to_evaluate: Tuple[Any, ...],
            nearest_neighbor: "BaseNearestNeighbors",
            batch_size_eval: Optional[int] = None,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            device: Optional[str] = None
    ) -> Tuple[Tuple[Any, ...], Tensor, Tensor]:
        """
        Internal function to optimize computations when the influence vectors have already been calculated.

        Finds the top-k most influential training points for each test sample.

        Parameters
        ----------
        sample_to_evaluate
            A batched tensor containing the samples which will be compare to the training dataset
        nearest_neighbor
            The nearest neighbor method
        batch_size_eval
            The batch size for evaluation
        evaluation_representation_provider
            Optional callable mapping ``(self.model, batch)`` to a custom
            representation used for scoring evaluation samples. When ``None``,
            the default ``_preprocess_samples`` path is used.
        device
            Device where the computation will be executed
        Returns
        -------
        sample_to_evaluate
            sample to evaluate
        influence_values
            Top-k influence values for each sample to evaluate.
        training_samples
            Top-k training sample for each sample to evaluate.
        """
        _ = device
        v_to_evaluate = self._get_evaluation_representation(
            sample_to_evaluate,
            evaluation_representation_provider,
        )
        if batch_size_eval is None:
            influences_values, training_samples = nearest_neighbor.query(v_to_evaluate)
        else:
            influences_values, training_samples = nearest_neighbor.query(v_to_evaluate, batch_size_eval)

        return sample_to_evaluate, influences_values, training_samples

    def _estimate_influence_values_from_influence_vector(
            self,
            samples_to_evaluate: Tuple[Tensor, ...],
            inf_vect: Tensor,
            preproc_samples_to_evaluate: Optional[Tensor] = None
    ) -> Tensor:
        """
        Internal function to optimize computations when the influence vectors have already been calculated.

        Computes the influence values between each of the test samples and one influence vector from the training
        dataset.

        Parameters
        ----------
        samples_to_evaluate
            A single batch of test samples for which we wish to compute the influence of leaving out the training
            data-points corresponding to the influence vector.
        inf_vect
            Tensor containing the influence vectors for one training batch.
        preproc_samples_to_evaluate
            Optional preprocessed representation of ``samples_to_evaluate`` to avoid recomputing jacobians.

        Returns
        -------
        influence_values
            Tensor containing the influence scores between
            ``samples_to_evaluate`` and the training points represented by
            ``inf_vect``.
        """
        if preproc_samples_to_evaluate is None:
            v_to_evaluate = self._preprocess_samples(samples_to_evaluate)
        else:
            v_to_evaluate = preproc_samples_to_evaluate
        value = self._estimate_influence_value_from_influence_vector(v_to_evaluate, inf_vect)

        return value

    def _estimate_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[Tensor, ...],
            samples_to_evaluate: Tuple[Tensor, ...]
    ) -> Tensor:
        """
        Estimates the influence value of leaving out a single training sample on the provided test sample.

        Parameters
        ----------
        train_samples
            A single training sample
        samples_to_evaluate
            A single test sample.

        Returns
        -------
        influence_values
            A tensor with the resulting influence value.
        """
        v_train = self._compute_influence_vector(train_samples)
        v_to_evaluate = self._preprocess_samples(samples_to_evaluate)
        influence_values = self._estimate_influence_value_from_influence_vector(v_to_evaluate, v_train)

        return influence_values

    @abstractmethod
    def _estimate_influence_value_from_influence_vector(
            self,
            preproc_test_sample: Any,
            influence_vector: Any
    ) -> Any:
        """
        Estimates the influence score of leaving out the influence vector corresponding to a given training
        data-point on a test sample that has already been pre-processed.

        Parameters
        ----------
        preproc_test_sample
            A single pre-processed test sample we wish to evaluate.
        influence_vector
            A single influence vector corresponding to a data-point from the training dataset.

        Returns
        -------
        influence_values
            A tensor with the resulting influence value.
        """
        raise NotImplementedError()
