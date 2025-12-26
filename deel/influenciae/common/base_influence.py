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
"""
from abc import abstractmethod
from enum import Enum
from warnings import warn

from .backend import BaseBackend
from ..utils import BatchSort, BaseNearestNeighbors, LinearNearestNeighbors, ORDER
from ..types import Optional, Tuple, Any


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

    # Backend should be set by subclasses that have access to a model
    backend: BaseBackend = None

    @abstractmethod
    def _compute_influence_value_from_batch(self, train_samples: Tuple[Any, ...]) -> Any:
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

    def compute_influence_values(self, train_set: Any, device: Optional[str] = None) -> Any:
        """
        Compute the influence score for each sample of the provided (full or partial) model's training dataset.

        If only looking for the values, consider using the utility in deel.influenciae.utils.tf_operations:
        extract_only_values for converting this result into a tensor.

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
        return self.backend.map_dataset(
            train_set,
            lambda *batch_data: (batch_data, self._compute_influence_value_from_batch(batch_data)),
            device
        )

    def _compute_influence_values(self, train_set: Any, device: Optional[str] = None) -> Any:
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
            return self.backend.concat(inf_val_list, axis=0)
        return None

    def compute_top_k_from_training_dataset(
            self,
            train_set: Any,
            k: int,
            order: ORDER = ORDER.DESCENDING
    ) -> Tuple[Any, Any]:
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
        self.backend.assert_batched_dataset(train_set)

        # Get element spec for BatchSort initialization
        elt_spec = self.backend.get_dataset_element_spec(train_set)
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
                shape = self.backend.tensor_shape(first_tensor)[1:]
                dtype = first_tensor.dtype
                break

        if shape is None:
            raise ValueError("Could not determine tensor shape from dataset")

        batch_sorted_dict = BatchSort(shape, (1, k), dtype=dtype, order=order)

        for batch in train_set:
            influence_values = self._compute_influence_value_from_batch(batch)
            if self.backend.tensor_ndim(influence_values) == 1:
                influence_values = self.backend.expand_dims(influence_values, axis=-1)

            batch_input = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_sorted_dict.add_all(
                self.backend.expand_dims(batch_input, axis=0),
                self.backend.transpose(influence_values)
            )

        best_samples, best_values = batch_sorted_dict.get()
        influence_values = self.backend.stack(best_values)
        training_samples = self.backend.concat(
            [self.backend.expand_dims(v, axis=0) for v in best_samples], axis=0
        )
        training_samples = self.backend.squeeze(training_samples, axis=0)
        influence_values = self.backend.squeeze(influence_values, axis=0)

        return training_samples, influence_values

    def _save_dataset(self, dataset: Any, load_or_save_path: str) -> None:
        """
        Save a dataset in the appropriate format for the backend.

        Parameters
        ----------
        dataset
            The dataset to save
        load_or_save_path
            The path to save the dataset
        """
        self.backend.save_dataset(dataset, load_or_save_path)

    def _load_dataset(self, dataset_path: str) -> Any:
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
        return self.backend.load_dataset(dataset_path)


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
        Preprocess a sample to evaluate

        Parameters
        ----------
        samples
            sample to evaluate
        Returns
        -------
        The preprocessed sample to evaluate
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
            train_set: Any,
            save_influence_vector_ds_path: Optional[str] = None,
            device: Optional[str] = None
    ) -> Any:
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
        inf_vect_ds = self.backend.map_dataset(
            train_set,
            lambda *batch: (batch, self._compute_influence_vector(batch)),
            device
        )

        if save_influence_vector_ds_path is not None:
            # Extract just the influence vectors and save
            inf_vect_list = []
            for item in inf_vect_ds:
                if isinstance(item, (list, tuple)):
                    inf_vect_list.append(item[-1])
                else:
                    inf_vect_list.append(item)

            # Unbatch and save
            unbatched_inf_vect = self.backend.unbatch_dataset(inf_vect_list)
            self._save_dataset(unbatched_inf_vect, save_influence_vector_ds_path)

        return inf_vect_ds

    def estimate_influence_values_in_batches(
            self,
            dataset_to_evaluate: Any,
            train_set: Any,
            influence_vector_in_cache: CACHE = CACHE.MEMORY,
            load_influence_vector_path: Optional[str] = None,
            save_influence_vector_path: Optional[str] = None,
            save_influence_value_path: Optional[str] = None,
            device: Optional[str] = None
    ) -> Any:
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
        device
            Device where the computation will be executed

        Returns
        -------
        influence_value_dataset
            A dataset containing the tuple: (samples_to_evaluate, dataset).

            - samples_to_evaluate: The batch of sample to evaluate.
            - dataset: Dataset containing tuples of batch of the training dataset and their influence score.
        """
        if not influence_vector_in_cache and load_influence_vector_path is None:
            warn("Warning: The computation is not efficient, thinks to use cache or disk save")

        if influence_vector_in_cache == CACHE.MEMORY:
            load_influence_vector_path = None

        if load_influence_vector_path is not None and influence_vector_in_cache == CACHE.DISK:
            inf_vect_ds = self._load_dataset(load_influence_vector_path)
            batch_size = self.backend.get_dataset_batch_size(train_set)
            inf_vect_ds = self.backend.zip_datasets(
                train_set,
                self.backend.batch_dataset(inf_vect_ds, batch_size)
            )
        else:
            inf_vect_ds = self.compute_influence_vector(train_set, save_influence_vector_path, device)

        if influence_vector_in_cache == CACHE.MEMORY:
            inf_vect_ds = self.backend.cache_dataset(inf_vect_ds)

        influence_value_dataset = self.backend.map_dataset(
            dataset_to_evaluate,
            lambda *batch_evaluate: self._estimate_inf_values_with_inf_vect_dataset(
                inf_vect_ds, batch_evaluate, device
            ),
            device
        )

        if save_influence_value_path is not None:
            for batch_idx, (_, samples_inf_val_dataset) in enumerate(influence_value_dataset):
                self._save_dataset(samples_inf_val_dataset, f"{save_influence_value_path}/batch_{batch_idx:06d}")

        return influence_value_dataset

    def top_k(  # pylint: disable=R0913
            self,
            dataset_to_evaluate: Any,
            train_set: Any,
            k: int = 5,
            nearest_neighbors: BaseNearestNeighbors = LinearNearestNeighbors(),
            influence_vector_in_cache: CACHE = CACHE.MEMORY,
            load_influence_vector_ds_path: Optional[str] = None,
            save_influence_vector_ds_path: Optional[str] = None,
            save_top_k_ds_path: Optional[str] = None,
            order: ORDER = ORDER.DESCENDING,
            d_type: Any = None,
            device: Optional[str] = None
    ) -> Any:
        """
        Find the top-k closest elements for each element of dataset to evaluate in the training dataset
        The method will return a dataset containing a tuple of:
            (Top-k influence values for each sample to evaluate, Top-k training sample for each sample to evaluate)

        Parameters
        ----------
        dataset_to_evaluate
            The dataset which contains the samples which will be compare to the training dataset
        train_set
            The dataset used to train the model.
        k
            the number of most influence samples to retain in training dataset
        nearest_neighbors
            The nearest neighbor method. The default method is a linear search
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
        device
            Device where the computation will be executed

        Returns
        -------
        top_k_dataset
            A dataset containing the tuple (samples_to_evaluate, influence_values, training_samples).

            - samples_to_evaluate: Top-k samples to evaluate.
            - influence_values: Top-k influence values for each sample to evaluate.
            - training_samples: Top-k training sample for each sample to evaluate.
        """
        if not influence_vector_in_cache and load_influence_vector_ds_path is None:
            warn("Warning: The computation is not efficient thinks to use cache or disk save")

        if influence_vector_in_cache == CACHE.MEMORY:
            load_influence_vector_ds_path = None

        if load_influence_vector_ds_path is not None and influence_vector_in_cache == CACHE.DISK:
            inf_vect_ds = self._load_dataset(load_influence_vector_ds_path)
            batch_size = self.backend.get_dataset_batch_size(train_set)
            inf_vect_ds = self.backend.zip_datasets(
                train_set,
                self.backend.batch_dataset(inf_vect_ds, batch_size)
            )
        else:
            inf_vect_ds = self.compute_influence_vector(train_set, save_influence_vector_ds_path, device)

        if influence_vector_in_cache == CACHE.MEMORY:
            inf_vect_ds = self.backend.cache_dataset(inf_vect_ds)

        batch_size_eval = self.backend.get_dataset_batch_size(dataset_to_evaluate)

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
            inf_vect_ds,
            self._estimate_influence_value_from_influence_vector,
            k,
            query_batch_size=batch_size_eval,
            d_type=d_type,
            order=order,
        )

        top_k_dataset = self.backend.map_dataset(
            dataset_to_evaluate,
            lambda *batch_evaluate: self._top_k_with_inf_vect_dataset_train(
                batch_evaluate, nearest_neighbors, batch_size_eval, device
            )
        )

        if save_top_k_ds_path is not None:
            self._save_dataset(top_k_dataset, save_top_k_ds_path)

        return top_k_dataset

    def _estimate_inf_values_with_inf_vect_dataset(
            self,
            inf_vect_dataset: Any,
            samples_to_evaluate: Tuple[Any, ...],
            device: Optional[str] = None
    ) -> Tuple[Tuple[Any, ...], Any]:
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
        device
            Device where the computation will be executed
        Returns
        -------
        A dataset containing the tuple:
            batch of the training dataset
            influence scores
        """
        samples_inf_val_dataset = self.backend.map_dataset(
            inf_vect_dataset,
            lambda *batch: (
                batch[:-1][0],
                self._estimate_influence_values_from_influence_vector(samples_to_evaluate, batch[-1])
            ),
            device
        )
        return samples_to_evaluate, samples_inf_val_dataset

    def _top_k_with_inf_vect_dataset_train(
            self,
            sample_to_evaluate: Tuple[Any, ...],
            nearest_neighbor: BaseNearestNeighbors,
            batch_size_eval: Optional[int] = None,
            device: Optional[str] = None
    ) -> Tuple[Tuple[Any, ...], Any, Tuple[Any, ...]]:
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
        v_to_evaluate = self._preprocess_samples(sample_to_evaluate)
        if batch_size_eval is None:
            influences_values, training_samples = nearest_neighbor.query(v_to_evaluate)
        else:
            influences_values, training_samples = nearest_neighbor.query(v_to_evaluate, batch_size_eval)

        return sample_to_evaluate, influences_values, training_samples

    def _estimate_influence_values_from_influence_vector(
            self,
            samples_to_evaluate: Tuple[Any, ...],
            inf_vect: Any
    ) -> Any:
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
            A tensor with one influence vector

        Returns
        -------
        Tuple:
            batch of the training dataset
            influence vector
        """
        v_to_evaluate = self._preprocess_samples(samples_to_evaluate)
        value = self._estimate_influence_value_from_influence_vector(v_to_evaluate, inf_vect)

        return value

    def _estimate_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[Any, ...],
            samples_to_evaluate: Tuple[Any, ...]
    ) -> Any:
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
