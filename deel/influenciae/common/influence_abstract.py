# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module defining the interface for all the different influence calculator classes.

The BaseInfluenceCalculator interface provides implementations to common to all of
their child classes (i.e. all the different techniques to compute a notion of influence).

The VectorBasedInfluenceCalculator provides optimized implementations for some methods
following the assumption that the computation can be written as a matrix-vector product
with a matrix that can be (pre)-computed and remains unchanged throughout the computation.
"""
from abc import abstractmethod
from os import path
from warnings import warn
from xml.dom import NotFoundErr

import tensorflow as tf

from ..utils import BatchSort, BaseNearestNeighbors, LinearNearestNeighbors
from ..utils.sorted_dict import ORDER

from ..utils import assert_batched_dataset
from ..types import Optional, Tuple


class BaseInfluenceCalculator:
    """
    The base implementation of an interface for all the influence calculators in the library.
    All the methods included in deel-influenciae implement these basic functions, with
    each their own notion of influence of a data-point on the model.
    """
    @abstractmethod
    def _compute_influence_value_from_batch(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        # Formerly compute_pairwise_influence_value
        """
        Compute the influence score for a single batch of training samples.

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

    @abstractmethod
    def compute_influence_vector(
        self,
        train_set: tf.data.Dataset,
        save_influence_vector_ds_path: Optional[str] = None
    ) -> tf.data.Dataset:
        # Formerly compute_influence_vector_dataset
        """
        Compute the influence vector for each sample of the provided (full or partial) model's training dataset.

        Parameters
        ----------
        train_set
            A TF dataset with the (full or partial) model's training dataset.
        save_influence_vector_ds_path
            The path to save or load the influence vector of the training dataset. If specified,
            load the dataset if it has already been computed, otherwise, compute the influence vector and
            then save it in the specified path.

        Returns
        -------
        A dataset containing the tuple: (batch of training samples, influence vector)
        """
        raise NotImplementedError()

    def compute_influence_values(self, train_set: tf.data.Dataset) -> tf.data.Dataset:
        # Formerly compute_influence_values_dataset
        """
        Compute the influence score for each sample of the provided (full or partial) model's training dataset.

        Parameters
        ----------
        train_set
            A TF dataset with the (full or partial) model's training dataset.

        Returns
        -------
        A dataset containing the tuple: (batch of training samples, influence score)
        """
        train_set = train_set.map(
            lambda *batch_data: (batch_data, self._compute_influence_value_from_batch(batch_data))
        )

        return train_set

    def _compute_influence_values(self, train_set: tf.data.Dataset) -> tf.Tensor:
        # Formerly compute_influence_values
        """
        Compute the influence score for each sample of the provided (full or partial) model's training dataset.
        For internal use only.
        This version returns a tensor instead of a dataset.

        Parameters
        ----------
        train_set
            A TF dataset with the (full or partial) model's training dataset.
        Returns
        -------
        influence score
            A tensor with the sample's influence scores.
        """
        influences_values = self.compute_influence_values(train_set)
        influences_values = influences_values.map(
            lambda _, inf_val: inf_val
        )
        inf_val = None
        for batch_inf in influences_values:
            inf_val = batch_inf if inf_val is None else tf.concat([inf_val, batch_inf], axis=0)

        return inf_val

    def compute_top_k_from_training_dataset(
            self,
            train_set: tf.data.Dataset,
            k: int,
            order: ORDER = ORDER.DESCENDING
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Same name
        """
        Compute the k most influential data-points of the model's training dataset by computing
        Cook's distance for each point individually.

        Parameters
        ----------
        train_set
            A TF dataset containing the points on which the model was trained.
        k
            An integer with the number of most important samples we wish to keep
        order
            Either ORDER.DESCENDING or ORDER.ASCENDING depending on if we wish to find the top-k or
            bottom-k samples, respectively.

        Returns
        -------
        A tuple containing:
            training_samples
                A tensor containing the k most influential samples of the training dataset for the model
                provided
            influences_values
                The influence score corresponding to these k most influential samples
        """
        assert_batched_dataset(train_set)
        elt_spec = train_set.element_spec[0]
        batch_sorted_dict = BatchSort(elt_spec.shape[1:], (1, k), dtype=elt_spec.dtype, order=order)

        for batch in train_set:
            influence_values = self._compute_influence_value_from_batch(batch)
            batch_sorted_dict.add_all(tf.expand_dims(batch[0], axis=0), tf.transpose(influence_values))

        best_samples, best_values = batch_sorted_dict.get()
        influence_values = tf.stack(best_values)
        training_samples = tf.concat(
            [tf.expand_dims(v, axis=0) for v in best_samples], axis=0
        )
        training_samples, influence_values = tf.squeeze(training_samples, axis=0), tf.squeeze(influence_values, axis=0)

        return training_samples, influence_values

    @abstractmethod
    def _compute_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[tf.Tensor, ...],
            samples_to_evaluate: Tuple[tf.Tensor, ...]
    ) -> tf.Tensor:
        # Formerly compute_influence_values_from_tensor
        """
        Compute the (individual) influence scores of a single batch of samples with respect to
        a batch of samples belonging to the model's training dataset.

        Parameters
        ----------
        train_samples
            A single batch of training samples (and their target values).
        samples_to_evaluate
            A single batch of samples of which we wish to compute the influence of removing the training
            samples.

        Returns
        -------
        A tensor containing the individual influence scores.
        """
        raise NotImplementedError()

    @abstractmethod
    def _compute_influence_values_in_batches(
            self,
            train_set: tf.data.Dataset,
            test_samples: Tuple[tf.Tensor, ...],
            load_influence_vector_ds_path: str = None,
            save_influence_vector_ds_path: str = None
    ) -> tf.data.Dataset:
        # Formerly compute_influence_values_for_sample_to_evaluate
        """
        Compute the influence scores between each individual test sample and each point in the provided
        (full or partial) training dataset.

        Parameters
        ----------
        train_set
            A TF dataset containing the model's training dataset (partial or full).
        test_samples
            A batched tensor containing the samples of which we wish to know the impact of leaving out each of
            the train samples.
        load_influence_vector_ds_path
            The path to load the influence vector (if it has already been calculated).
        save_influence_vector_ds_path
            The path to save the computed influence vector.

        Returns
        -------
        A dataset containing the tuple: (batch of training samples, influence score)
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_influence_values_in_batches(
            self,
            dataset_to_evaluate: tf.data.Dataset,  # TODO rename...
            train_set: tf.data.Dataset,
            influence_vector_in_cache: bool = True,
            load_influence_vector_path: str = None,
            save_influence_vector_path: str = None,
            load_influence_value_path: str = None,
            save_influence_value_path: str = None
    ) -> tf.data.Dataset:
        # Formerly compute_influence_values_for_dataset_to_evaluate
        """
        Compute the influence score between a dataset of samples to evaluate and the training dataset

        Parameters
        ----------
        dataset_to_evaluate
            the dataset containing the sample to evaluate
        train_set
            A TF dataset containing the model's training dataset (partial or full).
        influence_vector_in_cache
            If True and load_or_save_train_influence_vector_path=None, cache in memory the influence vector.
            If load_or_save_train_influence_vector_path!=None, the influence vector will be saved in the disk.
        load_influence_vector_path
            The path to load the influence vectors (if they have already been calculated).
        save_influence_vector_path
            The path to save the computed influence vector.
        load_influence_value_path
            The path to load the influence values (if they have already been calculated).
        save_influence_value_path
            The path to save the computed influence values.

        Returns
        -------
        A dataset containing the tuple:
            batch of sample to evaluate
            dataset:
                batch of the training dataset
                influence score
        """
        raise NotImplementedError()

    @abstractmethod
    def _top_k_from_batch(
            self,
            sample_to_evaluate: Tuple[tf.Tensor, ...],
            train_set: tf.data.Dataset,
            k: int = 5,
            nearest_neighbors: BaseNearestNeighbors = LinearNearestNeighbors(),
            d_type: tf.DType = tf.float32,
            load_influence_vector_ds_path=None,
            save_influence_vector_ds_path=None,
            order: ORDER = ORDER.DESCENDING
    ) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor, Tuple[tf.Tensor, ...]]:
        # Formerly top_k
        """
        Find the top-k closest elements of the training dataset for each sample to evaluate

        Parameters
        ----------
        sample_to_evaluate
            A batched tensor containing the samples of which we wish to know the impact of leaving out each of
            the train samples.
        train_set
            The dataset used to train the model.
        k
            the number of most influence samples to retain in training datatse
        nearest_neighbors
            The nearest neighbor method. The default method is a linear search
        load_influence_vector_ds_path
            The path to load the influence vectors (if they have already been calculated).
        save_influence_vector_ds_path
            The path to save the computed influence vectors.

        Returns
        -------
        A tuple with:
            influence_values
                Top-k influence values for each sample to evaluate.
            training_samples
                Top-k training sample for each sample to evaluate.
        """
        raise NotImplementedError()

    @abstractmethod
    def top_k(
            self,
            dataset_to_evaluate: tf.data.Dataset,
            train_set: tf.data.Dataset,
            k: int = 5,
            nearest_neighbors: BaseNearestNeighbors = LinearNearestNeighbors(),
            d_type: tf.DType = tf.float32,
            vector_influence_in_cache: bool = True,
            load_influence_vector_ds_path: str = None,
            save_influence_vector_ds_path: str = None,
            save_top_k_ds_path: str = None,
            order: ORDER = ORDER.DESCENDING
    ) -> tf.data.Dataset:
        # Formerly top_k_dataset
        """
        #TODO: Builds a dataset ((batch_samples_to_evaluate), influence VECTOR OR VALUE)
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
        vector_influence_in_cache:
            If True and load_or_save_train_influence_vector_path=None, cache in memory the influence vector
            If load_or_save_train_influence_vector_path!=None, the influence vector will be saved in the disk
        load_or_save_train_influence_vector_path
            The path to save or load the influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path
        load_or_save_top_k_path
            The path to save or load the result of the computation of the top-k elements

        Returns
        -------
        A dataset containing the tuple:
            samples_to_evaluate
                Top-k samples to evaluate.
            influence_values
                Top-k influence values for each sample to evaluate.
            training_samples
                Top-k training sample for each sample to evaluate.
        """
        raise NotImplementedError()

    def save_dataset(self, dataset: tf.data.Dataset, load_or_save_path: str) -> None:
        """
        Save a dataset in the TF dataset format in the specified path.

        Parameters
        ----------
        dataset
            The dataset to save
        load_or_save_path
            The path to save the dataset
        """
        tf.data.experimental.save(dataset, load_or_save_path)

    def load_dataset(self, dataset_path: str) -> tf.data.Dataset:
        """
        Loads a dataset in the TF format from the specified path.

        Parameters
        ----------
        dataset_path
            The path pointing to the file from which to load the dataset

        Returns
        -------
        dataset
            The target dataset
        """
        if path.exists(dataset_path):
            dataset = tf.data.experimental.load(dataset_path)
        else:
            raise NotFoundErr(f"The dataset path: {dataset_path} was not found")
        return dataset


class VectorBasedInfluenceCalculator(BaseInfluenceCalculator):
    """
    When the computation of the influence scores can be written as a vector inner product,
    there are ways to optimize and allow us to scale to large datasets.
    This interface is an abstraction of this optimization.
    """
    @abstractmethod
    def _compute_influence_vector(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Compute the influence vector for a training sample

        Parameters
        ----------
        train_samples
            sample to evaluate
        Returns
        -------
        The influence vector for the training sample
        """
        raise NotImplementedError()

    @abstractmethod
    def _preprocess_samples(self, samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        # Formerly preprocess_sample_to_evaluate
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
    def _compute_influence_value_from_influence_vector(
            self,
            preproc_test_sample: tf.Tensor,
            influence_vector: tf.Tensor
    ) -> tf.Tensor:
        # Formerly compute_influence_value_from_influence_vector
        """
        Compute the influence score for a preprocessed sample to evaluate and a training influence VECTOR

        Parameters
        ----------
        preproc_test_sample
            Preprocessed sample to evaluate
        influence_vector
            Training influence Vector
        Returns
        -------
        The influence score
        """
        raise NotImplementedError()

    def compute_influence_vector(
        self,
        train_set: tf.data.Dataset,
        save_influence_vector_ds_path: Optional[str] = None
    ) -> tf.data.Dataset:
        """
        Compute the influence vector for each sample of the training dataset

        Parameters
        ----------
        train_set
            The training dataset
        save_influence_vector_ds_path
            The path to save influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path

        Returns
        -------
        A dataset containing the tuple:
            batch of the training dataset
            influence vector
        """
        # TODO: Improve by saving only the influence vector
        inf_vect_ds = train_set.map(lambda *batch: (batch, self._compute_influence_vector(batch)))
        if save_influence_vector_ds_path is not None:
            inf_vect = inf_vect_ds.map(lambda *batch: batch[-1])
            self.save_dataset(inf_vect.unbatch(), save_influence_vector_ds_path)
        return inf_vect_ds

    @tf.function
    def _compute_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[tf.Tensor, ...],
            samples_to_evaluate: Tuple[tf.Tensor, ...]
    ) -> tf.Tensor:
        """
        Compute the influence vector between a sample to evaluate and a training sample

        Parameters
        ----------
        train_samples
            Training sample
        samples_to_evaluate
            Sample to evaluate

        Returns
        -------
        The influence score
        """
        v_train = self._compute_influence_vector(train_samples)
        v_to_evaluate = self._preprocess_samples(samples_to_evaluate)
        influence_values = self._compute_influence_value_from_influence_vector(v_to_evaluate, v_train)

        return influence_values

    def _compute_influence_values_in_batches(
            self,
            train_set: tf.data.Dataset,
            test_samples: Tuple[tf.Tensor, ...],
            load_influence_vector_ds_path: Optional[str] = None,
            save_influence_vector_ds_path: Optional[str] = None
    ) -> Tuple[Tuple[tf.Tensor, ...], tf.data.Dataset]:
        """
        Compute the influence score between samples to evaluate and training dataset

        Parameters
        ----------
        train_set
            The training dataset
        sample_to_evaluate
            A batched tensor containing the samples which will be compared to the training dataset
        load_or_save_train_influence_vector_path
            The path to save or load the influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path

        Returns
        -------
        A dataset containing the tuple:
            batch of the training dataset
            influence score
        """
        if load_influence_vector_ds_path is not None:
            inf_vect_ds = self.load_dataset(load_influence_vector_ds_path)
            batch_size = train_set._batch_size # pylint: disable=W0212
            inf_vect_ds = tf.data.Dataset.zip((train_set, inf_vect_ds.batch(batch_size)))
            if save_influence_vector_ds_path is not None:
                warn("Since you loaded the inf_vect_ds we ignore the saving option (useless)")
        else:
            inf_vect_ds = self.compute_influence_vector(train_set, save_influence_vector_ds_path)

        return self._compute_inf_values_with_inf_vect_dataset(inf_vect_ds, test_samples)

    def _compute_inf_values_with_inf_vect_dataset(
            self,
            inf_vect_dataset: tf.data.Dataset,
            samples_to_evaluate: Tuple[tf.Tensor, ...]) -> Tuple[Tuple[tf.Tensor, ...],tf.data.Dataset]:
        """
        Compute the influence score between samples to evaluate and vector of influence of training dataset

        Parameters
        ----------
        inf_vect_dataset
            The vector of influence of training dataset
        sample_to_evaluate
            A batched tensor containing the samples which will be compare to the training dataset

        Returns
        -------
        A dataset containing the tuple:
            batch of the training dataset
            influence score
        """
        samples_inf_val_dataset = inf_vect_dataset.map(
            lambda *batch:
            (batch[:-1][0],
             self._compute_influence_values_from_influence_vector(samples_to_evaluate, batch[-1]))
        )
        return samples_to_evaluate, samples_inf_val_dataset

    def _compute_influence_values_from_influence_vector(
        self,
        samples_to_evaluate: Tuple[tf.Tensor, ...],
        inf_vect: tf.Tensor
    ) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor]:
        """
        Compute the influence vector between a sample to evaluate and the training vector of influence

        Parameters
        ----------
        batch
            # TODO: Fill that
        ihvp
            The influence vector
        Returns
        -------
        Tuple:
            batch of the training dataset
            influence vector
        """
        v_to_evaluate = self._preprocess_samples(samples_to_evaluate)
        value = self._compute_influence_value_from_influence_vector(v_to_evaluate, inf_vect)
        return value

    def compute_influence_values_in_batches(
            self,
            dataset_to_evaluate: tf.data.Dataset,
            train_set: tf.data.Dataset,
            influence_vector_in_cache: bool = True,
            load_influence_vector_path: Optional[str] = None,
            save_influence_vector_path: Optional[str] = None,
            load_influence_value_path: Optional[str] = None,
            save_influence_value_path: Optional[str] = None
    ) -> tf.data.Dataset:
        """
        Compute the influence score between a dataset of samples to evaluate and the training dataset

        Parameters
        ----------
        dataset_to_evaluate
            the dataset containing the sample to evaluate
        train_set
            The training dataset
        influence_vector_in_cache
            If True and load_or_save_train_influence_vector_path=None, cache in memory the influence vector
            If load_or_save_train_influence_vector_path!=None, the influence vector will be saved in the disk
        load_or_save_train_influence_vector_path
            The path to save or load the influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path
        load_or_save_influence_value_path
            The path to save or load the result of the computation of the influence values

        Returns
        -------
        A dataset containing the tuple:
            batch of sample to evaluate
            dataset:
                batch of the training dataset
                influence score
        """
        if not influence_vector_in_cache and load_influence_vector_path is None:
            warn("Warning: The computation is not efficient, thinks to use cache or disk save")

        if influence_vector_in_cache:
            # TODO: Question the intended behavior here, is it save or load ?
            load_influence_vector_path = None

        if load_influence_vector_path is not None:
            inf_vect_ds = self.load_dataset(load_influence_vector_path)
            batch_size = train_set._batch_size # pylint: disable=W0212
            inf_vect_ds = tf.data.Dataset.zip((train_set, inf_vect_ds.batch(batch_size)))
        else:
            inf_vect_ds = self.compute_influence_vector(train_set,
                                                        save_influence_vector_path)

        if influence_vector_in_cache:
            inf_vect_ds = inf_vect_ds.cache()

        influence_value_dataset = dataset_to_evaluate.map(
            lambda *batch_evaluate: self._compute_inf_values_with_inf_vect_dataset(
                inf_vect_ds, batch_evaluate
                )
        )

        if save_influence_value_path is not None:
            for batch_idx, (_, samples_inf_val_dataset) in enumerate(influence_value_dataset):
                self.save_dataset(samples_inf_val_dataset, f"{save_influence_value_path}/batch_{batch_idx:06d}")

        return influence_value_dataset

    def _top_k_from_batch(
            self,
            sample_to_evaluate: Tuple[tf.Tensor, ...],
            train_set: tf.data.Dataset,
            k: int = 5,
            nearest_neighbors: BaseNearestNeighbors = LinearNearestNeighbors(),
            d_type: tf.DType = tf.float32,
            load_influence_vector_ds_path: Optional[str] = None,
            save_influence_vector_ds_path: Optional[str] = None,
            order: ORDER = ORDER.DESCENDING
    ) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor, Tuple[tf.Tensor, ...]]:
        """
        Find the top-k closest elements of the training dataset for each sample to evaluate

        Parameters
        ----------
        sample_to_evaluate
            A batched tensor containing the samples which will be compare to the training dataset
        train_set
            The dataset used to train the model.
        k
            the number of most influence samples to retain in training datatse
        nearest_neighbors
            The nearest neighbor method. The default method is a linear search
        load_or_save_train_influence_vector_path
            The path to save or load the influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path
        Returns
        -------
        sample_to_evaluate
            Sample to evaluate
        influence_values
            Top-k influence values for each sample to evaluate.
        training_samples
            Top-k training sample for each sample to evaluate.
        """
        if load_influence_vector_ds_path is not None:
            inf_vect_ds = self.load_dataset(load_influence_vector_ds_path)
            batch_size = train_set._batch_size # pylint: disable=W0212
            inf_vect_ds = tf.data.Dataset.zip((train_set, inf_vect_ds.batch(batch_size)))
        else:
            inf_vect_ds = self.compute_influence_vector(train_set, save_influence_vector_ds_path)
        batch_size_eval = int(tf.shape(sample_to_evaluate[0])[0])

        nearest_neighbors.build(
            inf_vect_ds,
            self._compute_influence_value_from_influence_vector,
            k,
            query_batch_size=batch_size_eval,
            d_type=d_type,
            order=order
        )

        return self._top_k_with_inf_vect_dataset_train(sample_to_evaluate, nearest_neighbors, batch_size_eval)

    def top_k(
            self,
            dataset_to_evaluate: tf.data.Dataset,
            train_set: tf.data.Dataset,
            k: int = 5,
            nearest_neighbors: BaseNearestNeighbors = LinearNearestNeighbors(),
            d_type: tf.DType = tf.float32,
            vector_influence_in_cache: bool = True,
            load_influence_vector_ds_path: Optional[str] = None,
            save_influence_vector_ds_path: Optional[str] = None,
            save_top_k_ds_path: Optional[str] = None,
            order: ORDER = ORDER.DESCENDING
    ) -> tf.data.Dataset:
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
        vector_influence_in_cache:
            If True and load_or_save_train_influence_vector_path=None, cache in memory the influence vector
            If load_or_save_train_influence_vector_path!=None, the influence vector will be saved in the disk
        load_or_save_train_influence_vector_path
            The path to save or load the influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path
        load_or_save_top_k_path
            The path to save or load the result of the computation of the top-k elements
        Returns
        -------
        A dataset containing the tuple:
            samples_to_evaluate
                samples to evaluate
            influence_values
                Top-k influence values for each sample to evaluate.
            training_samples
                Top-k training sample for each sample to evaluate.
        """
        if not vector_influence_in_cache and load_influence_vector_ds_path is None:
            warn("Warning: The computation is not efficience thinks to use cache or disk save")

        if vector_influence_in_cache:
            # TODO: Question the intended behavior here, is it save or load ?
            load_influence_vector_ds_path = None

        if load_influence_vector_ds_path is not None:
            inf_vect_ds = self.load_dataset(load_influence_vector_ds_path)
            batch_size = train_set._batch_size # pylint: disable=W0212
            inf_vect_ds = tf.data.Dataset.zip((train_set, inf_vect_ds.batch(batch_size)))
        else:
            inf_vect_ds = self.compute_influence_vector(train_set,
                                                        save_influence_vector_ds_path)

        if vector_influence_in_cache:
            inf_vect_ds = inf_vect_ds.cache()

        batch_size_eval = int(dataset_to_evaluate._batch_size) # pylint: disable=W0212
        nearest_neighbors.build(
            inf_vect_ds,
            self._compute_influence_value_from_influence_vector,
            k,
            query_batch_size=batch_size_eval,
            d_type = d_type,
            order=order,
        )

        top_k_dataset = dataset_to_evaluate.map(
            lambda *batch_evaluate: self._top_k_with_inf_vect_dataset_train(
                batch_evaluate, nearest_neighbors, batch_size_eval
            )
        )

        if save_top_k_ds_path is not None:
            self.save_dataset(top_k_dataset, save_top_k_ds_path)

        return top_k_dataset

    def _top_k_with_inf_vect_dataset_train(self,
                                           sample_to_evaluate: Tuple[tf.Tensor, ...],
                                           nearest_neighbor: BaseNearestNeighbors,
                                           batch_size_eval: Optional[int] = None) -> Tuple[
                                                Tuple[tf.Tensor, ...], tf.Tensor, Tuple[tf.Tensor, ...]]:
        """
        Find the top-k closest elements for each sample to evaluate

        Parameters
        ----------
        sample_to_evaluate
            A batched tensor containing the samples which will be compare to the training dataset
        nearest_neighbor
            The nearest neighbor method
        k
            the number of most influence samples to retain in training datatse
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
