# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TODO: Insert insightful description
"""
from abc import abstractmethod
from os import path
from warnings import warn
from xml.dom import NotFoundErr

import numpy as np
import tensorflow as tf

from ..utils import BatchSort, BaseNearestNeighbor, LinearNearestNeighbor

from ..utils import assert_batched_dataset
from ..types import Optional, Tuple

class BaseInfluenceCalculator:
    """
    The main abstraction of influence calculators
    """

    @abstractmethod
    def compute_pairwise_influence_value(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Compute the influence score for a single batch of training samples

        Parameters
        ----------
        train_samples
            A single batch of training sample
        Returns
        -------
        influence_values
            The influence score of each sample in the batch train_samples
        """
        raise NotImplementedError()


    @abstractmethod
    def compute_influence_vector_dataset(
        self,
        dataset_train: tf.data.Dataset,
        save_influence_vector_ds_path: str = None) -> tf.data.Dataset:
        """
        Compute the influence vector for each sample of the training dataset

        Parameters
        ----------
        dataset_train
            The training dataset
        load_or_save_train_influence_vector_path
            The path to save or load the influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path
        Returns
        -------
        A dataset containing the tuple:
            batch of the training dataset
            influence vector
        """
        raise NotImplementedError()


    def compute_influence_values_dataset(self, dataset_train: tf.data.Dataset) -> tf.data.Dataset:
        """
        #TODO: It is not using function specific to this class, could be put at higher level ?
        Compute the influence score for each sample of the training dataset

        Parameters
        ----------
        dataset_train
            The training dataset
        Returns
        -------
        A dataset containing the tuple:
            batch of the training dataset
            influence score
        """
        dataset_train = dataset_train.map(
            lambda *batch_data: (batch_data, self.compute_pairwise_influence_value(batch_data)))

        return dataset_train

    def compute_influence_values(self, dataset_train: tf.data.Dataset) -> np.array:
        """
        Compute the influence score for each sample of the training dataset

        Parameters
        ----------
        dataset_train
            The training dataset
        Returns
        -------
        influence score
            The influence score
        """
        influences_values = self.compute_influence_values_dataset(dataset_train)
        influences_values = influences_values.map(
            lambda _, inf_val: inf_val
        )
        inf_val = None
        for batch_inf in influences_values:
            inf_val = batch_inf if inf_val is None else tf.concat([inf_val, batch_inf], axis=0)
        return inf_val


    def compute_top_k_from_training_dataset(self, dataset_train: tf.data.Dataset, k: int) -> Tuple[
                                                                                                tf.Tensor, tf.Tensor]:
        """
        #TODO: Clarify documentation
        Compute the top-k most influent from training dataset

        Parameters
        ----------
        dataset_train
            The training dataset
        k
            the number of most important samples to keep
        Returns
        -------
        training_samples
            k most important samples of the training dataset
        influences_values
            influence score of k most important samples
        """
        assert_batched_dataset(dataset_train)
        elt_spec = dataset_train.element_spec[0]
        batch_sorted_dict = BatchSort(elt_spec.shape[1:], (1, k), dtype=elt_spec.dtype)

        for batch in dataset_train:
            influence_values = self.compute_pairwise_influence_value(batch)
            batch_sorted_dict.add_all(tf.expand_dims(batch[0], axis=0), tf.transpose(influence_values))

        best_samples, best_values = batch_sorted_dict.get()
        influence_values = tf.stack(best_values)
        training_samples = tf.concat(
            [tf.expand_dims(v, axis=0) for v in best_samples], axis=0
        )
        training_samples, influence_values = tf.squeeze(training_samples, axis=0), tf.squeeze(influence_values, axis=0)
        return training_samples, influence_values


    @abstractmethod
    def compute_influence_values_from_tensor(
            self,
            train_samples: Tuple[tf.Tensor, ...],
            samples_to_evaluate: Tuple[tf.Tensor, ...]
    ) -> tf.Tensor:
        """
        #TODO: simplify the name e.g. evaluate_influence_values_from_batch
        #TODO: in the doc it say VECTOR but the function name is VALUE
        Compute the influence vector between a single batch of samples to evaluate and a single batch of
        training samples

        Parameters
        ----------
        train_samples
            Training sample and its target value
        samples_to_evaluate
            Sample to evaluate
        Returns
        -------
        The influence score
        """
        raise NotImplementedError()


    @abstractmethod
    def compute_influence_values_for_sample_to_evaluate(
            self,
            dataset_train: tf.data.Dataset,
            samples_to_evaluate: Tuple[tf.Tensor, ...],
            load_influence_vector_ds_path: str = None,
            save_influence_vector_ds_path: str = None) -> tf.data.Dataset:
        """
        #TODO: Clarify what that means
        Compute the influence score between samples to evaluate and training dataset

        Parameters
        ----------
        dataset_train
            The training dataset
        sample_to_evaluate
            A batched tensor containing the samples which will be compare to the training dataset
        load_or_save_train_influence_vector_path
            The path to save or load the influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path
        Returns
        -------
        A dataset containing the tuple:
            batch of the training dataset
            influence score
        """
        raise NotImplementedError()


    @abstractmethod
    def compute_influence_values_for_dataset_to_evaluate(self, dataset_to_evaluate: tf.data.Dataset,
                                                         dataset_train: tf.data.Dataset,
                                                         vector_influence_in_cache: bool = True,
                                                         load_influence_vector_path: str = None,
                                                         save_influence_vector_path: str = None,
                                                         save_influence_value_path: str = None
                                                         ) -> tf.data.Dataset:
        """
        Compute the influence score between a dataset of samples to evaluate and the training dataset

        Parameters
        ----------
        dataset_to_evaluate
            the dataset containing the sample to evaluate
        dataset_train
            The training dataset
        vector_influence_in_cache
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
        raise NotImplementedError()


    @abstractmethod
    def top_k(self,
              sample_to_evaluate: Tuple[tf.Tensor, ...],
              dataset_train: tf.data.Dataset,
              k: int = 5,
              nearest_neighbor: BaseNearestNeighbor = LinearNearestNeighbor(),
              d_type: tf.DType = tf.float32,
              load_influence_vector_ds_path=None,
              save_influence_vector_ds_path=None) -> Tuple[
                Tuple[tf.Tensor, ...], tf.Tensor, Tuple[tf.Tensor, ...]]:
        """
        #TODO: It is like the substep of the following function ?
        Find the top-k closest elements of the training dataset for each sample to evaluate

        Parameters
        ----------
        sample_to_evaluate
            A batched tensor containing the samples which will be compare to the training dataset
        dataset_train
            The dataset used to train the model.
        k
            the number of most influence samples to retain in training datatse
        nearest_neighbor
            The nearest neighbor method. The default method is a linear search
        load_or_save_train_influence_vector_path
            The path to save or load the influence vector of the training dataset. If specify, load the dataset
            if already computed or compute the influence vector and then saved in the specific path
        Returns
        -------
        influence_values
            Top-k influence values for each sample to evaluate.
        training_samples
            Top-k training sample for each sample to evaluate.
        """
        raise NotImplementedError()


    @abstractmethod
    def top_k_dataset(self, dataset_to_evaluate: tf.data.Dataset, dataset_train: tf.data.Dataset,
                            k: int = 5,
                            nearest_neighbor: BaseNearestNeighbor = LinearNearestNeighbor(),
                            d_type: tf.DType = tf.float32,
                            vector_influence_in_cache: bool = True,
                            load_influence_vector_ds_path: str = None,
                            save_influence_vector_ds_path: str = None,
                            save_top_k_ds_path: str = None) -> tf.data.Dataset:
        """
        #TODO: Builds a dataset ((batch_samples_to_evaluate), influence VECTOR OR VALUE)
        Find the top-k closest elements for each element of dataset to evaluate in the training dataset
        The method will return a dataset containing a tuple of:
            (Top-k influence values for each sample to evaluate, Top-k training sample for each sample to evaluate)

        Parameters
        ----------
        dataset_to_evaluate
            The dataset which contains the samples which will be compare to the training dataset
        dataset_train
            The dataset used to train the model.
        k
            the number of most influence samples to retain in training dataset
        nearest_neighbor
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
        Save a dataset in the specified path

        Parameters
        ----------
        dataset
            The dataset to save
        load_or_save_path
            The path to save the dataset
        Returns
        -------
        None
        """
        tf.data.experimental.save(dataset, load_or_save_path)


    def load_dataset(self, dataset_path):
        """
        TODO: Docs
        """
        if path.exists(dataset_path):
            dataset = tf.data.experimental.load(dataset_path)
        else:
            raise NotFoundErr(f"The dataset path: {dataset_path} was not found")
        return dataset


class VectorBasedInfluenceCalculator(BaseInfluenceCalculator):
    """
    #TODO: Is it really base on influence vector compared to the previous abstract class ? Or is it focusing on IHVP
    #TODO: operations ?
    #TODO: Is it always related to IHVP or not ? Otherwise, we should make a Third class IHVPBasedInfluenceCalculator ?
    The abstraction of influence calculator based on influence vector
    """

    @abstractmethod
    def compute_influence_vector(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
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
    def preprocess_sample_to_evaluate(self, samples_to_evaluate: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Preprocess a sample to evaluate

        Parameters
        ----------
        samples_to_evaluate
            sample to evaluate
        Returns
        -------
        The preprocessed sample to evaluate
        """
        raise NotImplementedError()


    @abstractmethod
    def compute_influence_value_from_influence_vector(self, preproc_sample_to_evaluate,
                                                      influence_vector: tf.Tensor) -> tf.Tensor:
        """
        Compute the influence score for a preprocessed sample to evaluate and a training influence VECTOR

        Parameters
        ----------
        preproc_sample_to_evaluate
            Preprocessed sample to evaluate
        influence_vector
            Training influence Vvctor
        Returns
        -------
        The influence score
        """
        raise NotImplementedError()


    def compute_influence_vector_dataset(
        self,
        dataset_train: tf.data.Dataset,
        save_influence_vector_ds_path: str = None) -> tf.data.Dataset:
        """
        Compute the influence vector for each sample of the training dataset

        Parameters
        ----------
        dataset_train
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
        #TODO: Improve by saving only the influence vector
        inf_vect_ds = dataset_train.map(lambda *batch: (batch, self.compute_influence_vector(batch)))
        if save_influence_vector_ds_path is not None:
            # self.save_dataset(inf_vect_ds, save_influence_vector_ds_path)
            inf_vect = inf_vect_ds.map(lambda *batch: batch[-1])
            self.save_dataset(inf_vect.unbatch(), save_influence_vector_ds_path)
        return inf_vect_ds


    @tf.function
    def compute_influence_values_from_tensor(
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
        v_train = self.compute_influence_vector(train_samples)
        v_to_evaluate = self.preprocess_sample_to_evaluate(samples_to_evaluate)
        influence_values = self.compute_influence_value_from_influence_vector(v_to_evaluate, v_train)

        return influence_values


    def compute_influence_values_for_sample_to_evaluate(
            self,
            dataset_train: tf.data.Dataset,
            samples_to_evaluate: Tuple[tf.Tensor, ...],
            load_influence_vector_ds_path: str = None,
            save_influence_vector_ds_path: str = None) -> tf.data.Dataset:
        """
        Compute the influence score between samples to evaluate and training dataset

        Parameters
        ----------
        dataset_train
            The training dataset
        sample_to_evaluate
            A batched tensor containing the samples which will be compare to the training dataset
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
            batch_size = dataset_train._batch_size # pylint: disable=W0212
            inf_vect_ds = tf.data.Dataset.zip((dataset_train.unbatch(), inf_vect_ds)).batch(batch_size)
            if save_influence_vector_ds_path is not None:
                warn("Since you loaded the inf_vect_ds we ignore the saving option (useless)")
        else:
            inf_vect_ds = self.compute_influence_vector_dataset(dataset_train, save_influence_vector_ds_path)

        return self._compute_inf_values_with_inf_vect_dataset(inf_vect_ds, samples_to_evaluate)


    def _compute_inf_values_with_inf_vect_dataset(
            self,
            inf_vect_dataset: tf.data.Dataset,
            samples_to_evaluate: Tuple[tf.Tensor, ...]) -> tf.data.Dataset:
        """
        Compute the influence score between samples to evaluate and vector of influence of training dataset

        Parameters
        ----------
        ihvp_dataset
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
            lambda *batch: (batch[:-1][0],
                            self._compute_influence_values_from_influence_vector(samples_to_evaluate, batch[-1])
                                    )
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
            #TODO: Fill that
        ihvp
            The influence vector
        Returns
        -------
        Tuple:
            batch of the training dataset
            influence vector
        """
        v_to_evaluate = self.preprocess_sample_to_evaluate(samples_to_evaluate)
        value = self.compute_influence_value_from_influence_vector(v_to_evaluate, inf_vect)
        return value


    def compute_influence_values_for_dataset_to_evaluate(self, dataset_to_evaluate: tf.data.Dataset,
                                                         dataset_train: tf.data.Dataset,
                                                         vector_influence_in_cache: bool = True,
                                                         load_influence_vector_path: str = None,
                                                         save_influence_vector_path: str = None,
                                                         save_influence_value_path: str = None
                                                         ) -> tf.data.Dataset:
        """
        Compute the influence score between a dataset of samples to evaluate and the training dataset

        Parameters
        ----------
        dataset_to_evaluate
            the dataset containing the sample to evaluate
        dataset_train
            The training dataset
        vector_influence_in_cache
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

        if not vector_influence_in_cache and load_influence_vector_path is None:
            warn("Warning: The computation is not efficient, thinks to use cache or disk save")

        if vector_influence_in_cache:
            # TODO: Question the intended behavior here, is it save or load ?
            load_influence_vector_path = None

        if load_influence_vector_path is not None:
            inf_vect_ds = self.load_dataset(load_influence_vector_path)
            batch_size = dataset_train._batch_size # pylint: disable=W0212
            inf_vect_ds = tf.data.Dataset.zip((dataset_train.unbatch(), inf_vect_ds)).batch(batch_size)
        else:
            inf_vect_ds = self.compute_influence_vector_dataset(dataset_train,
                                                              save_influence_vector_path)

        if vector_influence_in_cache:
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

    def top_k(self,
              sample_to_evaluate: Tuple[tf.Tensor, ...],
              dataset_train: tf.data.Dataset,
              k: int = 5,
              nearest_neighbor: BaseNearestNeighbor = LinearNearestNeighbor(),
              d_type: tf.DType = tf.float32,
              load_influence_vector_ds_path=None,
              save_influence_vector_ds_path=None) -> Tuple[
                Tuple[tf.Tensor, ...], tf.Tensor, Tuple[tf.Tensor, ...]]:
        """
        Find the top-k closest elements of the training dataset for each sample to evaluate

        Parameters
        ----------
        sample_to_evaluate
            A batched tensor containing the samples which will be compare to the training dataset
        dataset_train
            The dataset used to train the model.
        k
            the number of most influence samples to retain in training datatse
        nearest_neighbor
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
            batch_size = dataset_train._batch_size # pylint: disable=W0212
            inf_vect_ds = tf.data.Dataset.zip((dataset_train.unbatch(), inf_vect_ds)).batch(batch_size)
        else:
            inf_vect_ds = self.compute_influence_vector_dataset(dataset_train, save_influence_vector_ds_path)
        batch_size_eval = int(tf.shape(sample_to_evaluate[0])[0])

        nearest_neighbor.build(
            inf_vect_ds,
            self.compute_influence_value_from_influence_vector,
            k,
            query_batch_size=batch_size_eval,
            d_type = d_type
        )

        return self._top_k_with_inf_vect_dataset_train(sample_to_evaluate, nearest_neighbor, batch_size_eval)


    def top_k_dataset(self, dataset_to_evaluate: tf.data.Dataset, dataset_train: tf.data.Dataset,
                            k: int = 5,
                            nearest_neighbor: BaseNearestNeighbor = LinearNearestNeighbor(),
                            d_type: tf.DType = tf.float32,
                            vector_influence_in_cache: bool = True,
                            load_influence_vector_ds_path: str = None,
                            save_influence_vector_ds_path: str = None,
                            save_top_k_ds_path: str = None) -> tf.data.Dataset:
        """
        Find the top-k closest elements for each element of dataset to evaluate in the training dataset
        The method will return a dataset containing a tuple of:
            (Top-k influence values for each sample to evaluate, Top-k training sample for each sample to evaluate)

        Parameters
        ----------
        dataset_to_evaluate
            The dataset which contains the samples which will be compare to the training dataset
        dataset_train
            The dataset used to train the model.
        k
            the number of most influence samples to retain in training dataset
        nearest_neighbor
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
            batch_size = dataset_train._batch_size # pylint: disable=W0212
            inf_vect_ds = tf.data.Dataset.zip((dataset_train.unbatch(), inf_vect_ds)).batch(batch_size)
        else:
            inf_vect_ds = self.compute_influence_vector_dataset(dataset_train,
                                                              save_influence_vector_ds_path)

        if vector_influence_in_cache:
            inf_vect_ds = inf_vect_ds.cache()

        batch_size_eval = int(dataset_to_evaluate._batch_size) # pylint: disable=W0212
        nearest_neighbor.build(
            inf_vect_ds,
            self.compute_influence_value_from_influence_vector,
            k,
            query_batch_size=batch_size_eval,
            d_type = d_type
        )

        top_k_dataset = dataset_to_evaluate.map(
            lambda *batch_evaluate: self._top_k_with_inf_vect_dataset_train(
                batch_evaluate, nearest_neighbor, batch_size_eval
            )
        )

        if save_top_k_ds_path is not None:
            self.save_dataset(top_k_dataset, save_top_k_ds_path)

        return top_k_dataset

    def _top_k_with_inf_vect_dataset_train(self,
                                        sample_to_evaluate: Tuple[tf.Tensor, ...],
                                        nearest_neighbor: BaseNearestNeighbor,
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
        v_to_evaluate = self.preprocess_sample_to_evaluate(sample_to_evaluate)

        if batch_size_eval is None:
            influences_values, training_samples = nearest_neighbor.query(v_to_evaluate)
        else:
            influences_values, training_samples = nearest_neighbor.query(v_to_evaluate, batch_size_eval)

        return sample_to_evaluate, influences_values, training_samples