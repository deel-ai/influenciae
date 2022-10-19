# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a Nearest Neighbors interface and a Linear Nearest Neighbors
algorithm. It will prove itself useful for finding the top-k most influential
examples of datasets, as implemented in the influence calculator interface.
"""
from abc import abstractmethod

import tensorflow as tf

from .sorted_dict import BatchSort, ORDER
from ..types import Callable, Optional, Tuple


class BaseNearestNeighbors:
    """
    A Nearest Neighbors interface for efficiently searching for specific data-points
    in a dataset.
    """

    @abstractmethod
    def build(
        self,
        dataset: tf.data.Dataset,
        dot_product_fun: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        k: int,
        query_batch_size: int,
        d_type: tf.DType = tf.float32,
        order: ORDER = ORDER.DESCENDING
    ) -> None:
        """
        Builds the nearest neighbors object that will be used to find the k neighbors
        inside the provided dataset.

        Parameters
        ----------
        dataset
            A TF dataset containing the points which shall be indexed.
        dot_product_fun
            The dot product function used to compute the distance between 2 points
        k
            An integer for the amount of samples to search for
        query_batch_size
            An integer for the query's batch size
        d_type
            The dataset's element's data-type
        order
            Either descending or ascending for the top or bottom results as per the similarity metric
        """
        raise NotImplementedError()

    @abstractmethod
    def query(self, vector_to_find: tf.Tensor, batch_size: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
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

    Attributes
    ----------
    dataset
        A TF dataset with the points from which the nearest neighbors will be searched.
    dot_product_fun
        A callable taking in two vectors and returning a notion of similarity between them.
    batched_sorted_dict
        A BatchSort instance that takes care of keeping the top/bottom most similar examples from
        the batch.
    """

    def __init__(self):
        self.dataset = None
        self.dot_product_fun = None
        self.batched_sorted_dict = None

    def build(
        self,
        dataset: tf.data.Dataset,
        dot_product_fun: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        k: int,
        query_batch_size: int,
        d_type: tf.DType = tf.float32,
        order: ORDER = ORDER.DESCENDING
        ) -> None:
        """
        Builds the linear nearest neighbors object that will be used to find the k neighbors
        inside the provided dataset.

        Parameters
        ----------
        dataset
            A TF dataset containing the points which shall be indexed.
        dot_product_fun
            The dot product function used to compute the distance between 2 points
        k
            An integer for the amount of samples to search for
        query_batch_size
            An integer for the query's batch size
        d_type
            The dataset's element's data-type
        order
            Either descending or ascending for the top or bottom results as per the similarity metric
        """
        self.dataset = dataset
        self.dot_product_fun = dot_product_fun
        batch_shape = self.dataset.element_spec[0][0].shape[1:]
        self.batched_sorted_dict = BatchSort(batch_shape, [query_batch_size, k], dtype=d_type, order=order)

    def query(self, vector_to_find: tf.Tensor, batch_size: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
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
            batch_size = tf.shape(vector_to_find)[0]

        dataset_iterator = iter(self.dataset)
        self.batched_sorted_dict.reset()

        def body_func(i):
            batch, ihvp = next(dataset_iterator)
            influence_values = self.dot_product_fun(vector_to_find, ihvp)

            self.batched_sorted_dict.add_all(
                tf.repeat(tf.expand_dims(batch[0], axis=0), batch_size, axis=0),
                influence_values
            )

            return (i+1, )

        tf.while_loop(
            cond=lambda i: i < self.dataset.cardinality(),
            body=body_func,
            loop_vars=[tf.constant(0, dtype=tf.int64)]
        )

        training_samples, influences_values = self.batched_sorted_dict.get()

        return influences_values, training_samples
