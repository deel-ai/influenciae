# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TODO: Insert insightful description
"""
from abc import abstractmethod

import tensorflow as tf

from .sorted_dict import BatchSort
from ..types import Callable, Optional

#TODO: Update documentation
class BaseNearestNeighbor:
    """
    Nearest Neighbor abstract to search over a dataset
    """

    @abstractmethod
    def build(
        self,
        dataset: tf.data.Dataset,
        dot_product_fun: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        k: int,
        query_batch_size: int,
        d_type: tf.DType = tf.float32
    ) -> None:
        """
        Build the neighbor object which will be used to find the k neighbor among a dataset

        Parameters
        ----------
        dataset
            Dataset containing the points which shall be indexed
        dot_product_fun
            The dot product function used to compute the distance between 2 points
        Returns
        -------
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def query(self, vector_to_find: tf.Tensor, batch_size: Optional[int] = None):
        """
        Find the k closest points to the dataset

        Parameters
        ----------
        vector_to_find
            A tensor of points to query.
        k
            the number of nearest neighbors to return
        Returns
        -------
            distances
            k closets points
        """
        raise NotImplementedError()


class LinearNearestNeighbor(BaseNearestNeighbor):
    """
    Nearest Neighbor based on a linear search over a dataset
    """

    def __init__(self):
        self.dataset = None
        self.dot_product_fun = None
        self.batched_sorted_dic = None

    def build(
        self,
        dataset: tf.data.Dataset,
        dot_product_fun: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        k: int,
        query_batch_size: int,
        d_type: tf.DType = tf.float32
        ) -> None:
        """
        Build the neighbor object which will be used to find the k neighbor among a dataset

        Parameters
        ----------
        dataset
            Dataset containing the points which shall be indexed
        dot_product_fun
            The dot product function used to compute the distance between 2 points
        Returns
        -------
            None
        """
        self.dataset = dataset
        self.dot_product_fun = dot_product_fun
        batch_shape = self.dataset.element_spec[0][0].shape[1:]
        self.batched_sorted_dic = BatchSort(batch_shape, [query_batch_size, k], dtype=d_type)

    def query(self, vector_to_find: tf.Tensor, batch_size: Optional[int] = None):
        """
        Find the k closest points to the dataset

        Parameters
        ----------
        vector_to_find
            A tensor of points to query.
        k
            the number of nearest neighbors to return
        Returns
        -------
            distances
            k closets points
        """
        if batch_size is None:
            batch_size = tf.shape(vector_to_find)[0]

        dataset_iterator = iter(self.dataset)
        self.batched_sorted_dic.reset()

        def body_func(i):
            batch, ihvp = next(dataset_iterator)
            influence_values = self.dot_product_fun(vector_to_find, ihvp)

            self.batched_sorted_dic.add_all(
                tf.repeat(tf.expand_dims(batch[0], axis=0), batch_size, axis=0),
                influence_values
            )

            return (i+1, )

        tf.while_loop(
            cond=lambda i: i < self.dataset.cardinality(),
            body=body_func,
            loop_vars=[tf.constant(0, dtype=tf.int64)]
        )

        training_samples, influences_values = self.batched_sorted_dic.get()

        return influences_values, training_samples
