# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module containing the implementation of a SortedDictionary, useful for computing the
top-k most influential samples in a large dataset.
"""
from enum import Enum

import tensorflow as tf
import numpy as np


class ORDER(Enum):
    """
    Enumeration for the two types of ordering for the sorting function.
    ASCENDING puts the elements with the smallest value first.
    DESCENDING puts the elements with the largest value first.
    """
    ASCENDING = 1
    DESCENDING = 2


class BatchSort:
    """
    An implementation of a SortedDictionary that accepts batches of elements and their corresponding scores
    and keeps only the k most/least important ones.

    Attributes
    ----------
    batch_shape
        A tuple with the shape of the dictionary's inputs.
    k_shape
        A tuple with the shape of the elements to keep.
    dtype
        The data-type of the input's values. By default, float32 will be used.
    order
        Either ASCENDING or DESCENDING depending on whether the most or the least important elements are to
        be kept.
    """
    def __init__(self, batch_shape, k_shape, dtype=tf.float32, order: ORDER = ORDER.DESCENDING):
        self.k = k_shape[1]
        shape = tf.concat((k_shape, batch_shape), axis=0)
        self.best_batch = tf.Variable(tf.zeros(shape, dtype=dtype), trainable=False)
        self.order = order
        if self.order == ORDER.DESCENDING:
            self.best_values = tf.Variable(tf.ones(k_shape, dtype=dtype) * (-np.inf), trainable=False)
        else:
            self.best_values = tf.Variable(tf.ones(k_shape, dtype=dtype) * np.inf, trainable=False)

    def add_all(self, batch_key: tf.Tensor, batch_values: tf.Tensor) -> None:
        """
        Add a new batch of data (element and values) and update the sorted dictionary retaining only the
        top/bottom-k elements.

        Parameters
        ----------
        batch_key
            A batch of new elements in the form of a tensor.
        batch_values
            A batch of their corresponding values in the form of a tensor.
        """
        current_score = tf.concat([self.best_values, batch_values], axis=1)
        current_batch = tf.concat([self.best_batch, batch_key], axis=1)

        if self.order == ORDER.DESCENDING:
            indexes = tf.argsort(current_score, axis=1, direction='DESCENDING')
        else:
            indexes = tf.argsort(current_score, axis=1, direction='ASCENDING')

        indexes = indexes[:, :self.k]

        current_best_score = tf.gather(current_score, indexes, batch_dims=1)
        current_best_samples = tf.gather(current_batch, indexes, batch_dims=1)

        self.best_values.assign(current_best_score)
        self.best_batch.assign(current_best_samples)

    def get(self):
        """
        A getter method for the top-k elements and their corresponding scores.

        Returns
        -------
        top_k_tuple
            A tuple with the top_k elements and their scores.
        """
        return self.best_batch, self.best_values

    def reset(self):
        """
        Resets the values of the whole object and prepares it to start sorting from scratch.
        """
        self.best_batch.assign(tf.zeros_like(self.best_batch))

        if self.order == ORDER.DESCENDING:
            self.best_values.assign((-np.inf) * tf.ones_like(self.best_values))
        else:
            self.best_values.assign(np.inf * tf.ones_like(self.best_values))
