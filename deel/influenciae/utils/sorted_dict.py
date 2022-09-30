# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TODO
"""
from operator import neg
from sortedcontainers import SortedDict

import tensorflow as tf
import numpy as np

from ..types import Sequence, Tuple

class MaximumSortedDict:
    """
    Sorted Dictionary with a maximum size. Keep the items which have the highest keys

    Parameters
    ----------
    size_maximum: The size maximum of the dictionary
    """
    def __init__(self, size_maximum: int = -1):
        self.sorted_dict = SortedDict(neg)
        self.size_maximum = size_maximum

    def add(self, key: any, value: any) -> None:
        """
        Add element in the dictionary and pop the lowest item if the dictionary is full

        Parameters
        ----------
        key
            the key to add to the dictionary
        value
            the value to add to the dictionary

        Returns
        -------
            None
        """
        self.sorted_dict[key] = value
        if self.size_maximum > 0 and len(self.sorted_dict) > self.size_maximum:
            self.sorted_dict.pop(self.sorted_dict.keys()[-1])

    def add_all(self, keys: Sequence, values: Sequence) -> None:
        """
        Add a list of keys and values in the dictionary

        Parameters
        ----------
        keys
            list of keys to add
        values
            list of values to add

        Returns
        -------
            None
        """
        assert len(keys) == len(values)
        for k, v in zip(keys, values):
            self.add(k, v)

    def get_key_values(self) -> SortedDict:
        """
        Return the sorted dictionary

        Returns
        -------
            sorted dictionary
        """
        return self.sorted_dict


class BatchedSortedDict:
    """
    Manage a batch of sorted dictionary

    Parameters
    ----------
    batch_size
        the number of dictionaries
    size_maximum
        the number of elements to keep on each dictionaries
    """
    def __init__(self, batch_size: int, size_maximum: int = -1):
        self.batch_dict = []
        for _ in range(batch_size):
            self.batch_dict.append(MaximumSortedDict(size_maximum))

    def add_all(self, batch_key: tf.Tensor, batch_values: tf.Tensor) -> None:
        """
        Add a list of keys values in each each dictionary

        Parameters
        ----------
        batch_key
            a batch of list of keys
        batch_values
            a batch of list of values

        Returns
        -------
            None
        """
        assert tf.shape(batch_key)[0] == tf.shape(batch_values)[0]
        assert tf.shape(batch_key)[0] == len(self.batch_dict)

        for k, v, d in zip(batch_key, batch_values, self.batch_dict):
            d.add_all(k.numpy(), v)

    def get(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Convert the dictionaries to a tensor of keys and a tensor of values

        Returns
        -------
            tensor of keys of all dictionary and tensor of values of all dictionary
        """
        batch_keys = []
        batch_values = []
        for d in self.batch_dict:
            k = tf.stack(d.get_key_values().keys())
            v = tf.concat([tf.expand_dims(v, axis=0) for v in d.get_key_values().values()], axis=0)
            batch_keys.append((tf.expand_dims(k, axis=0)))
            batch_values.append((tf.expand_dims(v, axis=0)))

        batch_keys = tf.concat(batch_keys, axis=0)
        batch_values = tf.concat(batch_values, axis=0)
        return batch_keys, batch_values

class BatchSort:
    """
    #TODO: Add documentation
    """
    def __init__(self, batch_shape, k_shape, dtype=tf.float32):
        self.k = k_shape[1]
        shape = tf.concat((k_shape, batch_shape), axis=0)
        self.best_batch = tf.Variable(tf.zeros(shape, dtype=dtype), trainable=False)
        self.best_values = tf.Variable(tf.ones(k_shape, dtype=dtype) * (-np.inf), trainable=False)

    def add_all(self, batch_key: tf.Tensor, batch_values: tf.Tensor) -> None:
        """
        TODO
        """
        current_score = tf.concat([self.best_values, batch_values], axis=1)
        current_batch = tf.concat([self.best_batch, batch_key], axis=1)
        indexes = tf.argsort(current_score,axis=1,direction='DESCENDING')
        indexes = indexes[:, :self.k]

        current_best_score = tf.gather(current_score, indexes,batch_dims=1)
        current_best_samples = tf.gather(current_batch, indexes,batch_dims=1)

        self.best_values.assign(current_best_score)
        self.best_batch.assign(current_best_samples)

    def get(self):
        """
        TODO
        """
        return self.best_batch, self.best_values

    def reset(self):
        """
        TODO
        """
        self.best_batch.assign(tf.zeros_like(self.best_batch))
        self.best_values.assign((-np.inf) * tf.ones_like(self.best_values))
