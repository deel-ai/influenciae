from sortedcontainers import SortedDict
from operator import neg
import tensorflow as tf
from typing import Sequence, Tuple


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

    def add(self, key: any, value: any):
        """
            Add element in the dictionary and pop the lowest item if the dictionary is full
        :param key: the key to add to the dictionary
        :param value: the value to add to the dictionary
        :return:
        """
        self.sorted_dict[key] = value
        if self.size_maximum > 0 and len(self.sorted_dict) > self.size_maximum:
            self.sorted_dict.pop(self.sorted_dict.keys()[-1])

    def add_all(self, keys: Sequence, values: Sequence) -> None:
        """
            Add a list of keys and values in the dictionary
        :param keys: list of keys to add
        :param values: list of values to add
        """
        assert len(keys) == len(values)
        for k, v in zip(keys, values):
            self.add(k, v)

    def get_key_values(self) -> SortedDict:
        """
            Return the sorted dictionary
        :return: sorted dictionary
        """
        return self.sorted_dict


class BatchedSortedDict:
    """
        Manage a batch of sorted dictionary

        Parameters
        ----------
        batch_size: the number of dictionaries
        size_maximum: the number of elements to keep on each dictionaries
    """

    def __init__(self, batch_size: int, size_maximum: int = -1):
        self.batch_dict = []
        for _ in range(batch_size):
            self.batch_dict.append(MaximumSortedDict(size_maximum))

    def add_all(self, batch_key: tf.Tensor, batch_values: tf.Tensor) -> None:
        """
        Add a list of keys values in each each dictionary
        :param batch_key: a batch of list of keys
        :param batch_values: a batch of list of values
        :return: None
        """
        assert tf.shape(batch_key)[0] == tf.shape(batch_values)[0]
        assert tf.shape(batch_key)[0] == len(self.batch_dict)

        for k, v, d in zip(batch_key, batch_values, self.batch_dict):
            d.add_all(k.numpy(), v)

    def get(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Convert the dictionaries to a tensor of keys and a tensor of values
        :return: tensor of keys of all dictionary and tensor of values of all dictionary
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
