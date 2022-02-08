"""
Custom operations related to tensorflow objects
"""

import tensorflow as tf

from ..types import Union


def find_layer(model: tf.keras.Model, layer: Union[str, int]) -> tf.keras.layers.Layer:
    """
    Finds a layer in a model either by its name or by its index.

    Parameters
    ----------
    model
        Model on which to search.
    layer
        Layer name or layer index

    Returns
    -------
    layer
        Layer found
    """
    if isinstance(layer, str):
        return model.get_layer(layer)
    if isinstance(layer, int):
        return model.layers[layer]
    raise ValueError(f"Could not find any layer {layer}.")


def is_dataset_batched(dataset: tf.data.Dataset) -> Union[int, bool]:
    """
    Ensure the dataset is batched, if true return the batch_size else false.

    Parameters
    ----------
    dataset
        Tensorflow dataset to check.

    Returns
    -------
    batch_size
        False if the dataset if not batched else the batch_size of the dataset.
    """
    if hasattr(dataset, '_batch_size'):
        return dataset._batch_size # pylint: disable=W0212

    return False


def assert_batched_dataset(dataset: tf.data.Dataset):
    """
    Throw an error if the dataset is not batched.

    Parameters
    ----------
    dataset
        Tensorflow dataset to check.
    """
    if not is_dataset_batched(dataset):
        raise ValueError("The dataset must be batched before performing this operation.")


def dataset_size(dataset: tf.data.Dataset):
    """
    Compute the size of a batched tensorflow dataset: batch_size * number of batchs

    Parameters
    ----------
    dataset
        Tensorflow dataset to check.

    Returns
    -------
    size
        Number of points in the dataset
    """
    assert_batched_dataset(dataset)

    size = dataset.cardinality().numpy() * dataset._batch_size
    return size
