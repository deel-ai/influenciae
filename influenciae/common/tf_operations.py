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
