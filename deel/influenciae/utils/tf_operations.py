# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Custom operations related to tensorflow objects
"""

import tensorflow as tf

from ..types import Union, Tuple


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


def from_layer_name_to_layer_idx(model: tf.keras.Model, layer_name: str) -> int:
    """
    Finds the layer index of the corresponding layer name for the model

    Parameters
    ----------
    model
        Model on which to search
    layer_name
        Layer name

    Returns
    -------
    layer_idx
        The idx of the corresponding layer
    """
    for layer_idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return layer_idx
    raise ValueError(f'No such layer: {layer_name}. Existing layers are: '
                       f'{list(layer.name for layer in model.layers)}.')


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

    size = dataset.cardinality().numpy() * dataset._batch_size # pylint: disable=W0212
    return size


def default_process_batch(batch: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, tf.Tensor, Union[tf.Tensor, None]]:
    """
    A processing function to get the information into the right format for ingestion in the case of
    the default case where the batch is in the format (x, y) and there's no weight in a per-sample basis.

    Parameters
    ----------
    batch
        A tuple of tensors, where the information is in the format (x, y).

    Returns
    -------
    processed_batch
        A tuple containing explicitly the inputs, the targets and the per-sample weights (if present, None otherwise)
    """
    y_true = batch[1]
    model_inp = batch[0]
    sample_weight = None

    return model_inp, y_true, sample_weight
