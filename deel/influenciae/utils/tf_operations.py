# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Custom operations related to tensorflow objects
"""

import numpy as np
import tensorflow as tf

from ..types import Union, Tuple, Optional, Callable


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
        return dataset._batch_size  # pylint: disable=W0212

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

    size = dataset.cardinality().numpy() * dataset._batch_size  # pylint: disable=W0212
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


def dataset_to_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    """
    Transforms a (batched) dataset into a tensor for easier manipulation.

    Parameters
    ----------
    dataset
        A batched TF dataset to transform into a tensor.

    Returns
    -------
    tensor
        A TF tensor with the evaluated dataset.
    """
    assert_batched_dataset(dataset)
    if isinstance(dataset.element_spec, Tuple):
        tensor = [list(dataset.map(lambda *w: w[i]).unbatch()) for i in range(len(dataset.element_spec))]  # pylint: disable=W0640
    else:
        tensor = tf.concat(list(dataset), axis=0)

    return tensor


def extract_only_values(dataset: tf.data.Dataset) -> tf.Tensor:
    """
    Utility function to extract all the influence values from a dataset containing tuples (data-point, influence score).
    The dataset much be batched before applying this operation.

    Parameters
    ----------
    dataset
        A batched TF dataset with tuples (data-point, influence score).

    Returns
    -------
    influence_values
        A tensor with only the dataset's influence values.
    """
    return dataset_to_tensor(dataset.map(lambda *inputs: inputs[1]))


def array_to_dataset(
        array: Union[np.ndarray, tf.Tensor, Tuple[np.ndarray, ...], Tuple[tf.Tensor, ...]],
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        buffer_size: int = 1000
) -> tf.data.Dataset:
    """
    Converts a dataset in the form of a numpy array or tensor (in tuples when it contains the labels) into a
    TF dataset, and if desired, shuffles and batches it for easier ingestion by other functions.

    Parameters
    ----------
    array
        Either a numpy array (or tensor) with the samples or a tuple with a set of samples and labels (or value
        associated to them).
    batch_size
        An optional integer indicating the batch size. If None, the dataset won't be batched
    shuffle
        A boolean indicating whether the dataset should be shuffled.
    buffer_size
        An integer for both the shuffle operation and computing the prefetch size (when batching).

    Returns
    -------
    dataset
        A TF dataset with the data in the array, possibly shuffled, batched and prefetched.
    """
    if isinstance(array, np.ndarray):
        array = tf.convert_to_tensor(array, dtype=tf.float32)
    elif isinstance(array, Tuple) and isinstance(array[0], np.ndarray):
        array = tf.stack(list(
            zip(tf.convert_to_tensor(array[0], dtype=tf.float32), tf.convert_to_tensor(array[1], dtype=tf.float32))
        ), axis=0)

    dataset = tf.data.Dataset.from_tensor_slices(array)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    if batch_size is not None:
        assert batch_size > 0
        assert buffer_size > 0
        dataset = dataset.batch(batch_size).prefetch(buffer_size // batch_size)

    return dataset


def get_device(device: Optional[str]) -> str:
    """
    Gets the name of the device to use. If there are any available GPUs, it will use the first one
    in the system, otherwise, it will use the CPU.

    Parameters
    ----------
    device
        A string specifying the device on which to run the computations. If None, it will search
        for available GPUs, and if none are found, it will return the first CPU.

    Returns
    -------
    device
        A string with the name of the device on which to run the computations.
    """
    if device is not None:
        return device

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices is None or len(physical_devices) == 0:
        return 'cpu:0'
    return 'GPU:0'


def map_to_device(dataset: tf.data.Dataset, map_fun: Callable, device: Optional[str] = None) -> tf.data.Dataset:
    """
    Performs a map function on the preferred device. If none is specified, the first available GPU is
    chosen, and if there aren't any, the computations are done on the CPU.

    Parameters
    ----------
    dataset
        The dataset on which to apply the map function.
    map_fun
        A callable with the function to be applied to the whole CPU.
    device
        An (optional) string with the device on which to perform the map function.
        If none is specified, the first available GPU is chosen, and if there aren't any, the
        computations are done on the CPU.

    Returns
    -------
    mapped_dataset
        The dataset with the map function applied to it.
    """
    device = get_device(device)

    def map_fun_device(*args):
        with tf.device(device):
            result = map_fun(*args)
        return result

    return dataset.map(map_fun_device)
