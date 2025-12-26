# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TensorFlow backend implementation.
"""
from typing import Any, List, Tuple, Callable, Optional

import numpy as np
import tensorflow as tf

from .backend import BaseBackend, Framework


class TensorFlowBackend(BaseBackend):
    """
    TensorFlow-specific backend implementation.
    """

    @property
    def framework(self) -> Framework:
        return Framework.TENSORFLOW

    def get_model_weights(self, model: tf.keras.Model, layers: Optional[List[tf.keras.layers.Layer]] = None) -> List[tf.Variable]:
        """
        Get trainable weights from a Keras model.

        Parameters
        ----------
        model
            The Keras model to extract weights from.
        layers
            Optional list of specific layers to get weights from.
            If None, gets weights from all layers.

        Returns
        -------
        weights
            List of weight tensors (tf.Variable).
        """
        if layers is None:
            layers = model.layers

        weights = []
        for layer in layers:
            if hasattr(layer, 'weights') and layer.weights:
                weights.extend(layer.weights)
        return weights

    def get_num_params(self, weights: List[tf.Variable]) -> int:
        """Get the total number of parameters."""
        return int(tf.reduce_sum([tf.size(w) for w in weights]).numpy())

    def compute_loss(
        self,
        model: tf.keras.Model,
        loss_function: Callable,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Compute the loss for a batch of samples."""
        predictions = model(inputs)
        if sample_weight is not None:
            return loss_function(targets, predictions, sample_weight)
        return loss_function(targets, predictions)

    @tf.function
    def compute_jacobian(
        self,
        model: tf.keras.Model,
        weights: List[tf.Variable],
        loss_function: Callable,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Compute the Jacobian of the loss with respect to weights."""
        batch_size = tf.shape(targets)[0]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(weights)
            predictions = model(inputs)
            if sample_weight is not None:
                loss = loss_function(targets, predictions, sample_weight)
            else:
                loss = loss_function(targets, predictions)

        jacobian = tape.jacobian(loss, weights)

        # Flatten and concatenate
        jacobian = [tf.reshape(j, (batch_size, -1)) for j in jacobian]
        jacobian = tf.concat(jacobian, axis=1)

        return jacobian

    @tf.function
    def compute_gradient(
        self,
        model: tf.keras.Model,
        weights: List[tf.Variable],
        loss_function: Callable,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Compute the gradient of the loss with respect to weights."""
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(weights)
            predictions = model(inputs)
            if sample_weight is not None:
                loss = tf.expand_dims(loss_function(targets, predictions, sample_weight), axis=-1)
            else:
                loss = tf.expand_dims(loss_function(targets, predictions), axis=-1)

        gradients = tape.gradient(loss, weights)

        # Flatten and concatenate
        gradients = [tf.reshape(g, (-1,)) for g in gradients]
        gradients = tf.concat(gradients, axis=0)

        return gradients

    def concat(self, tensors: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
        """Concatenate tensors along an axis."""
        return tf.concat(tensors, axis=axis)

    def stack(self, tensors: List[tf.Tensor], axis: int = 0) -> tf.Tensor:
        """Stack tensors along a new axis."""
        return tf.stack(tensors, axis=axis)

    def reshape(self, tensor: tf.Tensor, shape: Tuple[int, ...]) -> tf.Tensor:
        """Reshape a tensor."""
        return tf.reshape(tensor, shape)

    def to_numpy(self, tensor: tf.Tensor) -> np.ndarray:
        """Convert a tensor to numpy array."""
        return tensor.numpy()

    def get_batch_size(self, tensor: tf.Tensor) -> int:
        """Get the batch size (first dimension) of a tensor."""
        return tf.shape(tensor)[0]

    def reduce_sum(self, tensor: tf.Tensor, axis: Optional[int] = None) -> tf.Tensor:
        """Reduce sum along an axis."""
        return tf.reduce_sum(tensor, axis=axis)

    def expand_dims(self, tensor: tf.Tensor, axis: int) -> tf.Tensor:
        """Add a new axis to a tensor."""
        return tf.expand_dims(tensor, axis=axis)

    def squeeze(self, tensor: tf.Tensor, axis: Optional[int] = None) -> tf.Tensor:
        """Remove dimensions of size 1."""
        if axis is None:
            return tf.squeeze(tensor)
        return tf.squeeze(tensor, axis=axis)

    def transpose(self, tensor: tf.Tensor) -> tf.Tensor:
        """Transpose a tensor (swap last two dimensions)."""
        return tf.transpose(tensor)

    def tensor_shape(self, tensor: tf.Tensor) -> Tuple[int, ...]:
        """Get the shape of a tensor."""
        return tuple(tensor.shape.as_list())

    def tensor_ndim(self, tensor: tf.Tensor) -> int:
        """Get the number of dimensions of a tensor."""
        return len(tensor.shape)

    def matmul(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Matrix multiplication."""
        return tf.matmul(a, b)

    def find_layer_by_name(self, model: tf.keras.Model, layer_name: str) -> Tuple[int, tf.keras.layers.Layer]:
        """Find a layer by name and return its index and the layer."""
        for layer_idx, layer in enumerate(model.layers):
            if layer.name == layer_name:
                return layer_idx, layer
        raise ValueError(
            f'No such layer: {layer_name}. Existing layers are: '
            f'{[layer.name for layer in model.layers]}.'
        )

    def get_layers(self, model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
        """Get all layers from a model."""
        return model.layers

    def forward(self, model: tf.keras.Model, inputs: tf.Tensor) -> tf.Tensor:
        """Run forward pass on a model."""
        return model(inputs)

    def find_last_weight_layer(self, model: tf.keras.Model) -> int:
        """
        Find and return the id of the last layer with weights.

        Parameters
        ----------
        model
            The Keras model.

        Returns
        -------
        layer_id
            Id (e.g. -1, -2...) of the layer found.
        """
        num_layers = len(model.layers)
        # Start from -1 (the last layer) and work backwards
        for layer_id in range(1, num_layers + 1):
            layer = model.layers[-layer_id]
            if hasattr(layer, 'weights') and layer.weights:
                return -layer_id
        raise ValueError('No layers with weights found for the model.')

    def get_layer_index(self, model: tf.keras.Model, layer: Any) -> int:
        """
        Get the index of a layer in a model.

        Parameters
        ----------
        model
            The Keras model.
        layer
            Layer name (str), index (int), or layer object.

        Returns
        -------
        layer_idx
            Index of the layer.
        """
        if layer is None:
            return self.find_last_weight_layer(model) + len(model.layers)
        elif isinstance(layer, str):
            idx, _ = self.find_layer_by_name(model, layer)
            return idx
        elif isinstance(layer, int):
            if layer < 0:
                return layer + len(model.layers)
            return layer
        else:
            raise ValueError(f"layer should be None, a string, or an int, got {type(layer)}")

    def get_weights_for_layer_range(
        self,
        model: tf.keras.Model,
        start_layer: Optional[Any] = None,
        last_layer: Optional[Any] = None
    ) -> List[tf.Variable]:
        """
        Get weights for a range of layers.

        Parameters
        ----------
        model
            The Keras model.
        start_layer
            Starting layer (name, index, or None for auto-detect).
        last_layer
            Ending layer (name, index, or None for just start_layer).

        Returns
        -------
        weights
            List of weight tensors.
        """
        start_idx = self.get_layer_index(model, start_layer)

        if last_layer is None:
            layers = [model.layers[start_idx]]
        else:
            end_idx = self.get_layer_index(model, last_layer)
            assert end_idx >= start_idx, \
                f"last_layer index ({end_idx}) should be >= start_layer index ({start_idx})"
            layers = model.layers[start_idx:end_idx + 1]

        return self.get_model_weights(model, layers)

    # Dataset operations
    def map_dataset(
        self,
        dataset: tf.data.Dataset,
        map_fn: Callable,
        device: Optional[str] = None
    ) -> tf.data.Dataset:
        """Apply a mapping function to each batch in a dataset."""
        if device is not None:
            def device_map_fn(*args):
                with tf.device(device):
                    return map_fn(*args)
            return dataset.map(device_map_fn)
        return dataset.map(map_fn)

    def cache_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Cache a dataset in memory."""
        return dataset.cache()

    def save_dataset(self, dataset: tf.data.Dataset, path: str) -> None:
        """Save a dataset to disk."""
        tf.data.experimental.save(dataset, path)

    def load_dataset(self, path: str) -> tf.data.Dataset:
        """Load a dataset from disk."""
        from os import path as os_path
        from xml.dom import NotFoundErr
        if os_path.exists(path):
            return tf.data.experimental.load(path)
        raise NotFoundErr(f"The dataset path: {path} was not found")

    def get_dataset_batch_size(self, dataset: tf.data.Dataset) -> int:
        """Get the batch size of a dataset."""
        return int(dataset._batch_size)  # pylint: disable=W0212

    def zip_datasets(
        self,
        dataset1: tf.data.Dataset,
        dataset2: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Zip two datasets together."""
        return tf.data.Dataset.zip((dataset1, dataset2))

    def batch_dataset(self, dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        """Batch a dataset."""
        return dataset.batch(batch_size)

    def unbatch_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Unbatch a dataset."""
        return dataset.unbatch()

    def get_dataset_element_spec(self, dataset: tf.data.Dataset) -> Any:
        """Get the element spec of a dataset."""
        return dataset.element_spec

    def assert_batched_dataset(self, dataset: tf.data.Dataset) -> None:
        """Assert that a dataset is batched."""
        from ..utils import assert_batched_dataset
        assert_batched_dataset(dataset)

