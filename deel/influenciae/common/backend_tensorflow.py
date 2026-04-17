# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TensorFlow backend implementation.
"""
# pylint: disable=too-many-lines
import os
from typing import Any, List, Tuple, Callable, Optional, Sequence, Iterable, cast

import numpy as np
import tensorflow as tf

from .backend import BaseBackend, Framework


class TensorFlowBackend(BaseBackend):  # pylint: disable=too-many-public-methods
    """
    TensorFlow-specific backend implementation.
    """

    @property
    def framework(self) -> Framework:
        return Framework.TENSORFLOW

    @staticmethod
    def _describe_weight(weight: Any, index: int) -> str:
        """Return a readable identifier for a watched weight."""
        weight_name = getattr(weight, 'name', None)
        if weight_name is None:
            return f"index {index}"
        return f"index {index} ({weight_name})"

    @staticmethod
    def _extract_watch_tensor(weight: Any) -> Optional[Any]:
        """Extract a watchable TensorFlow object from a weight container."""
        if isinstance(weight, (tf.Variable, tf.Tensor)):
            return weight

        tensor_candidate = None

        for attr_name in ('_variable', 'variable', '_value', 'value', 'handle'):
            attr_value = getattr(weight, attr_name, None)
            if callable(attr_value):
                try:
                    attr_value = attr_value()
                except TypeError:
                    continue
            if isinstance(attr_value, tf.Variable):
                return attr_value
            if isinstance(attr_value, tf.Tensor) and tensor_candidate is None:
                tensor_candidate = attr_value

            for nested_attr_name in ('_variable', 'variable', '_value', 'value', 'handle'):
                nested_value = getattr(attr_value, nested_attr_name, None)
                if callable(nested_value):
                    try:
                        nested_value = nested_value()
                    except TypeError:
                        continue
                if isinstance(nested_value, tf.Variable):
                    return nested_value
                if isinstance(nested_value, tf.Tensor) and tensor_candidate is None:
                    tensor_candidate = nested_value

        if tensor_candidate is not None:
            return tensor_candidate

        return None

    def normalize_weights_to_watch(self, weights: List[Any]) -> List[Any]:
        """Normalize watched weights to objects accepted by GradientTape."""
        normalized_weights: List[Any] = []
        for idx, weight in enumerate(tf.nest.flatten(weights)):
            watch_tensor = self._extract_watch_tensor(weight)
            if watch_tensor is None:
                raise TypeError(
                    "Could not watch weight "
                    f"{self._describe_weight(weight, idx)} of type {type(weight)}. "
                    "Expected tf.Variable/tf.Tensor or an object exposing one through "
                    "`value`, `_value`, or `variable`."
                )
            normalized_weights.append(watch_tensor)

        return normalized_weights

    def _raise_if_disconnected(
        self,
        operation_name: str,
        tensors: Sequence[Any],
        watched_weights: List[Any],
    ) -> None:
        """Raise an explicit error if any gradient/Jacobian component is missing."""
        missing_indices = [idx for idx, tensor in enumerate(tensors) if tensor is None]
        if not missing_indices:
            return

        missing_weights = ", ".join(
            self._describe_weight(watched_weights[idx], idx)
            for idx in missing_indices
        )
        raise ValueError(
            f"{operation_name} returned None for watched weight(s): {missing_weights}. "
            "This indicates a disconnected computation graph between the loss/output and these weights."
        )

    def get_model_weights(
        self,
        model: tf.keras.Model,
        layers: Optional[List[tf.keras.layers.Layer]] = None,
    ) -> List[tf.Variable]:
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
            return self.normalize_weights_to_watch(list(model.trainable_weights))

        weights = []
        for layer in layers:
            trainable_weights = getattr(layer, 'trainable_weights', None)
            if trainable_weights:
                weights.extend(self.normalize_weights_to_watch(list(trainable_weights)))
        return weights

    def clone_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Clone a Keras model and copy its weights."""
        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())
        return cloned_model

    def validate_loss_no_reduction(self, loss_function: Callable) -> None:
        """Validate that a TensorFlow loss function has no reduction."""
        loss_reduction = getattr(loss_function, 'reduction', None)
        if loss_reduction is not None and loss_reduction is not tf.keras.losses.Reduction.NONE:
            raise ValueError('The loss function must not have reduction (use Reduction.NONE).')

    def is_dense_linear_layer(self, layer: tf.keras.layers.Layer) -> bool:
        """Return whether a layer is a Dense layer."""
        return isinstance(layer, tf.keras.layers.Dense)

    def layer_has_bias(self, layer: tf.keras.layers.Layer) -> bool:
        """Return whether a layer uses a bias term."""
        return bool(getattr(layer, 'use_bias', False))

    def get_layer_io_features(self, layer: tf.keras.layers.Layer) -> Tuple[int, int]:
        """Return input/output feature sizes for a Dense layer."""
        if not self.is_dense_linear_layer(layer):
            raise ValueError(f"Expected a Dense layer, got {type(layer)}")

        kernel = getattr(layer, 'kernel', None)
        if kernel is not None:
            return int(kernel.shape[0]), int(kernel.shape[1])

        input_shape = getattr(layer, 'input_shape', None)
        if input_shape is None:
            raise ValueError("Could not infer Dense input/output features from an unbuilt layer")
        return int(input_shape[-1]), int(layer.units)

    def create_linear_model(
        self,
        input_shape: Tuple[int, ...],
        out_features: int,
        use_bias: bool = False,
        l2_regularization: float = 0.0,
        dtype: Optional[Any] = None,
        reference_weight: Optional[tf.Tensor] = None,
    ) -> tf.keras.Model:
        """Create a Keras model containing a single Dense layer."""
        if reference_weight is not None and dtype is None:
            dtype = reference_weight.dtype
        if dtype is None:
            dtype = tf.float32

        inputs = tf.keras.layers.Input(shape=input_shape, dtype=dtype)
        outputs = tf.keras.layers.Dense(
            out_features,
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.L2(l2_regularization),
            dtype=dtype,
        )(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_linear_weight_axes(self) -> Tuple[int, int]:
        """TensorFlow Dense kernels are shaped as (in_features, out_features)."""
        return 0, 1

    def get_num_params(self, weights: List[tf.Variable]) -> int:
        """Get the total number of parameters."""
        watched_weights = self.normalize_weights_to_watch(weights)
        return int(tf.reduce_sum([tf.size(w) for w in watched_weights]).numpy())

    def compute_loss(
        self,
        model: tf.keras.Model,
        loss_function: Callable,
        inputs: tf.Tensor,
        targets: Any,
        sample_weight: Optional[Any] = None
    ) -> tf.Tensor:
        """Compute the loss for a batch of samples."""
        predictions = model(inputs)
        if sample_weight is not None:
            return loss_function(targets, predictions, sample_weight)
        return loss_function(targets, predictions)

    def compute_jacobian(
        self,
        model: tf.keras.Model,
        weights: List[tf.Variable],
        loss_function: Callable,
        inputs: tf.Tensor,
        targets: Any,
        sample_weight: Optional[Any] = None
    ) -> tf.Tensor:
        """Compute the Jacobian of the loss with respect to weights."""
        watched_weights = self.normalize_weights_to_watch(weights)
        batch_size = tf.shape(inputs)[0]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(watched_weights)
            predictions = model(inputs)
            if sample_weight is not None:
                loss = loss_function(targets, predictions, sample_weight)
            else:
                loss = loss_function(targets, predictions)
            loss = self.ensure_per_sample_loss(loss)

        jacobian = tape.jacobian(loss, watched_weights)
        self._raise_if_disconnected('compute_jacobian', jacobian, watched_weights)

        # Flatten and concatenate
        jacobian = [tf.reshape(j, (batch_size, -1)) for j in jacobian]
        jacobian = tf.concat(jacobian, axis=1)

        return jacobian

    def compute_gradient(
        self,
        model: tf.keras.Model,
        weights: List[tf.Variable],
        loss_function: Callable,
        inputs: tf.Tensor,
        targets: Any,
        sample_weight: Optional[Any] = None
    ) -> tf.Tensor:
        """Compute the gradient of the loss with respect to weights."""
        watched_weights = self.normalize_weights_to_watch(weights)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(watched_weights)
            predictions = model(inputs)
            if sample_weight is not None:
                loss = loss_function(targets, predictions, sample_weight)
            else:
                loss = loss_function(targets, predictions)
            loss = self.ensure_per_sample_loss(loss)

        gradients = tape.gradient(loss, watched_weights)
        self._raise_if_disconnected('compute_gradient', gradients, watched_weights)

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

    def ensure_per_sample_loss(self, loss: tf.Tensor) -> tf.Tensor:
        """Ensure TensorFlow losses are represented as per-sample vectors."""
        loss_rank = loss.shape.rank

        if loss_rank == 0:
            raise ValueError("Loss function must return per-sample losses (reduction='none')")

        if loss_rank is None:
            # control dependencies to make sure that this executes at correct time in graph
            with tf.control_dependencies([
                tf.debugging.assert_rank_at_least(
                    loss,
                    1,
                    message="Loss function must return per-sample losses (reduction='none')",
                )
            ]):
                loss = tf.identity(loss)
            loss = tf.reshape(loss, (tf.shape(loss)[0], -1))
            return tf.reduce_sum(loss, axis=1)

        if loss_rank > 1:
            loss = tf.reshape(loss, (tf.shape(loss)[0], -1))
            loss = tf.reduce_sum(loss, axis=1)

        return loss

    def normalize_binary_targets(self, targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        """Normalize binary TensorFlow targets to match logits shape."""
        if logits.shape.rank == 2 and logits.shape[-1] == 1 and targets.shape.rank == 1:
            return tf.expand_dims(targets, axis=-1)
        return targets

    def reduce_sum(self, tensor: tf.Tensor, axis: Optional[int] = None, keepdims: bool = False) -> tf.Tensor:
        """Reduce sum along an axis."""
        return tf.reduce_sum(tensor, axis=axis, keepdims=keepdims)

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
        rank = tensor.shape.rank
        if rank is not None:
            if rank < 2:
                return tensor
            return tf.linalg.matrix_transpose(tensor)

        dynamic_rank = tf.rank(tensor)
        return tf.cond(
            dynamic_rank < 2,
            lambda: tensor,
            lambda: tf.linalg.matrix_transpose(tensor),
        )

    def tensor_shape(self, tensor: tf.Tensor) -> Tuple[int, ...]:
        """Get the shape of a tensor."""
        return tuple(tensor.shape.as_list())

    def tensor_ndim(self, tensor: tf.Tensor) -> int:
        """Get the number of dimensions of a tensor."""
        return len(tensor.shape)

    def matmul(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Matrix multiplication."""
        return tf.matmul(a, b)

    def multiply(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Element-wise multiplication."""
        return tf.math.multiply(a, b)

    def abs(self, tensor: tf.Tensor) -> tf.Tensor:
        """Compute absolute value of a tensor."""
        return tf.abs(tensor)

    def argmax(self, tensor: tf.Tensor, axis: int) -> tf.Tensor:
        """Return indices of maximum values along an axis."""
        return tf.argmax(tensor, axis=axis)

    def gather_along_axis(
        self,
        tensor: tf.Tensor,
        indices: tf.Tensor,
        axis: int,
        batch_dims: int = 0
    ) -> tf.Tensor:
        """Gather values from tensor along an axis using indices."""
        return tf.gather(tensor, indices, axis=axis, batch_dims=batch_dims)

    def get_output_shape(self, model: tf.keras.Model) -> Tuple[Optional[int], ...]:
        """Get the output shape of a model."""
        return tuple(model.output_shape)

    def split_model(
        self,
        model: tf.keras.Model,
        target_layer: Any
    ) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Split a model into two sub-models at a target layer.

        Parameters
        ----------
        model
            The Keras model to split.
        target_layer
            Layer name (str) or index (int) at which to split.

        Returns
        -------
        feature_extractor
            Model containing layers up to (but not including) target_layer.
        head
            Model containing the target_layer and beyond.
        """
        # Clone the model to avoid modifying the original
        cloned_model = self.clone_model(model)

        # Find the cut layer
        if isinstance(target_layer, str):
            cut_layer = cloned_model.get_layer(target_layer)
        elif isinstance(target_layer, int):
            cut_layer = cloned_model.layers[target_layer]
        else:
            raise ValueError(f"Could not find any layer {target_layer}.")

        # Create the feature extractor (up to but not including target layer)
        feature_extractor = tf.keras.Model(
            inputs=cloned_model.inputs,
            outputs=cut_layer.input
        )

        # Create the head (from target layer onwards)
        head = tf.keras.Model(
            inputs=cut_layer.input,
            outputs=cloned_model.outputs
        )

        return feature_extractor, head

    def normalize(self, tensor: tf.Tensor, axis: Optional[int] = None, keepdims: bool = False) -> tf.Tensor:
        """Normalize a tensor along an axis using L2 norm."""
        return tensor / tf.norm(tensor, axis=axis, keepdims=keepdims)

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

    def get_named_layers(
        self,
        model: tf.keras.Model,
        recursive: bool = False,
    ) -> List[Tuple[str, tf.keras.layers.Layer]]:
        """Get named top-level or recursively discovered Keras layers."""
        if not recursive:
            layers = model.layers
            return [(str(layer.name), layer) for layer in layers]

        def _collect_from_layers_attr(root: tf.keras.layers.Layer) -> List[Tuple[str, tf.keras.layers.Layer]]:
            """Recursively collect layers through the public ``layers`` attribute."""
            collected: List[Tuple[str, tf.keras.layers.Layer]] = []
            visited_containers = set()

            def _walk(container: tf.keras.layers.Layer, prefix: str) -> None:
                container_id = id(container)
                if container_id in visited_containers:
                    return
                visited_containers.add(container_id)

                children = list(getattr(container, "layers", []) or [])
                for child in children:
                    child_name = str(getattr(child, "name", f"layer_{len(collected)}"))
                    full_name = f"{prefix}.{child_name}" if prefix else child_name
                    collected.append((full_name, child))
                    _walk(child, full_name)

            _walk(root, "")
            return collected

        layers_attr_named = _collect_from_layers_attr(model)
        layer_path_by_id = {id(layer): name for name, layer in layers_attr_named}

        def _collect_from_flatten_layers() -> List[Tuple[str, tf.keras.layers.Layer]]:
            """Collect recursively via Keras internals when available."""
            flatten_layers = getattr(model, "_flatten_layers", None)
            if not callable(flatten_layers):
                return []

            flattened_layers = None
            for call_kwargs in (
                {"include_self": False, "recursive": True},
                {"include_self": False},
                {},
            ):
                try:
                    flattened_candidate = flatten_layers(**call_kwargs)
                    if not hasattr(flattened_candidate, "__iter__"):
                        continue
                    flattened_layers = list(cast(Iterable[Any], flattened_candidate))
                    break
                except TypeError:
                    continue

            if flattened_layers is None:
                return []

            collected: List[Tuple[str, tf.keras.layers.Layer]] = []
            for layer in flattened_layers:
                if not isinstance(layer, tf.keras.layers.Layer) or layer is model:
                    continue
                layer_name = layer_path_by_id.get(id(layer), str(getattr(layer, "name", f"layer_{len(collected)}")))
                collected.append((layer_name, layer))
            return collected

        def _collect_from_submodules() -> List[Tuple[str, tf.keras.layers.Layer]]:
            """Collect recursively via ``submodules`` when available."""
            submodules = list(getattr(model, "submodules", []) or [])
            collected: List[Tuple[str, tf.keras.layers.Layer]] = []
            for layer in submodules:
                if not isinstance(layer, tf.keras.layers.Layer) or layer is model:
                    continue
                layer_name = layer_path_by_id.get(id(layer), str(getattr(layer, "name", f"layer_{len(collected)}")))
                collected.append((layer_name, layer))
            return collected

        named_layers: List[Tuple[str, tf.keras.layers.Layer]] = []
        seen_layer_ids = set()
        for source in (
            _collect_from_flatten_layers(),
            _collect_from_submodules(),
            layers_attr_named,
        ):
            for layer_name, layer in source:
                layer_id = id(layer)
                if layer_id in seen_layer_ids:
                    continue
                seen_layer_ids.add(layer_id)
                normalized_name = str(layer_name) if layer_name else str(getattr(layer, "name", ""))
                if not normalized_name:
                    normalized_name = f"layer_{len(named_layers)}"
                named_layers.append((normalized_name, layer))

        if named_layers:
            return named_layers

        layers = model.layers
        return [(str(layer.name), layer) for layer in layers]

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
            if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
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

        if isinstance(layer, str):
            idx, _ = self.find_layer_by_name(model, layer)
            return idx

        if isinstance(layer, int):
            if layer < 0:
                return layer + len(model.layers)
            return layer

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
        """
        Apply a mapping function to each batch in a dataset.

        By default, this will execute on GPU if available.

        Parameters
        ----------
        dataset
            The TensorFlow dataset to map over.
        map_fn
            The function to apply to each batch.
        device
            Optional device specification. If None, defaults to GPU:0 if available,
            otherwise CPU:0. Set to "CPU:0" to force CPU execution.

        Returns
        -------
        mapped_dataset
            A new dataset with the mapping function applied.
        """
        # Determine target device: prefer GPU if available
        if device is None:
            gpus = tf.config.list_physical_devices('GPU')
            device = '/GPU:0' if gpus else '/CPU:0'

        # Normalize device string format
        if not device.startswith('/'):
            device = '/' + device

        # Wrap the map function to execute on the specified device
        def device_map_fn(*args):
            with tf.device(device):
                return map_fn(*args)

        return dataset.map(device_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    def cache_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Cache a dataset in memory."""
        return dataset.cache()

    def save_dataset(self, dataset: tf.data.Dataset, path: str) -> None:
        """Save a dataset to disk."""
        tf.data.experimental.save(dataset, path)

    def load_dataset(self, path: str) -> tf.data.Dataset:
        """Load a dataset from disk."""
        if os.path.exists(path):
            return tf.data.experimental.load(path)
        raise FileNotFoundError(f"The dataset path: {path} was not found")

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

    def create_dataset_from_tensors(self, tensors: tf.Tensor, batch_size: int) -> tf.data.Dataset:
        """Create a batched dataset from tensors."""
        return tf.data.Dataset.from_tensors(tensors).batch(batch_size)

    def create_dataset_from_tensor_slices(
        self,
        tensors: Any,
        batch_size: int,
    ) -> tf.data.Dataset:
        """Create a batched dataset by slicing tensors along their first dimension."""
        return tf.data.Dataset.from_tensor_slices(tensors).batch(batch_size)

    def unbatch_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Unbatch a dataset."""
        return dataset.unbatch()

    def shuffle_dataset(self, dataset: tf.data.Dataset, buffer_size: int) -> tf.data.Dataset:
        """Shuffle a dataset."""
        return dataset.shuffle(buffer_size)

    def take_dataset(self, dataset: tf.data.Dataset, count: int) -> tf.data.Dataset:
        """Take a number of elements from a dataset."""
        return dataset.take(count)

    def get_dataset_size(self, dataset: tf.data.Dataset) -> int:
        """Get the total number of elements in a dataset."""
        self.assert_batched_dataset(dataset)
        cardinality_value = dataset.cardinality()
        if isinstance(cardinality_value, int):
            cardinality = cardinality_value
        else:
            cardinality = int(cardinality_value.numpy())
        batch_size = int(dataset._batch_size)  # pylint: disable=W0212
        return cardinality * batch_size

    def get_dataset_element_spec(self, dataset: tf.data.Dataset) -> Any:
        """Get the element spec of a dataset."""
        return dataset.element_spec

    def assert_batched_dataset(self, dataset: tf.data.Dataset) -> None:
        """Assert that a dataset is batched."""
        if not hasattr(dataset, '_batch_size') or dataset._batch_size is None:  # pylint: disable=W0212
            raise ValueError("The dataset must be batched before performing this operation.")

    # Linear algebra operations for IHVP
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> tf.Tensor:
        """Create a tensor of zeros."""
        if dtype is None:
            dtype = tf.float32
        return tf.zeros(shape, dtype=dtype)

    def zeros_like(self, tensor: tf.Tensor) -> tf.Tensor:
        """Create a tensor of zeros with the same shape and dtype as the input."""
        return tf.zeros_like(tensor)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> tf.Tensor:
        """Create a tensor of ones."""
        if dtype is None:
            dtype = tf.float32
        return tf.ones(shape, dtype=dtype)

    def ones_like(self, tensor: tf.Tensor) -> tf.Tensor:
        """Create a tensor of ones with the same shape and dtype as the input."""
        return tf.ones_like(tensor)

    def eye(self, n: int, dtype: Any = None) -> tf.Tensor:
        """Create an identity matrix of size (n, n)."""
        if dtype is None:
            dtype = tf.float32
        return tf.eye(n, dtype=dtype)

    def argsort(self, tensor: tf.Tensor, axis: int = -1, descending: bool = False) -> tf.Tensor:
        """Return the indices that would sort the tensor along an axis."""
        direction = 'DESCENDING' if descending else 'ASCENDING'
        return tf.argsort(tensor, axis=axis, direction=direction)

    def copy(self, tensor: tf.Tensor) -> tf.Tensor:
        """Create a copy of a tensor."""
        return tf.identity(tensor)

    def sqrt(self, tensor: tf.Tensor) -> tf.Tensor:
        """Compute element-wise square root."""
        return tf.sqrt(tensor)

    def maximum(self, a: Any, b: Any) -> tf.Tensor:
        """Element-wise maximum of two tensors/scalars."""
        return tf.maximum(a, b)

    def pinv(self, matrix: tf.Tensor) -> tf.Tensor:
        """Compute the Moore-Penrose pseudo-inverse of a matrix."""
        return tf.linalg.pinv(matrix)

    def cast(self, tensor: tf.Tensor, dtype: Any) -> tf.Tensor:
        """Cast a tensor to a different dtype."""
        return tf.cast(tensor, dtype)

    def get_dtype(self, tensor: tf.Tensor) -> Any:
        """Get the dtype of a tensor."""
        return tensor.dtype

    def to_cpu(self, tensor: Any) -> Any:
        """Move a tensor (or nested tensor structure) to CPU memory."""
        if isinstance(tensor, tf.Tensor):
            with tf.device('/CPU:0'):
                return tf.identity(tensor)
        if isinstance(tensor, list):
            return [self.to_cpu(item) for item in tensor]
        if isinstance(tensor, tuple):
            return tuple(self.to_cpu(item) for item in tensor)
        if isinstance(tensor, dict):
            return {key: self.to_cpu(value) for key, value in tensor.items()}
        return tensor

    def to_device(self, tensor: Any, reference: Optional[Any] = None) -> Any:
        """Move a tensor (or nested tensor structure) to a compute device."""
        if isinstance(tensor, list):
            return [self.to_device(item, reference=reference) for item in tensor]
        if isinstance(tensor, tuple):
            return tuple(self.to_device(item, reference=reference) for item in tensor)
        if isinstance(tensor, dict):
            return {key: self.to_device(value, reference=reference) for key, value in tensor.items()}
        if not isinstance(tensor, tf.Tensor):
            return tensor

        target_device = getattr(reference, 'device', None)
        if target_device:
            with tf.device(target_device):
                return tf.identity(tensor)
        return tf.identity(tensor)

    def float32_dtype(self) -> Any:
        """Return the float32 dtype for the framework."""
        return tf.float32

    def float64_dtype(self) -> Any:
        """Return the float64 dtype for the framework."""
        return tf.float64

    def int32_dtype(self) -> Any:
        """Return the int32 dtype for the framework."""
        return tf.int32

    def int64_dtype(self) -> Any:
        """Return the int64 dtype for the framework."""
        return tf.int64

    def constant(self, value: Any, dtype: Any = None) -> tf.Tensor:
        """Create a constant tensor."""
        return tf.constant(value, dtype=dtype)

    def convert_to_tensor(self, value: Any, dtype: Any = None) -> tf.Tensor:
        """Convert a value to a tensor."""
        return tf.convert_to_tensor(value, dtype=dtype)

    def reduce_prod(self, tensor: tf.Tensor, axis: Optional[int] = None) -> tf.Tensor:
        """Reduce product along an axis."""
        return tf.reduce_prod(tensor, axis=axis)

    def compute_hessian(
        self,
        model: Any,
        weights: List[tf.Variable],
        loss_function: Callable,
        dataset: tf.data.Dataset,
        nb_params: int,
        jacobian_fn: Optional[Callable] = None
    ) -> tf.Tensor:
        """Compute the Hessian matrix of the loss with respect to weights using second-order AD."""
        _ = jacobian_fn
        watched_weights = self.normalize_weights_to_watch(weights)

        # Get dtype from dataset
        element_spec = dataset.element_spec
        if isinstance(element_spec, (tuple, list)):
            first_spec = element_spec[0]
        elif isinstance(element_spec, dict):
            first_spec = next(iter(element_spec.values()))
        else:
            first_spec = element_spec
        dtype = getattr(first_spec, 'dtype', tf.float32)

        hess = tf.zeros((nb_params, nb_params), dtype=dtype)
        nb_elt = 0

        for batch in dataset:
            batch_size = tf.shape(batch[0])[0]

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_hess:
                tape_hess.watch(watched_weights)
                # Compute jacobian inside the tape so we can take second derivatives
                with tf.GradientTape(watch_accessed_variables=False) as tape_inner:
                    tape_inner.watch(watched_weights)
                    predictions = model(batch[0])
                    loss = loss_function(batch[1], predictions)
                    loss = self.ensure_per_sample_loss(loss)
                grads = tape_inner.jacobian(loss, watched_weights)
                self._raise_if_disconnected('compute_hessian (inner jacobian)', grads, watched_weights)
                grads = [tf.reshape(g, (batch_size, -1)) for g in grads]
                grads = tf.concat(grads, axis=1)

            curr_hess = tape_hess.jacobian(grads, watched_weights)
            self._raise_if_disconnected('compute_hessian', curr_hess, watched_weights)
            curr_hess = [tf.reshape(h, shape=(batch_size, nb_params, -1)) for h in curr_hess]
            curr_hess = tf.concat(curr_hess, axis=-1)
            curr_hess = tf.reduce_sum(curr_hess, axis=0)
            hess += tf.cast(curr_hess, dtype=hess.dtype)
            nb_elt += batch_size

        return hess / tf.cast(nb_elt, dtype=hess.dtype)

    def compute_hvp_single(
        self,
        model: Any,
        weights: List[tf.Variable],
        loss_function: Callable,
        v: List[tf.Tensor],
        inputs: tf.Tensor,
        targets: tf.Tensor
    ) -> tf.Tensor:
        """Compute Hessian-vector product using forward-over-backward AD."""
        watched_weights = self.normalize_weights_to_watch(weights)

        with tf.autodiff.ForwardAccumulator(watched_weights, v) as acc:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(watched_weights)
                predictions = model(inputs)
                loss = loss_function(targets, predictions)
                loss = self.ensure_per_sample_loss(loss)
                loss = tf.reduce_sum(loss)
            backward = tape.gradient(loss, watched_weights)
        self._raise_if_disconnected('compute_hvp_single', backward, watched_weights)

        hvp_list = acc.jvp(backward)
        self._raise_if_disconnected('compute_hvp_single', hvp_list, watched_weights)

        # Flatten and concatenate
        hvp = [tf.reshape(h, shape=(-1,)) for h in hvp_list]
        hvp = tf.concat(hvp, axis=0)

        return hvp

    def compute_hvp_batch(
        self,
        model: Any,
        weights: List[tf.Variable],
        loss_function: Callable,
        v: List[tf.Tensor],
        inputs: tf.Tensor,
        targets: tf.Tensor
    ) -> tf.Tensor:
        """Compute Hessian-vector product for a batch using forward-over-backward AD."""
        watched_weights = self.normalize_weights_to_watch(weights)

        with tf.autodiff.ForwardAccumulator(watched_weights, v) as acc:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(watched_weights)
                predictions = model(inputs)
                loss = loss_function(targets, predictions)
                loss = self.ensure_per_sample_loss(loss)
                loss = tf.reduce_sum(loss)
            grads = tape.gradient(loss, watched_weights)
        self._raise_if_disconnected('compute_hvp_batch', grads, watched_weights)

        hvp_list = acc.jvp(grads)
        self._raise_if_disconnected('compute_hvp_batch', hvp_list, watched_weights)

        hvp = [tf.reshape(h, shape=(-1,)) for h in hvp_list]
        hvp = tf.concat(hvp, axis=0)

        return hvp

    def map_fn(self, fn: Callable, elems: tf.Tensor, output_signature: Optional[Any] = None) -> tf.Tensor:
        """Apply a function to each element in a batch."""
        if output_signature is None:
            return tf.map_fn(fn=fn, elems=elems)
        return tf.map_fn(fn=fn, elems=elems, fn_output_signature=output_signature)

    def get_dataset_cardinality(self, dataset: tf.data.Dataset) -> int:
        """Get the number of batches in a dataset."""
        return int(dataset.cardinality())

    def is_sequential_model(self, model: tf.keras.Model) -> bool:
        """Check if a model is a Sequential model."""
        return isinstance(model, tf.keras.Sequential)

    def create_sequential_from_layers(self, layers: List[tf.keras.layers.Layer]) -> tf.keras.Model:
        """Create a Sequential model from a list of layers."""
        return tf.keras.Sequential(layers)

    # Additional operations for boundary-based calculators
    def norm(  # pylint: disable=redefined-builtin
        self,
        tensor: tf.Tensor,
        ord: Optional[int] = None,
        axis: Optional[int] = None,
    ) -> tf.Tensor:
        """Compute the norm of a tensor."""
        if axis is None:
            # Flatten and compute norm
            flat = tf.reshape(tensor, (-1,))
            if ord is None:
                return tf.norm(flat)
            return tf.norm(flat, ord=ord)
        if ord is None:
            return tf.norm(tensor, axis=axis)
        return tf.norm(tensor, ord=ord, axis=axis)

    def top_k(self, tensor: tf.Tensor, k: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Return the top k values and their indices from a tensor."""
        return tf.math.top_k(tensor, k=k)

    def arange(self, start: int, end: int, dtype: Any = None) -> tf.Tensor:
        """Create a tensor with values from start to end."""
        if dtype is None:
            dtype = tf.int32
        return tf.range(start, end, dtype=dtype)

    def tile(self, tensor: tf.Tensor, multiples: Tuple[int, ...]) -> tf.Tensor:
        """Tile a tensor by repeating it along each dimension."""
        return tf.tile(tensor, multiples)

    def repeat(self, tensor: tf.Tensor, repeats: int, axis: int) -> tf.Tensor:
        """Repeat elements of a tensor along an axis."""
        return tf.repeat(tensor, repeats, axis=axis)

    def sign(self, tensor: tf.Tensor) -> tf.Tensor:
        """Compute the element-wise sign of a tensor."""
        return tf.sign(tensor)

    def pow(self, tensor: tf.Tensor, exponent: Any) -> tf.Tensor:
        """Raise tensor elements to a power."""
        return tf.pow(tensor, exponent)

    def logical_and(self, a: Any, b: Any) -> tf.Tensor:
        """Compute element-wise logical AND."""
        return tf.logical_and(a, b)

    def reduce_any(self, tensor: tf.Tensor, axis: Optional[int] = None) -> tf.Tensor:
        """Compute logical OR reduction along an axis."""
        return tf.reduce_any(tensor, axis=axis)

    def argmin(self, tensor: tf.Tensor, axis: int) -> tf.Tensor:
        """Return indices of minimum values along an axis."""
        return tf.argmin(tensor, axis=axis)

    def reduce_mean(self, tensor: tf.Tensor, axis: Optional[int] = None, keepdims: bool = False) -> tf.Tensor:
        """Compute the mean along an axis."""
        return tf.reduce_mean(tensor, axis=axis, keepdims=keepdims)

    def clone_variable(self, variable: tf.Variable) -> tf.Tensor:
        """Create a copy of a variable (weight tensor)."""
        return tf.identity(variable)

    def assign_variable(self, variable: tf.Variable, value: tf.Tensor) -> None:
        """Assign a value to a variable in-place."""
        variable.assign(value)

    def compute_output_jacobian(
        self,
        model: tf.keras.Model,
        inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute the Jacobian of the model output with respect to the input."""
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            outputs = model(inputs)
        jacobian = tape.jacobian(outputs, inputs)
        return outputs, jacobian

    def compute_output_jacobian_wrt_weights(
        self,
        model: tf.keras.Model,
        weights: List[tf.Variable],
        inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Compute the Jacobian of the model output with respect to the weights."""
        watched_weights = self.normalize_weights_to_watch(weights)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(watched_weights)
            outputs = model(inputs)
        jacobian = tape.jacobian(outputs, watched_weights)
        self._raise_if_disconnected('compute_output_jacobian_wrt_weights', jacobian, watched_weights)
        return outputs, jacobian

    def boolean_mask(self, tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Apply a boolean mask to a tensor."""
        return tf.boolean_mask(tensor, mask)

    def while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        loop_vars: List[Any],
        maximum_iterations: Optional[int] = None,
        parallel_iterations: int = 10
    ) -> List[Any]:
        """Execute a while loop with the given condition and body functions."""
        # Wrap cond_fn and body_fn to work with TensorFlow's while_loop
        # TensorFlow passes loop_vars as a flat list, but we want to unpack them
        def wrapped_cond(*args):
            return cond_fn(*args)

        def wrapped_body(*args):
            result = body_fn(*args)
            # Ensure the result is a list to match input structure
            if isinstance(result, tuple):
                result = list(result)
            return result

        result = tf.while_loop(
            wrapped_cond,
            wrapped_body,
            loop_vars,
            maximum_iterations=maximum_iterations,
            parallel_iterations=parallel_iterations
        )
        return result

    # Arnoldi algorithm specific operations
    def random_normal(self, shape: Tuple[int, ...], dtype: Any = None) -> tf.Tensor:
        """Generate random tensor from normal distribution."""
        if dtype is None:
            dtype = tf.float32
        return tf.random.normal(shape, dtype=dtype)

    def diag_part(self, tensor: tf.Tensor, k: int = 0) -> tf.Tensor:
        """Extract diagonal from a matrix with offset k."""
        return tf.linalg.diag_part(tensor, k=k)

    @tf.autograph.experimental.do_not_convert
    def eigh_tridiagonal(
        self,
        maindiag: tf.Tensor,
        superdiag: tf.Tensor,
        eigvals_only: bool = False
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Compute eigenvalues and eigenvectors of a symmetric tridiagonal matrix."""
        # Use cpu device for eigh_tridiagonal as it's not supported on GPU
        with tf.device('cpu'):
            result = tf.linalg.eigh_tridiagonal(maindiag, superdiag, eigvals_only=eigvals_only)
        if eigvals_only:
            return result, None
        eig_vals, eig_vectors = result
        return eig_vals, eig_vectors

    @tf.autograph.experimental.do_not_convert
    def eig(self, tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute eigenvalues and eigenvectors of a square matrix."""
        return tf.linalg.eig(tensor)

    @tf.autograph.experimental.do_not_convert
    def real(self, tensor: tf.Tensor) -> tf.Tensor:
        """Return the real part of a complex tensor."""
        return tf.math.real(tensor)

    # ------------------------------------------------------------------
    # K-FAC / EK-FAC support operations
    # ------------------------------------------------------------------

    @tf.autograph.experimental.do_not_convert
    def eigh(self, tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Symmetric eigendecomposition using tf.linalg.eigh."""
        eigenvalues, eigenvectors = tf.linalg.eigh(tensor)
        return eigenvalues, eigenvectors

    @tf.autograph.experimental.do_not_convert
    def kron(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Kronecker product of two 2-D matrices via reshape + tensordot."""
        a_shape = tf.shape(a)
        b_shape = tf.shape(b)
        # a: (m, n), b: (p, q) -> result: (m*p, n*q)
        # Expand: a[:, :, None, None] * b[None, None, :, :] -> (m, n, p, q)
        # Then transpose to (m, p, n, q) and reshape to (m*p, n*q)
        product = a[:, :, tf.newaxis, tf.newaxis] * b[tf.newaxis, tf.newaxis, :, :]
        product = tf.transpose(product, perm=[0, 2, 1, 3])
        return tf.reshape(product, [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]])

    @tf.autograph.experimental.do_not_convert
    def outer(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Outer product of two 1-D vectors."""
        return tf.tensordot(a, b, axes=0)

    def is_linear_layer(self, layer: Any) -> bool:
        """Check whether *layer* is tf.keras.layers.Dense."""
        return isinstance(layer, tf.keras.layers.Dense)

    def is_conv2d_layer(self, layer: Any) -> bool:
        """Check whether *layer* is tf.keras.layers.Conv2D."""
        return isinstance(layer, tf.keras.layers.Conv2D)

    @tf.autograph.experimental.do_not_convert
    def get_layer_weight_and_bias(self, layer: tf.keras.layers.Layer) -> Tuple[Any, Optional[Any]]:
        """Return the kernel and optional bias of a Dense or Conv2D layer."""
        weight = layer.kernel
        bias = getattr(layer, 'bias', None)
        # Keras stores bias as None when use_bias=False.
        # In Keras 3, trainable weights can be wrapped objects (not strict
        # ``tf.Variable`` instances), so we keep any non-None bias reference.
        return weight, bias

    def register_forward_hook(self, layer: tf.keras.layers.Layer, hook: Callable) -> Any:
        """
        Register a forward hook on a Keras layer.

        Uses a lightweight wrapper around ``layer.call`` that invokes the hook
        after the original forward pass.  Returns a handle object whose
        ``remove()`` restores the original ``call``.
        """
        original_call = layer.call

        def hooked_call(*args, **kwargs):
            # Keras ``call`` receives the layer input as the first positional arg.
            output = original_call(*args, **kwargs)
            # Match PyTorch convention: hook(layer, input, output)
            layer_input = args[0] if args else kwargs.get('inputs', None)
            hook(layer, layer_input, output)
            return output

        layer.call = hooked_call

        class _Handle:
            """Minimal handle that restores the original ``call``."""
            def remove(self):
                layer.call = original_call

        return _Handle()

    def register_backward_hook(self, layer: tf.keras.layers.Layer, hook: Callable) -> Any:
        """
        Register a backward hook on a Keras layer.

        Keras does not natively support backward hooks.  This implementation
        stores the hook reference so that the caller (e.g. ``KroneckerFactors``)
        can invoke it manually inside a ``GradientTape`` context.  The returned
        handle's ``remove()`` simply clears the stored reference.
        """
        # Store on the layer so the factor computation code can retrieve it.
        if not hasattr(layer, '_kfac_backward_hooks'):
            layer._kfac_backward_hooks = []

        layer._kfac_backward_hooks.append(hook)

        class _Handle:
            """Minimal handle that removes the hook from the layer."""
            def remove(self):
                try:
                    layer._kfac_backward_hooks.remove(hook)
                except ValueError:
                    pass

        return _Handle()

    def remove_hook(self, handle: Any) -> None:
        """Remove a previously registered hook."""
        handle.remove()

    def svd_lowrank(self, matrix: Any, rank: int) -> Tuple[Any, Any, Any]:
        """
        Compute a low-rank truncated SVD using ``tf.linalg.svd``.

        TensorFlow does not have a native randomised low-rank SVD, so this
        implementation computes the full SVD and retains only the top *rank*
        singular triplets.  For large matrices consider pre-projecting before
        calling this method.

        Parameters
        ----------
        matrix
            2-D tensor of shape ``(m, n)``.
        rank
            Number of singular triplets to retain.

        Returns
        -------
        U
            Left singular vectors, shape ``(m, rank)``.
        S
            Singular values, shape ``(rank,)``.
        Vh
            Right singular vectors (transposed), shape ``(rank, n)``.
        """
        s_full, u_full, v_full = tf.linalg.svd(matrix, full_matrices=False)
        u = u_full[:, :rank]
        s = s_full[:rank]
        vh = tf.transpose(v_full[:, :rank])
        return u, s, vh

    def einsum(self, equation: str, *operands: Any) -> Any:
        """Evaluate an Einstein summation using ``tf.einsum``."""
        return tf.einsum(equation, *operands)
