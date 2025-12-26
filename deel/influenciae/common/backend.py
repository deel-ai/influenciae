# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Backend abstraction layer for framework-agnostic operations.
Supports both TensorFlow and PyTorch.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Tuple, Callable, Optional, Union

import numpy as np


class Framework(Enum):
    """Enum for supported deep learning frameworks."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"


def get_available_frameworks() -> List[Framework]:
    """
    Detect which deep learning frameworks are available.

    Returns
    -------
    frameworks
        List of available frameworks.
    """
    available = []
    try:
        import tensorflow  # noqa: F401
        available.append(Framework.TENSORFLOW)
    except ImportError:
        pass

    try:
        import torch  # noqa: F401
        available.append(Framework.PYTORCH)
    except ImportError:
        pass

    return available


def detect_framework(model: Any) -> Framework:
    """
    Detect which framework a model belongs to.

    Parameters
    ----------
    model
        The model to check.

    Returns
    -------
    framework
        The detected framework.

    Raises
    ------
    ValueError
        If the model framework cannot be detected.
    """
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return Framework.TENSORFLOW
    except ImportError:
        pass

    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return Framework.PYTORCH
    except ImportError:
        pass

    raise ValueError(
        f"Could not detect framework for model of type {type(model)}. "
        "Supported frameworks: TensorFlow (tf.keras.Model) and PyTorch (nn.Module)"
    )


class BaseBackend(ABC):
    """
    Abstract base class for framework-specific backend operations.
    """

    @property
    @abstractmethod
    def framework(self) -> Framework:
        """Return the framework this backend supports."""
        pass

    @abstractmethod
    def get_model_weights(self, model: Any, layers: Optional[List[Any]] = None) -> List[Any]:
        """
        Get trainable weights from a model.

        Parameters
        ----------
        model
            The model to extract weights from.
        layers
            Optional list of specific layers to get weights from.

        Returns
        -------
        weights
            List of weight tensors.
        """
        pass

    @abstractmethod
    def get_num_params(self, weights: List[Any]) -> int:
        """
        Get the total number of parameters in a list of weights.

        Parameters
        ----------
        weights
            List of weight tensors.

        Returns
        -------
        num_params
            Total number of parameters.
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: Any,
        loss_function: Callable,
        inputs: Any,
        targets: Any,
        sample_weight: Optional[Any] = None
    ) -> Any:
        """
        Compute the loss for a batch of samples.

        Parameters
        ----------
        model
            The model to compute loss for.
        loss_function
            The loss function to use.
        inputs
            Input batch.
        targets
            Target batch.
        sample_weight
            Optional sample weights.

        Returns
        -------
        loss
            Loss values for each sample (unreduced).
        """
        pass

    @abstractmethod
    def compute_jacobian(
        self,
        model: Any,
        weights: List[Any],
        loss_function: Callable,
        inputs: Any,
        targets: Any,
        sample_weight: Optional[Any] = None
    ) -> Any:
        """
        Compute the Jacobian of the loss with respect to weights.

        Parameters
        ----------
        model
            The model to compute Jacobian for.
        weights
            List of weight tensors to compute Jacobian for.
        loss_function
            The loss function to use.
        inputs
            Input batch.
        targets
            Target batch.
        sample_weight
            Optional sample weights.

        Returns
        -------
        jacobian
            Jacobian matrix (batch_size, num_params).
        """
        pass

    @abstractmethod
    def compute_gradient(
        self,
        model: Any,
        weights: List[Any],
        loss_function: Callable,
        inputs: Any,
        targets: Any,
        sample_weight: Optional[Any] = None
    ) -> Any:
        """
        Compute the gradient of the loss with respect to weights.

        Parameters
        ----------
        model
            The model to compute gradient for.
        weights
            List of weight tensors to compute gradient for.
        loss_function
            The loss function to use.
        inputs
            Input batch.
        targets
            Target batch.
        sample_weight
            Optional sample weights.

        Returns
        -------
        gradient
            Gradient vector (num_params,).
        """
        pass

    @abstractmethod
    def concat(self, tensors: List[Any], axis: int = 0) -> Any:
        """Concatenate tensors along an axis."""
        pass

    @abstractmethod
    def stack(self, tensors: List[Any], axis: int = 0) -> Any:
        """Stack tensors along a new axis."""
        pass

    @abstractmethod
    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        """Reshape a tensor."""
        pass

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a tensor to numpy array."""
        pass

    @abstractmethod
    def get_batch_size(self, tensor: Any) -> int:
        """Get the batch size (first dimension) of a tensor."""
        pass

    @abstractmethod
    def reduce_sum(self, tensor: Any, axis: Optional[int] = None) -> Any:
        """Reduce sum along an axis."""
        pass

    @abstractmethod
    def expand_dims(self, tensor: Any, axis: int) -> Any:
        """Add a new axis to a tensor."""
        pass

    @abstractmethod
    def squeeze(self, tensor: Any, axis: Optional[int] = None) -> Any:
        """Remove dimensions of size 1."""
        pass

    @abstractmethod
    def transpose(self, tensor: Any) -> Any:
        """Transpose a tensor (swap last two dimensions)."""
        pass

    @abstractmethod
    def tensor_shape(self, tensor: Any) -> Tuple[int, ...]:
        """Get the shape of a tensor."""
        pass

    @abstractmethod
    def tensor_ndim(self, tensor: Any) -> int:
        """Get the number of dimensions of a tensor."""
        pass

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        pass

    @abstractmethod
    def find_layer_by_name(self, model: Any, layer_name: str) -> Tuple[int, Any]:
        """
        Find a layer by name and return its index and the layer.

        Parameters
        ----------
        model
            The model to search in.
        layer_name
            Name of the layer to find.

        Returns
        -------
        layer_idx
            Index of the layer.
        layer
            The layer object.
        """
        pass

    @abstractmethod
    def get_layers(self, model: Any) -> List[Any]:
        """Get all layers from a model."""
        pass

    @abstractmethod
    def forward(self, model: Any, inputs: Any) -> Any:
        """Run forward pass on a model."""
        pass

    @abstractmethod
    def get_weights_for_layer_range(
        self,
        model: Any,
        start_layer: Optional[Any] = None,
        last_layer: Optional[Any] = None
    ) -> List[Any]:
        """
        Get weights for a range of layers.

        Parameters
        ----------
        model
            The model to get weights from.
        start_layer
            Starting layer (name, index, or None for auto-detect).
        last_layer
            Ending layer (name, index, or None for just start_layer).

        Returns
        -------
        weights
            List of weight tensors.
        """
        pass

    # Dataset operations
    @abstractmethod
    def map_dataset(self, dataset: Any, map_fn: Callable, device: Optional[str] = None) -> Any:
        """
        Apply a mapping function to each batch in a dataset.

        Parameters
        ----------
        dataset
            The dataset to map over (tf.data.Dataset or PyTorch DataLoader).
        map_fn
            The function to apply to each batch.
        device
            Optional device to execute on.

        Returns
        -------
        mapped_dataset
            A new dataset/iterable with the map function applied.
        """
        pass

    @abstractmethod
    def cache_dataset(self, dataset: Any) -> Any:
        """
        Cache a dataset in memory.

        Parameters
        ----------
        dataset
            The dataset to cache.

        Returns
        -------
        cached_dataset
            The cached dataset.
        """
        pass

    @abstractmethod
    def save_dataset(self, dataset: Any, path: str) -> None:
        """
        Save a dataset to disk.

        Parameters
        ----------
        dataset
            The dataset to save.
        path
            Path to save the dataset.
        """
        pass

    @abstractmethod
    def load_dataset(self, path: str) -> Any:
        """
        Load a dataset from disk.

        Parameters
        ----------
        path
            Path to load the dataset from.

        Returns
        -------
        dataset
            The loaded dataset.
        """
        pass

    @abstractmethod
    def get_dataset_batch_size(self, dataset: Any) -> int:
        """
        Get the batch size of a dataset.

        Parameters
        ----------
        dataset
            The dataset.

        Returns
        -------
        batch_size
            The batch size.
        """
        pass

    @abstractmethod
    def zip_datasets(self, dataset1: Any, dataset2: Any) -> Any:
        """
        Zip two datasets together.

        Parameters
        ----------
        dataset1
            First dataset.
        dataset2
            Second dataset.

        Returns
        -------
        zipped_dataset
            The zipped dataset.
        """
        pass

    @abstractmethod
    def batch_dataset(self, dataset: Any, batch_size: int) -> Any:
        """
        Batch a dataset.

        Parameters
        ----------
        dataset
            The dataset to batch.
        batch_size
            The batch size.

        Returns
        -------
        batched_dataset
            The batched dataset.
        """
        pass

    @abstractmethod
    def unbatch_dataset(self, dataset: Any) -> Any:
        """
        Unbatch a dataset.

        Parameters
        ----------
        dataset
            The dataset to unbatch.

        Returns
        -------
        unbatched_dataset
            The unbatched dataset.
        """
        pass

    @abstractmethod
    def get_dataset_element_spec(self, dataset: Any) -> Any:
        """
        Get the element spec of a dataset.

        Parameters
        ----------
        dataset
            The dataset.

        Returns
        -------
        element_spec
            The element specification.
        """
        pass

    @abstractmethod
    def assert_batched_dataset(self, dataset: Any) -> None:
        """
        Assert that a dataset is batched.

        Parameters
        ----------
        dataset
            The dataset to check.

        Raises
        ------
        ValueError
            If the dataset is not batched.
        """
        pass

    # Linear algebra operations for IHVP
    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create a tensor of zeros."""
        pass

    @abstractmethod
    def pinv(self, matrix: Any) -> Any:
        """Compute the Moore-Penrose pseudo-inverse of a matrix."""
        pass

    @abstractmethod
    def cast(self, tensor: Any, dtype: Any) -> Any:
        """Cast a tensor to a different dtype."""
        pass

    @abstractmethod
    def get_dtype(self, tensor: Any) -> Any:
        """Get the dtype of a tensor."""
        pass

    @abstractmethod
    def float32_dtype(self) -> Any:
        """Return the float32 dtype for the framework."""
        pass

    @abstractmethod
    def int32_dtype(self) -> Any:
        """Return the int32 dtype for the framework."""
        pass

    @abstractmethod
    def int64_dtype(self) -> Any:
        """Return the int64 dtype for the framework."""
        pass

    @abstractmethod
    def constant(self, value: Any, dtype: Any = None) -> Any:
        """Create a constant tensor."""
        pass

    @abstractmethod
    def convert_to_tensor(self, value: Any, dtype: Any = None) -> Any:
        """Convert a value to a tensor."""
        pass

    @abstractmethod
    def reduce_prod(self, tensor: Any, axis: Optional[int] = None) -> Any:
        """Reduce product along an axis."""
        pass

    @abstractmethod
    def compute_hessian(
        self,
        model: Any,
        weights: List[Any],
        loss_function: Callable,
        dataset: Any,
        nb_params: int,
        jacobian_fn: Optional[Callable] = None
    ) -> Any:
        """
        Compute the Hessian matrix of the loss with respect to weights.

        Parameters
        ----------
        model
            The model.
        weights
            List of weight tensors.
        loss_function
            The loss function.
        dataset
            The dataset to compute Hessian over.
        nb_params
            Number of parameters.
        jacobian_fn
            Optional function to compute batch Jacobian. If provided, uses this
            for more accurate Hessian computation.

        Returns
        -------
        hessian
            The Hessian matrix.
        """
        pass

    @abstractmethod
    def compute_hvp_single(
        self,
        model: Any,
        weights: List[Any],
        loss_function: Callable,
        v: Any,
        inputs: Any,
        targets: Any
    ) -> Any:
        """
        Compute Hessian-vector product for a single sample using forward-over-backward AD.

        Parameters
        ----------
        model
            The model.
        weights
            List of weight tensors.
        loss_function
            The loss function.
        v
            The vector to multiply with the Hessian.
        inputs
            Input sample.
        targets
            Target sample.

        Returns
        -------
        hvp
            The Hessian-vector product.
        """
        pass

    @abstractmethod
    def map_fn(self, fn: Callable, elems: Any) -> Any:
        """
        Apply a function to each element in a batch.

        Parameters
        ----------
        fn
            The function to apply.
        elems
            The elements to map over.

        Returns
        -------
        result
            The mapped results.
        """
        pass

    @abstractmethod
    def get_dataset_cardinality(self, dataset: Any) -> int:
        """
        Get the number of batches in a dataset.

        Parameters
        ----------
        dataset
            The dataset.

        Returns
        -------
        cardinality
            Number of batches.
        """
        pass

    @abstractmethod
    def is_sequential_model(self, model: Any) -> bool:
        """Check if a model is a Sequential model."""
        pass

    @abstractmethod
    def create_sequential_from_layers(self, layers: List[Any]) -> Any:
        """Create a Sequential model from a list of layers."""
        pass


def get_backend(framework: Framework) -> BaseBackend:
    """
    Get the backend for a specific framework.

    Parameters
    ----------
    framework
        The framework to get the backend for.

    Returns
    -------
    backend
        The backend instance for the framework.
    """
    if framework == Framework.TENSORFLOW:
        from .backend_tensorflow import TensorFlowBackend
        return TensorFlowBackend()
    elif framework == Framework.PYTORCH:
        from .backend_pytorch import PyTorchBackend
        return PyTorchBackend()
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def get_backend_for_model(model: Any) -> BaseBackend:
    """
    Get the appropriate backend for a model.

    Parameters
    ----------
    model
        The model to get the backend for.

    Returns
    -------
    backend
        The backend instance appropriate for the model.
    """
    framework = detect_framework(model)
    return get_backend(framework)

