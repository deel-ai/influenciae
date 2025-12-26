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

