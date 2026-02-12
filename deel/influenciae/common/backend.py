# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Backend abstraction layer for framework-agnostic operations.
Supports both TensorFlow and PyTorch.
"""
# pylint: disable=too-many-lines
from abc import ABC, abstractmethod
from enum import Enum
import importlib.util
from typing import Any, List, Tuple, Callable, Optional, Union

import numpy as np

from .._optional_imports import import_optional_attr, import_optional_module


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
    if importlib.util.find_spec("tensorflow") is not None:
        available.append(Framework.TENSORFLOW)

    if importlib.util.find_spec("torch") is not None:
        available.append(Framework.PYTORCH)

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
        tf = import_optional_module("tensorflow", extra="tensorflow")
        if isinstance(model, tf.keras.Model):
            return Framework.TENSORFLOW
    except ImportError:
        pass

    try:
        nn = import_optional_module("torch.nn", extra="pytorch")
        if isinstance(model, nn.Module):
            return Framework.PYTORCH
    except ImportError:
        pass

    raise ValueError(
        f"Could not detect framework for model of type {type(model)}. "
        "Supported frameworks: TensorFlow (tf.keras.Model) and PyTorch (nn.Module)"
    )


def detect_tensor_framework(tensor: Any) -> Framework:
    """
    Detect which framework a tensor belongs to.

    Parameters
    ----------
    tensor
        The tensor to check.

    Returns
    -------
    framework
        The detected framework.

    Raises
    ------
    ValueError
        If the tensor framework cannot be detected.
    """
    try:
        tf = import_optional_module("tensorflow", extra="tensorflow")
        if tf.is_tensor(tensor):
            return Framework.TENSORFLOW
    except ImportError:
        pass

    try:
        torch = import_optional_module("torch", extra="pytorch")
        if isinstance(tensor, torch.Tensor):
            return Framework.PYTORCH
    except ImportError:
        pass

    raise ValueError(
        f"Could not detect framework for tensor of type {type(tensor)}. "
        "Supported frameworks: TensorFlow (tf.Tensor) and PyTorch (torch.Tensor)"
    )


def detect_dtype_framework(dtype: Any) -> Optional[Framework]:
    """
    Detect which framework a dtype belongs to.

    Parameters
    ----------
    dtype
        The dtype to check.

    Returns
    -------
    framework
        The detected framework, or None if it cannot be determined.
    """
    # Check for TensorFlow dtype
    try:
        tf = import_optional_module("tensorflow", extra="tensorflow")
        if isinstance(dtype, tf.DType):
            return Framework.TENSORFLOW
        # Also check for string representation of TensorFlow dtypes
        dtype_str = str(dtype).lower()
        if dtype_str.startswith('tf.') or dtype_str.startswith('<dtype:'):
            return Framework.TENSORFLOW
    except ImportError:
        pass

    # Check for PyTorch dtype
    try:
        torch = import_optional_module("torch", extra="pytorch")
        if isinstance(dtype, torch.dtype):
            return Framework.PYTORCH
        # Also check for string representation of PyTorch dtypes
        dtype_str = str(dtype).lower()
        if dtype_str.startswith('torch.'):
            return Framework.PYTORCH
    except ImportError:
        pass

    # Could not determine framework from dtype
    return None


def get_backend_for_tensor(tensor: Any) -> "BaseBackend":
    """
    Get the appropriate backend for a tensor.

    Parameters
    ----------
    tensor
        The tensor to get the backend for.

    Returns
    -------
    backend
        The backend instance appropriate for the tensor.
    """
    framework = detect_tensor_framework(tensor)
    return get_backend(framework)


class BaseBackend(ABC):  # pylint: disable=too-many-public-methods
    """
    Abstract base class for framework-specific backend operations.
    """

    @property
    @abstractmethod
    def framework(self) -> Framework:
        """Return the framework this backend supports."""

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

    @abstractmethod
    def concat(self, tensors: List[Any], axis: int = 0) -> Any:
        """Concatenate tensors along an axis."""

    @abstractmethod
    def stack(self, tensors: List[Any], axis: int = 0) -> Any:
        """Stack tensors along a new axis."""

    @abstractmethod
    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        """Reshape a tensor."""

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a tensor to numpy array."""

    @abstractmethod
    def get_batch_size(self, tensor: Any) -> int:
        """Get the batch size (first dimension) of a tensor."""

    @abstractmethod
    def reduce_sum(self, tensor: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Reduce sum along an axis."""

    @abstractmethod
    def expand_dims(self, tensor: Any, axis: int) -> Any:
        """Add a new axis to a tensor."""

    @abstractmethod
    def squeeze(self, tensor: Any, axis: Optional[int] = None) -> Any:
        """Remove dimensions of size 1."""

    @abstractmethod
    def transpose(self, tensor: Any) -> Any:
        """Transpose a tensor (swap last two dimensions)."""

    @abstractmethod
    def tensor_shape(self, tensor: Any) -> Tuple[int, ...]:
        """Get the shape of a tensor."""

    @abstractmethod
    def tensor_ndim(self, tensor: Any) -> int:
        """Get the number of dimensions of a tensor."""

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""

    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication."""

    @abstractmethod
    def abs(self, tensor: Any) -> Any:
        """Compute absolute value of a tensor."""

    @abstractmethod
    def argmax(self, tensor: Any, axis: int) -> Any:
        """Return indices of maximum values along an axis."""

    @abstractmethod
    def gather_along_axis(self, tensor: Any, indices: Any, axis: int, batch_dims: int = 0) -> Any:
        """
        Gather values from tensor along an axis using indices.

        Parameters
        ----------
        tensor
            The source tensor.
        indices
            The indices to gather.
        axis
            The axis along which to gather.
        batch_dims
            Number of batch dimensions.

        Returns
        -------
        gathered
            The gathered tensor.
        """

    @abstractmethod
    def get_output_shape(self, model: Any) -> Tuple[Optional[int], ...]:
        """
        Get the output shape of a model.

        Parameters
        ----------
        model
            The model.

        Returns
        -------
        shape
            The output shape.
        """

    @abstractmethod
    def split_model(self, model: Any, target_layer: Union[str, int]) -> Tuple[Any, Any]:
        """
        Split a model into two sub-models at a target layer.

        Parameters
        ----------
        model
            The model to split.
        target_layer
            The layer name or index at which to split.

        Returns
        -------
        feature_extractor
            Model containing layers up to (but not including) target_layer.
        head
            Model containing the target_layer and beyond.
        """

    @abstractmethod
    def normalize(self, tensor: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """
        Normalize a tensor along an axis using L2 norm.

        Parameters
        ----------
        tensor
            The tensor to normalize.
        axis
            The axis along which to normalize.
        keepdims
            Whether to keep the reduced dimension.

        Returns
        -------
        normalized_tensor
            The normalized tensor.
        """

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

    @abstractmethod
    def get_layers(self, model: Any) -> List[Any]:
        """Get all layers from a model."""

    @abstractmethod
    def forward(self, model: Any, inputs: Any) -> Any:
        """Run forward pass on a model."""

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

    @abstractmethod
    def create_dataset_from_tensors(self, tensors: Any, batch_size: int) -> Any:
        """
        Create a batched dataset from a single tensor or tuple of tensors.

        Parameters
        ----------
        tensors
            A tensor or tuple of tensors to create a dataset from.
        batch_size
            The batch size for the resulting dataset.

        Returns
        -------
        dataset
            A batched dataset containing the tensors.
        """

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

    @abstractmethod
    def shuffle_dataset(self, dataset: Any, buffer_size: int) -> Any:
        """
        Shuffle a dataset.

        Parameters
        ----------
        dataset
            The dataset to shuffle.
        buffer_size
            The buffer size for shuffling.

        Returns
        -------
        shuffled_dataset
            The shuffled dataset.
        """

    @abstractmethod
    def take_dataset(self, dataset: Any, count: int) -> Any:
        """
        Take a number of elements from a dataset.

        Parameters
        ----------
        dataset
            The dataset.
        count
            The number of elements to take.

        Returns
        -------
        taken_dataset
            The dataset with only the first `count` elements.
        """

    @abstractmethod
    def get_dataset_size(self, dataset: Any) -> int:
        """
        Get the total number of elements in a dataset.

        Parameters
        ----------
        dataset
            The dataset.

        Returns
        -------
        size
            The total number of elements.
        """

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

    # Linear algebra operations for IHVP
    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create a tensor of zeros."""

    @abstractmethod
    def zeros_like(self, tensor: Any) -> Any:
        """Create a tensor of zeros with the same shape and dtype as the input."""

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create a tensor of ones."""

    @abstractmethod
    def ones_like(self, tensor: Any) -> Any:
        """Create a tensor of ones with the same shape and dtype as the input."""

    @abstractmethod
    def argsort(self, tensor: Any, axis: int = -1, descending: bool = False) -> Any:
        """
        Return the indices that would sort the tensor along an axis.

        Parameters
        ----------
        tensor
            The input tensor.
        axis
            The axis along which to sort.
        descending
            If True, sort in descending order. Default is False (ascending).

        Returns
        -------
        indices
            Indices that would sort the tensor.
        """

    @abstractmethod
    def copy(self, tensor: Any) -> Any:
        """Create a copy of a tensor."""

    @abstractmethod
    def sqrt(self, tensor: Any) -> Any:
        """Compute element-wise square root."""

    @abstractmethod
    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum of two tensors/scalars."""

    @abstractmethod
    def pinv(self, matrix: Any) -> Any:
        """Compute the Moore-Penrose pseudo-inverse of a matrix."""

    @abstractmethod
    def cast(self, tensor: Any, dtype: Any) -> Any:
        """Cast a tensor to a different dtype."""

    @abstractmethod
    def get_dtype(self, tensor: Any) -> Any:
        """Get the dtype of a tensor."""

    @abstractmethod
    def float32_dtype(self) -> Any:
        """Return the float32 dtype for the framework."""

    @abstractmethod
    def int32_dtype(self) -> Any:
        """Return the int32 dtype for the framework."""

    @abstractmethod
    def int64_dtype(self) -> Any:
        """Return the int64 dtype for the framework."""

    @abstractmethod
    def constant(self, value: Any, dtype: Any = None) -> Any:
        """Create a constant tensor."""

    @abstractmethod
    def convert_to_tensor(self, value: Any, dtype: Any = None) -> Any:
        """Convert a value to a tensor."""

    @abstractmethod
    def reduce_prod(self, tensor: Any, axis: Optional[int] = None) -> Any:
        """Reduce product along an axis."""

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

    @abstractmethod
    def compute_hvp_batch(
        self,
        model: Any,
        weights: List[Any],
        loss_function: Callable,
        v: Any,
        inputs: Any,
        targets: Any
    ) -> Any:
        """
        Compute Hessian-vector product for a batch using forward-over-backward AD.

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
            Input batch.
        targets
            Target batch.

        Returns
        -------
        hvp
            The Hessian-vector product summed over the batch.
        """

    @abstractmethod
    def map_fn(self, fn: Callable, elems: Any, output_signature: Optional[Any] = None) -> Any:
        """
        Apply a function to each element in a batch.

        Parameters
        ----------
        fn
            The function to apply.
        elems
            The elements to map over.
        output_signature
            Optional output signature/type spec for the mapped function.

        Returns
        -------
        result
            The mapped results.
        """

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

    @abstractmethod
    def is_sequential_model(self, model: Any) -> bool:
        """Check if a model is a Sequential model."""

    @abstractmethod
    def create_sequential_from_layers(self, layers: List[Any]) -> Any:
        """Create a Sequential model from a list of layers."""

    # Additional operations for boundary-based calculators
    @abstractmethod
    def norm(  # pylint: disable=redefined-builtin
        self,
        tensor: Any,
        ord: Optional[int] = None,
        axis: Optional[int] = None,
    ) -> Any:
        """
        Compute the norm of a tensor.

        Parameters
        ----------
        tensor
            The input tensor.
        ord
            Order of the norm (e.g., 1, 2, or None for Frobenius).
        axis
            The axis along which to compute the norm.

        Returns
        -------
        norm_value
            The computed norm.
        """

    @abstractmethod
    def top_k(self, tensor: Any, k: int) -> Tuple[Any, Any]:
        """
        Return the top k values and their indices from a tensor.

        Parameters
        ----------
        tensor
            The input tensor.
        k
            The number of top elements to return.

        Returns
        -------
        values
            The top k values.
        indices
            The indices of the top k values.
        """

    @abstractmethod
    def arange(self, start: int, end: int, dtype: Any = None) -> Any:
        """
        Create a tensor with values from start to end.

        Parameters
        ----------
        start
            Start value.
        end
            End value (exclusive).
        dtype
            Data type for the tensor.

        Returns
        -------
        range_tensor
            Tensor with values [start, start+1, ..., end-1].
        """

    @abstractmethod
    def tile(self, tensor: Any, multiples: Tuple[int, ...]) -> Any:
        """
        Tile a tensor by repeating it along each dimension.

        Parameters
        ----------
        tensor
            The input tensor.
        multiples
            Number of times to repeat along each dimension.

        Returns
        -------
        tiled_tensor
            The tiled tensor.
        """

    @abstractmethod
    def repeat(self, tensor: Any, repeats: int, axis: int) -> Any:
        """
        Repeat elements of a tensor along an axis.

        Parameters
        ----------
        tensor
            The input tensor.
        repeats
            Number of times to repeat each element.
        axis
            The axis along which to repeat.

        Returns
        -------
        repeated_tensor
            The repeated tensor.
        """

    @abstractmethod
    def sign(self, tensor: Any) -> Any:
        """
        Compute the element-wise sign of a tensor.

        Parameters
        ----------
        tensor
            The input tensor.

        Returns
        -------
        sign_tensor
            Tensor with -1, 0, or 1 based on the sign of each element.
        """

    @abstractmethod
    def pow(self, tensor: Any, exponent: Any) -> Any:
        """
        Raise tensor elements to a power.

        Parameters
        ----------
        tensor
            The input tensor.
        exponent
            The exponent value.

        Returns
        -------
        powered_tensor
            The tensor with each element raised to the power.
        """

    @abstractmethod
    def logical_and(self, a: Any, b: Any) -> Any:
        """
        Compute element-wise logical AND.

        Parameters
        ----------
        a
            First tensor.
        b
            Second tensor.

        Returns
        -------
        result
            Element-wise logical AND result.
        """

    @abstractmethod
    def reduce_any(self, tensor: Any, axis: Optional[int] = None) -> Any:
        """
        Compute logical OR reduction along an axis.

        Parameters
        ----------
        tensor
            The input boolean tensor.
        axis
            The axis along which to reduce.

        Returns
        -------
        result
            Reduced tensor.
        """

    @abstractmethod
    def argmin(self, tensor: Any, axis: int) -> Any:
        """
        Return indices of minimum values along an axis.

        Parameters
        ----------
        tensor
            The input tensor.
        axis
            The axis along which to find the minimum.

        Returns
        -------
        indices
            Indices of the minimum values.
        """

    @abstractmethod
    def reduce_mean(self, tensor: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """
        Compute the mean along an axis.

        Parameters
        ----------
        tensor
            The input tensor.
        axis
            The axis along which to compute the mean.
        keepdims
            Whether to keep the reduced dimension.

        Returns
        -------
        mean_tensor
            The mean value(s).
        """

    @abstractmethod
    def clone_variable(self, variable: Any) -> Any:
        """
        Create a copy of a variable (weight tensor).

        Parameters
        ----------
        variable
            The variable to clone.

        Returns
        -------
        cloned_variable
            A copy of the variable.
        """

    @abstractmethod
    def assign_variable(self, variable: Any, value: Any) -> None:
        """
        Assign a value to a variable in-place.

        Parameters
        ----------
        variable
            The variable to update.
        value
            The new value.
        """

    @abstractmethod
    def compute_output_jacobian(
        self,
        model: Any,
        inputs: Any
    ) -> Tuple[Any, Any]:
        """
        Compute the Jacobian of the model output with respect to the input.

        Parameters
        ----------
        model
            The model.
        inputs
            The input tensor.

        Returns
        -------
        outputs
            The model outputs.
        jacobian
            The Jacobian of outputs with respect to inputs.
        """

    @abstractmethod
    def compute_output_jacobian_wrt_weights(
        self,
        model: Any,
        weights: List[Any],
        inputs: Any
    ) -> Tuple[Any, List[Any]]:
        """
        Compute the Jacobian of the model output with respect to the weights.

        Parameters
        ----------
        model
            The model.
        weights
            The weight tensors.
        inputs
            The input tensor.

        Returns
        -------
        outputs
            The model outputs.
        jacobian
            List of Jacobians of outputs with respect to each weight tensor.
        """

    @abstractmethod
    def boolean_mask(self, tensor: Any, mask: Any) -> Any:
        """
        Apply a boolean mask to a tensor.

        Parameters
        ----------
        tensor
            The input tensor.
        mask
            The boolean mask.

        Returns
        -------
        masked_tensor
            The masked tensor.
        """

    @abstractmethod
    def while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        loop_vars: List[Any],
        maximum_iterations: Optional[int] = None
    ) -> List[Any]:
        """
        Execute a while loop with the given condition and body functions.

        Parameters
        ----------
        cond_fn
            A function that takes loop_vars and returns a boolean tensor.
        body_fn
            A function that takes loop_vars and returns updated loop_vars.
        loop_vars
            Initial values for the loop variables.
        maximum_iterations
            Optional maximum number of iterations.

        Returns
        -------
        loop_vars
            Final values of the loop variables.
        """

    # Arnoldi algorithm specific operations
    @abstractmethod
    def random_normal(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """
        Generate random tensor from normal distribution.

        Parameters
        ----------
        shape
            Shape of the output tensor.
        dtype
            Data type for the tensor.

        Returns
        -------
        tensor
            Random tensor from normal distribution.
        """

    @abstractmethod
    def diag_part(self, tensor: Any, k: int = 0) -> Any:
        """
        Extract diagonal from a matrix with offset k.

        Parameters
        ----------
        tensor
            The input matrix.
        k
            Diagonal offset (0 for main diagonal, positive for upper, negative for lower).

        Returns
        -------
        diagonal
            The diagonal elements.
        """

    @abstractmethod
    def eigh_tridiagonal(
        self,
        maindiag: Any,
        superdiag: Any,
        eigvals_only: bool = False
    ) -> Tuple[Any, Optional[Any]]:
        """
        Compute eigenvalues and eigenvectors of a symmetric tridiagonal matrix.

        Parameters
        ----------
        maindiag
            Main diagonal of the tridiagonal matrix.
        superdiag
            Super diagonal of the tridiagonal matrix.
        eigvals_only
            If True, only compute eigenvalues.

        Returns
        -------
        eig_vals
            The eigenvalues.
        eig_vectors
            The eigenvectors (None if eigvals_only is True).
        """

    @abstractmethod
    def eig(self, tensor: Any) -> Tuple[Any, Any]:
        """
        Compute eigenvalues and eigenvectors of a square matrix.

        Parameters
        ----------
        tensor
            The input square matrix.

        Returns
        -------
        eig_vals
            The eigenvalues.
        eig_vectors
            The eigenvectors.
        """

    @abstractmethod
    def real(self, tensor: Any) -> Any:
        """
        Return the real part of a complex tensor.

        Parameters
        ----------
        tensor
            The input tensor (possibly complex).

        Returns
        -------
        real_tensor
            The real part of the tensor.
        """

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
        tensorflow_backend_cls = import_optional_attr(
            ".backend_tensorflow",
            "TensorFlowBackend",
            package=__package__,
            extra="tensorflow",
        )
        return tensorflow_backend_cls()

    if framework == Framework.PYTORCH:
        pytorch_backend_cls = import_optional_attr(
            ".backend_pytorch",
            "PyTorchBackend",
            package=__package__,
            extra="pytorch",
        )
        return pytorch_backend_cls()

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
