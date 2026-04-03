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
from typing import Any, List, Tuple, Callable, Optional, Union

import numpy as np

from .._optional_imports import import_optional_attr, import_optional_module, is_module_available
from ..types import DatasetLike, DType, ElementSpec, Layer, LossFunction, Model, Tensor, WeightVariable


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
    if is_module_available("tensorflow"):
        available.append(Framework.TENSORFLOW)

    if is_module_available("torch"):
        available.append(Framework.PYTORCH)

    return available


def detect_framework(model: Model) -> Framework:
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


def detect_tensor_framework(tensor: Tensor) -> Framework:
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


def detect_dtype_framework(dtype: DType) -> Optional[Framework]:
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


def get_backend_for_tensor(tensor: Tensor) -> "BaseBackend":
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
    def get_model_weights(
        self,
        model: Model,
        layers: Optional[List[Layer]] = None,
    ) -> List[WeightVariable]:
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
    def clone_model(self, model: Model) -> Model:
        """
        Clone a model and copy its weights.

        Parameters
        ----------
        model
            The model to clone.

        Returns
        -------
        cloned_model
            A cloned model with the same weights.
        """

    @abstractmethod
    def validate_loss_no_reduction(self, loss_function: LossFunction) -> None:
        """
        Validate that a loss function returns per-sample losses.

        Parameters
        ----------
        loss_function
            The loss function to validate.

        Raises
        ------
        ValueError
            If the loss function uses a reduction other than no reduction.
        """

    @abstractmethod
    def is_dense_linear_layer(self, layer: Layer) -> bool:
        """Return whether a layer is a Dense/Linear layer."""

    @abstractmethod
    def layer_has_bias(self, layer: Layer) -> bool:
        """Return whether a layer uses a bias parameter."""

    @abstractmethod
    def get_layer_io_features(self, layer: Layer) -> Tuple[int, int]:
        """
        Return the input and output feature sizes for a Dense/Linear layer.

        Parameters
        ----------
        layer
            The layer to inspect.

        Returns
        -------
        in_features
            Input feature dimension.
        out_features
            Output feature dimension.
        """

    @abstractmethod
    def create_linear_model(
        self,
        input_shape: Tuple[int, ...],
        out_features: int,
        use_bias: bool = False,
        l2_regularization: float = 0.0,
        dtype: Optional[DType] = None,
        reference_weight: Optional[Tensor] = None,
    ) -> Model:
        """
        Create a linear model matching the backend's Dense/Linear semantics.

        Parameters
        ----------
        input_shape
            Shape of a single input sample excluding the batch dimension.
        out_features
            Number of output features.
        use_bias
            Whether the linear layer uses a bias term.
        l2_regularization
            Optional L2 regularization coefficient for the linear weights.
        dtype
            Optional dtype for the created model.
        reference_weight
            Optional tensor/parameter used to align backend-specific dtype/device state.

        Returns
        -------
        model
            A backend-native linear model.
        """

    @abstractmethod
    def get_linear_weight_axes(self) -> Tuple[int, int]:
        """Return the input-axis and output-axis order for linear layer weights."""

    @abstractmethod
    def get_num_params(self, weights: List[WeightVariable]) -> int:
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
    def normalize_weights_to_watch(self, weights: List[WeightVariable]) -> List[WeightVariable]:
        """
        Normalize weights to objects supported by backend autodiff watch APIs.

        Parameters
        ----------
        weights
            Candidate weight objects to be watched.

        Returns
        -------
        normalized_weights
            Framework-compatible watched objects.
        """

    @abstractmethod
    def compute_loss(
        self,
        model: Model,
        loss_function: LossFunction,
        inputs: Tensor,
        targets: Tensor,
        sample_weight: Optional[Tensor] = None
    ) -> Tensor:
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
        model: Model,
        weights: List[WeightVariable],
        loss_function: LossFunction,
        inputs: Tensor,
        targets: Tensor,
        sample_weight: Optional[Tensor] = None
    ) -> Tensor:
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
        model: Model,
        weights: List[WeightVariable],
        loss_function: LossFunction,
        inputs: Tensor,
        targets: Tensor,
        sample_weight: Optional[Tensor] = None
    ) -> Tensor:
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
    def concat(self, tensors: List[Tensor], axis: int = 0) -> Tensor:
        """Concatenate tensors along an axis."""

    @abstractmethod
    def stack(self, tensors: List[Tensor], axis: int = 0) -> Tensor:
        """Stack tensors along a new axis."""

    @abstractmethod
    def reshape(self, tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
        """Reshape a tensor."""

    @abstractmethod
    def to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert a tensor to numpy array."""

    @abstractmethod
    def get_batch_size(self, tensor: Tensor) -> int:
        """Get the batch size (first dimension) of a tensor."""

    @abstractmethod
    def ensure_per_sample_loss(self, loss: Tensor) -> Tensor:
        """
        Ensure that a loss tensor is represented as a per-sample vector.

        Parameters
        ----------
        loss
            Loss tensor returned by a framework loss function.

        Returns
        -------
        per_sample_loss
            A tensor with one scalar loss per sample.

        Raises
        ------
        ValueError
            If the loss is scalar and therefore already reduced.
        """

    @abstractmethod
    def normalize_binary_targets(self, targets: Tensor, logits: Tensor) -> Tensor:
        """
        Normalize binary-classification targets to match logits shape.

        Parameters
        ----------
        targets
            The target tensor.
        logits
            The logits tensor produced by the model.

        Returns
        -------
        normalized_targets
            Targets reshaped if necessary to match binary logits.
        """

    @abstractmethod
    def reduce_sum(self, tensor: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        """Reduce sum along an axis."""

    @abstractmethod
    def expand_dims(self, tensor: Tensor, axis: int) -> Tensor:
        """Add a new axis to a tensor."""

    @abstractmethod
    def squeeze(self, tensor: Tensor, axis: Optional[int] = None) -> Tensor:
        """Remove dimensions of size 1."""

    @abstractmethod
    def transpose(self, tensor: Tensor) -> Tensor:
        """Transpose a tensor (swap last two dimensions)."""

    @abstractmethod
    def tensor_shape(self, tensor: Tensor) -> Tuple[int, ...]:
        """Get the shape of a tensor."""

    @abstractmethod
    def tensor_ndim(self, tensor: Tensor) -> int:
        """Get the number of dimensions of a tensor."""

    @abstractmethod
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication."""

    @abstractmethod
    def multiply(self, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication."""

    @abstractmethod
    def abs(self, tensor: Tensor) -> Tensor:
        """Compute absolute value of a tensor."""

    @abstractmethod
    def argmax(self, tensor: Tensor, axis: int) -> Tensor:
        """Return indices of maximum values along an axis."""

    @abstractmethod
    def gather_along_axis(self, tensor: Tensor, indices: Tensor, axis: int, batch_dims: int = 0) -> Tensor:
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
    def get_output_shape(self, model: Model) -> Tuple[Optional[int], ...]:
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
    def split_model(self, model: Model, target_layer: Union[str, int]) -> Tuple[Model, Model]:
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
    def normalize(self, tensor: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
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
    def find_layer_by_name(self, model: Model, layer_name: str) -> Tuple[int, Layer]:
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
    def find_last_weight_layer(self, model: Model) -> int:
        """Return the negative index of the last layer with trainable weights."""

    @abstractmethod
    def get_layer_index(self, model: Model, layer: Any) -> int:
        """Resolve a layer name/index/None to a concrete non-negative index."""

    @abstractmethod
    def get_layers(self, model: Model) -> List[Layer]:
        """Get all layers from a model."""

    @abstractmethod
    def forward(self, model: Model, inputs: Tensor) -> Tensor:
        """Run forward pass on a model."""

    @abstractmethod
    def get_weights_for_layer_range(
        self,
        model: Model,
        start_layer: Optional[Union[str, int]] = None,
        last_layer: Optional[Union[str, int]] = None
    ) -> List[WeightVariable]:
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
    def map_dataset(self, dataset: DatasetLike, map_fn: Callable, device: Optional[str] = None) -> DatasetLike:
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
    def cache_dataset(self, dataset: DatasetLike) -> DatasetLike:
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
    def save_dataset(self, dataset: DatasetLike, path: str) -> None:
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
    def load_dataset(self, path: str) -> DatasetLike:
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
    def get_dataset_batch_size(self, dataset: DatasetLike) -> int:
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
    def zip_datasets(self, dataset1: DatasetLike, dataset2: DatasetLike) -> DatasetLike:
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
    def batch_dataset(self, dataset: DatasetLike, batch_size: int) -> DatasetLike:
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
    def create_dataset_from_tensors(self, tensors: Union[Tensor, Tuple[Tensor, ...]], batch_size: int) -> DatasetLike:
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
    def create_dataset_from_tensor_slices(
        self,
        tensors: Union[Tensor, Tuple[Tensor, ...]],
        batch_size: int,
    ) -> DatasetLike:
        """
        Create a batched dataset by slicing tensors along the first dimension.

        Parameters
        ----------
        tensors
            A tensor or tuple of tensors sharing the same leading dimension.
        batch_size
            The batch size for the resulting dataset.

        Returns
        -------
        dataset
            A batched dataset containing one element per tensor slice.
        """

    @abstractmethod
    def unbatch_dataset(self, dataset: DatasetLike) -> DatasetLike:
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
    def shuffle_dataset(self, dataset: DatasetLike, buffer_size: int) -> DatasetLike:
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
    def take_dataset(self, dataset: DatasetLike, count: int) -> DatasetLike:
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
    def get_dataset_size(self, dataset: DatasetLike) -> int:
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
    def get_dataset_element_spec(self, dataset: DatasetLike) -> ElementSpec:
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
    def assert_batched_dataset(self, dataset: DatasetLike) -> None:
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
    def zeros(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> Tensor:
        """Create a tensor of zeros."""

    @abstractmethod
    def zeros_like(self, tensor: Tensor) -> Tensor:
        """Create a tensor of zeros with the same shape and dtype as the input."""

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> Tensor:
        """Create a tensor of ones."""

    @abstractmethod
    def ones_like(self, tensor: Tensor) -> Tensor:
        """Create a tensor of ones with the same shape and dtype as the input."""

    @abstractmethod
    def argsort(self, tensor: Tensor, axis: int = -1, descending: bool = False) -> Tensor:
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
    def copy(self, tensor: Tensor) -> Tensor:
        """Create a copy of a tensor."""

    @abstractmethod
    def sqrt(self, tensor: Tensor) -> Tensor:
        """Compute element-wise square root."""

    @abstractmethod
    def maximum(self, a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
        """Element-wise maximum of two tensors/scalars."""

    @abstractmethod
    def pinv(self, matrix: Tensor) -> Tensor:
        """Compute the Moore-Penrose pseudo-inverse of a matrix."""

    @abstractmethod
    def cast(self, tensor: Union[Tensor, float, int, bool], dtype: DType) -> Tensor:
        """Cast a tensor to a different dtype."""

    @abstractmethod
    def get_dtype(self, tensor: Tensor) -> DType:
        """Get the dtype of a tensor."""

    @abstractmethod
    def float32_dtype(self) -> DType:
        """Return the float32 dtype for the framework."""

    @abstractmethod
    def int32_dtype(self) -> DType:
        """Return the int32 dtype for the framework."""

    @abstractmethod
    def int64_dtype(self) -> DType:
        """Return the int64 dtype for the framework."""

    @abstractmethod
    def constant(self, value: Any, dtype: Optional[DType] = None) -> Tensor:
        """Create a constant tensor."""

    @abstractmethod
    def convert_to_tensor(self, value: Any, dtype: Optional[DType] = None) -> Tensor:
        """Convert a value to a tensor."""

    @abstractmethod
    def reduce_prod(self, tensor: Tensor, axis: Optional[int] = None) -> Tensor:
        """Reduce product along an axis."""

    @abstractmethod
    def compute_hessian(
        self,
        model: Model,
        weights: List[WeightVariable],
        loss_function: LossFunction,
        dataset: DatasetLike,
        nb_params: int,
        jacobian_fn: Optional[Callable] = None
    ) -> Tensor:
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
        model: Model,
        weights: List[WeightVariable],
        loss_function: LossFunction,
        v: List[Tensor],
        inputs: Tensor,
        targets: Tensor
    ) -> Tensor:
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
        model: Model,
        weights: List[WeightVariable],
        loss_function: LossFunction,
        v: List[Tensor],
        inputs: Tensor,
        targets: Tensor
    ) -> Tensor:
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
    def map_fn(
        self,
        fn: Callable,
        elems: Union[Tensor, Tuple[Tensor, ...]],
        output_signature: Optional[ElementSpec] = None,
    ) -> Tensor:
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
    def get_dataset_cardinality(self, dataset: DatasetLike) -> int:
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
    def is_sequential_model(self, model: Model) -> bool:
        """Check if a model is a Sequential model."""

    @abstractmethod
    def create_sequential_from_layers(self, layers: List[Layer]) -> Model:
        """Create a Sequential model from a list of layers."""

    # Additional operations for boundary-based calculators
    @abstractmethod
    def norm(  # pylint: disable=redefined-builtin
        self,
        tensor: Tensor,
        ord: Optional[int] = None,
        axis: Optional[int] = None,
    ) -> Tensor:
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
    def top_k(self, tensor: Tensor, k: int) -> Tuple[Tensor, Tensor]:
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
    def arange(self, start: int, end: int, dtype: Optional[DType] = None) -> Tensor:
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
    def tile(self, tensor: Tensor, multiples: Tuple[int, ...]) -> Tensor:
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
    def repeat(self, tensor: Tensor, repeats: int, axis: int) -> Tensor:
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
    def sign(self, tensor: Tensor) -> Tensor:
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
    def pow(self, tensor: Tensor, exponent: Union[Tensor, float, int]) -> Tensor:
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
    def logical_and(self, a: Tensor, b: Tensor) -> Tensor:
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
    def reduce_any(self, tensor: Tensor, axis: Optional[int] = None) -> Tensor:
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
    def argmin(self, tensor: Tensor, axis: int) -> Tensor:
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
    def reduce_mean(self, tensor: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
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
    def clone_variable(self, variable: WeightVariable) -> Tensor:
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
    def assign_variable(self, variable: WeightVariable, value: Tensor) -> None:
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
        model: Model,
        inputs: Tensor
    ) -> Tuple[Tensor, Tensor]:
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
        model: Model,
        weights: List[WeightVariable],
        inputs: Tensor
    ) -> Tuple[Tensor, List[Tensor]]:
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
    def boolean_mask(self, tensor: Tensor, mask: Tensor) -> Tensor:
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
        maximum_iterations: Optional[int] = None,
        parallel_iterations: int = 10,
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
        parallel_iterations
            Backend hint controlling loop parallelism when supported.

        Returns
        -------
        loop_vars
            Final values of the loop variables.
        """

    # Arnoldi algorithm specific operations
    @abstractmethod
    def random_normal(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> Tensor:
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
    def diag_part(self, tensor: Tensor, k: int = 0) -> Tensor:
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
        maindiag: Tensor,
        superdiag: Tensor,
        eigvals_only: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
    def eig(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
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
    def real(self, tensor: Tensor) -> Tensor:
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


def get_backend_for_model(model: Model) -> BaseBackend:
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
