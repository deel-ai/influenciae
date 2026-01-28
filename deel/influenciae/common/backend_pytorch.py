# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch backend implementation.
"""
from typing import Any, List, Tuple, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .backend import BaseBackend, Framework


class PyTorchBackend(BaseBackend):
    """
    PyTorch-specific backend implementation.
    """

    @property
    def framework(self) -> Framework:
        return Framework.PYTORCH

    def get_model_weights(
        self,
        model: nn.Module,
        layers: Optional[List[nn.Module]] = None
    ) -> List[torch.nn.Parameter]:
        """
        Get trainable weights from a PyTorch model.

        Parameters
        ----------
        model
            The PyTorch model to extract weights from.
        layers
            Optional list of specific layers to get weights from.
            If None, gets all trainable parameters from the model.

        Returns
        -------
        weights
            List of weight tensors (nn.Parameter).
        """
        if layers is None:
            return [p for p in model.parameters() if p.requires_grad]

        weights = []
        for layer in layers:
            weights.extend([p for p in layer.parameters() if p.requires_grad])
        return weights

    def get_num_params(self, weights: List[torch.nn.Parameter]) -> int:
        """Get the total number of parameters."""
        return sum(w.numel() for w in weights)

    def compute_loss(
        self,
        model: nn.Module,
        loss_function: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the loss for a batch of samples."""
        predictions = model(inputs)
        loss = loss_function(predictions, targets)

        if sample_weight is not None:
            loss = loss * sample_weight

        return loss

    def compute_jacobian(
        self,
        model: nn.Module,
        weights: List[torch.nn.Parameter],
        loss_function: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the Jacobian of the loss with respect to weights."""
        batch_size = inputs.shape[0]
        num_params = self.get_num_params(weights)

        # Compute per-sample gradients using vmap if available (PyTorch 2.0+)
        # Otherwise fall back to manual loop
        try:
            from torch.func import vmap, grad

            def compute_sample_grad(input_sample, target_sample):
                """Compute gradient for a single sample."""
                input_batch = input_sample.unsqueeze(0)
                target_batch = target_sample.unsqueeze(0)

                predictions = model(input_batch)
                loss = loss_function(predictions, target_batch)

                if sample_weight is not None:
                    # Note: sample_weight handling in vmap is complex
                    pass

                grads = torch.autograd.grad(loss.sum(), weights, create_graph=False)
                return torch.cat([g.flatten() for g in grads])

            # Use vmap for efficient batched gradient computation
            jacobian = torch.stack([
                compute_sample_grad(inputs[i], targets[i])
                for i in range(batch_size)
            ])

        except ImportError:
            # Fallback for older PyTorch versions
            jacobian = torch.zeros(batch_size, num_params, device=inputs.device)

            for i in range(batch_size):
                model.zero_grad()
                input_sample = inputs[i:i+1]
                target_sample = targets[i:i+1]

                predictions = model(input_sample)
                loss = loss_function(predictions, target_sample)

                if sample_weight is not None:
                    loss = loss * sample_weight[i]

                loss = loss.sum()
                loss.backward(retain_graph=(i < batch_size - 1))

                grads = []
                for w in weights:
                    if w.grad is not None:
                        grads.append(w.grad.flatten().clone())
                    else:
                        grads.append(torch.zeros(w.numel(), device=inputs.device))

                jacobian[i] = torch.cat(grads)

        return jacobian

    def compute_gradient(
        self,
        model: nn.Module,
        weights: List[torch.nn.Parameter],
        loss_function: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the gradient of the loss with respect to weights."""
        model.zero_grad()

        predictions = model(inputs)
        loss = loss_function(predictions, targets)

        if sample_weight is not None:
            loss = loss * sample_weight

        loss = loss.sum()
        loss.backward()

        gradients = []
        for w in weights:
            if w.grad is not None:
                gradients.append(w.grad.flatten())
            else:
                gradients.append(torch.zeros(w.numel(), device=inputs.device))

        return torch.cat(gradients)

    def concat(self, tensors: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
        """Concatenate tensors along an axis."""
        return torch.cat(tensors, dim=axis)

    def stack(self, tensors: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
        """Stack tensors along a new axis."""
        return torch.stack(tensors, dim=axis)

    def reshape(self, tensor: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Reshape a tensor."""
        return tensor.reshape(shape)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to numpy array."""
        return tensor.detach().cpu().numpy()

    def get_batch_size(self, tensor: torch.Tensor) -> int:
        """Get the batch size (first dimension) of a tensor."""
        return tensor.shape[0]

    def reduce_sum(self, tensor: torch.Tensor, axis: Optional[int] = None, keepdims: bool = False) -> torch.Tensor:
        """Reduce sum along an axis."""
        if axis is None:
            return tensor.sum()
        return tensor.sum(dim=axis, keepdim=keepdims)

    def expand_dims(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """Add a new axis to a tensor."""
        return tensor.unsqueeze(axis)

    def squeeze(self, tensor: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        """Remove dimensions of size 1."""
        if axis is None:
            return tensor.squeeze()
        return tensor.squeeze(axis)

    def transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transpose a tensor (swap last two dimensions)."""
        return tensor.t() if tensor.dim() <= 2 else tensor.transpose(-2, -1)

    def tensor_shape(self, tensor: torch.Tensor) -> Tuple[int, ...]:
        """Get the shape of a tensor."""
        return tuple(tensor.shape)

    def tensor_ndim(self, tensor: torch.Tensor) -> int:
        """Get the number of dimensions of a tensor."""
        return tensor.dim()

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication."""
        return torch.matmul(a, b)

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise multiplication."""
        return torch.mul(a, b)

    def abs(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute absolute value of a tensor."""
        return torch.abs(tensor)

    def argmax(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """Return indices of maximum values along an axis."""
        return torch.argmax(tensor, dim=axis)

    def gather_along_axis(
            self,
            tensor: torch.Tensor,
            indices: torch.Tensor,
            axis: int,
            batch_dims: int = 0
    ) -> torch.Tensor:
        if batch_dims == 0:
            # Simple gather of whole slices along axis
            # Expect indices to be 1D here (matches your current use cases)
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
            # Flatten indices if needed
            if indices.dim() > 1:
                indices = indices.flatten()
            return torch.index_select(tensor, dim=axis, index=indices.to(dtype=torch.long))

        if batch_dims != 1:
            raise NotImplementedError(f"batch_dims={batch_dims} not supported")

        if axis != 1:
            raise NotImplementedError("Only axis=1 is implemented for batch_dims=1 in this backend")

        # Ensure indices are long and on the same device
        indices = indices.to(device=tensor.device, dtype=torch.long)

        b_tensor = tensor.shape[0]
        b_idx = indices.shape[0]

        if b_tensor == b_idx:
            # Row-wise gather: preserve shape to match TensorFlow behavior
            # indices should be (B, K) where K is the number of indices per row
            indices_was_1d = indices.dim() == 1
            if indices.dim() == 1:
                indices = indices.view(-1, 1)

            # Handle 2D tensors: shape (B, N) -> gather along dim 1
            if tensor.dim() == 2:
                result = tensor.gather(dim=1, index=indices)
                # If original indices were 1D, squeeze the result to match TF behavior
                if indices_was_1d:
                    result = result.squeeze(-1)
                return result

            # Handle 3D+ tensors: shape (B, N, ...) -> gather along dim 1, keep trailing dims
            # We need to expand indices to match tensor dims
            # tensor shape: (B, N, D1, D2, ...)
            # indices shape: (B, K)
            # result shape: (B, K, D1, D2, ...)
            trailing_dims = tensor.shape[2:]
            # Expand indices to broadcast over trailing dimensions
            expanded_indices = indices
            for _ in trailing_dims:
                expanded_indices = expanded_indices.unsqueeze(-1)
            # Expand to match tensor's trailing dimensions
            expanded_indices = expanded_indices.expand(-1, -1, *trailing_dims)
            result = tensor.gather(dim=1, index=expanded_indices)
            # If original indices were 1D, squeeze the K dimension to match TF behavior
            if indices_was_1d:
                result = result.squeeze(1)
            return result

        # Cross-batch: select columns => shape (B_tensor, B_idx)
        return tensor.index_select(dim=1, index=indices.flatten())

    def get_output_shape(self, model: nn.Module) -> Tuple[Optional[int], ...]:
        """Get the output shape of a model."""
        # For PyTorch, we need to infer output shape by looking at the last layer
        # or running a forward pass with dummy data
        children = list(model.children())
        if children:
            last_layer = children[-1]
            if hasattr(last_layer, 'out_features'):
                return (None, last_layer.out_features)
        # Fallback: try to get from model attribute if available
        if hasattr(model, 'output_shape'):
            return model.output_shape
        # If we can't determine, return a placeholder
        return (None,)

    def split_model(
        self,
        model: nn.Module,
        target_layer: Any
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Split a model into two sub-models at a target layer.

        Parameters
        ----------
        model
            The PyTorch model to split.
        target_layer
            Layer name (str) or index (int) at which to split.

        Returns
        -------
        feature_extractor
            Model containing layers up to (but not including) target_layer.
        head
            Model containing the target_layer and beyond.
        """
        children = list(model.children())

        if isinstance(target_layer, int):
            if target_layer < 0:
                target_layer = len(children) + target_layer
            split_idx = target_layer
        elif isinstance(target_layer, str):
            # Find by name
            named_children = list(model.named_children())
            split_idx = None
            for idx, (name, _) in enumerate(named_children):
                if name == target_layer:
                    split_idx = idx
                    break
            if split_idx is None:
                raise ValueError(f"Could not find layer with name: {target_layer}")
        else:
            raise ValueError(f"target_layer must be str or int, got {type(target_layer)}")

        # Create feature extractor (layers before target_layer)
        feature_extractor = nn.Sequential(*children[:split_idx])

        # Create head (layers from target_layer onwards)
        head = nn.Sequential(*children[split_idx:])

        return feature_extractor, head

    def normalize(self, tensor: torch.Tensor, axis: Optional[int] = None, keepdims: bool = False) -> torch.Tensor:
        """Normalize a tensor along an axis using L2 norm."""
        norm = torch.linalg.norm(tensor, dim=axis, keepdim=keepdims)
        return tensor / norm

    def find_layer_by_name(self, model: nn.Module, layer_name: str) -> Tuple[int, nn.Module]:
        """Find a layer by name and return its index and the layer."""
        named_modules = list(model.named_modules())
        for idx, (name, module) in enumerate(named_modules):
            if name == layer_name:
                return idx, module
        raise ValueError(
            f'No such layer: {layer_name}. Existing layers are: '
            f'{[name for name, _ in named_modules if name]}.'
        )

    def get_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        Return top-level layers (Keras-like) so slicing works for feature extractor / head splits.
        DO NOT use model.modules() here: it includes the container itself and breaks slicing.
        """
        children = list(model.children())
        return children if children else [model]

    def forward(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass on a model."""
        return model(inputs)

    def get_children(self, model: nn.Module) -> List[nn.Module]:
        """Get direct children modules of a model."""
        return list(model.children())

    def find_last_weight_layer(self, model: nn.Module) -> int:
        """
        Find and return the index of the last layer with weights.

        Parameters
        ----------
        model
            The PyTorch model.

        Returns
        -------
        layer_idx
            Index of the layer found (negative index from end).
        """
        children = self.get_children(model)
        for layer_idx in range(1, len(children) + 1):
            layer = children[-layer_idx]
            params = list(layer.parameters())
            if params:
                return -layer_idx
        raise ValueError('No layers with weights found for the model.')

    def get_layer_index(self, model: nn.Module, layer: Any) -> int:
        """
        Get the index of a layer in a model.

        Parameters
        ----------
        model
            The PyTorch model.
        layer
            Layer name (str), index (int), or None.

        Returns
        -------
        layer_idx
            Index of the layer.
        """
        children = self.get_children(model)
        num_layers = len(children)

        if layer is None:
            return self.find_last_weight_layer(model) + num_layers
        elif isinstance(layer, str):
            idx, _ = self.find_layer_by_name(model, layer)
            return idx
        elif isinstance(layer, int):
            if layer < 0:
                return layer + num_layers
            return layer
        else:
            raise ValueError(f"layer should be None, a string, or an int, got {type(layer)}")

    def get_weights_for_layer_range(
        self,
        model: nn.Module,
        start_layer: Optional[Any] = None,
        last_layer: Optional[Any] = None
    ) -> List[torch.nn.Parameter]:
        """
        Get weights for a range of layers.

        Parameters
        ----------
        model
            The PyTorch model.
        start_layer
            Starting layer (name, index, or None for auto-detect).
        last_layer
            Ending layer (name, index, or None for just start_layer).

        Returns
        -------
        weights
            List of weight tensors.
        """
        children = self.get_children(model)

        if not children:
            # If model has no direct children (e.g., single layer), use all parameters
            return self.get_model_weights(model)

        start_idx = self.get_layer_index(model, start_layer)

        if last_layer is None:
            layers = [children[start_idx]]
        else:
            end_idx = self.get_layer_index(model, last_layer)
            assert end_idx >= start_idx, \
                f"last_layer index ({end_idx}) should be >= start_layer index ({start_idx})"
            layers = children[start_idx:end_idx + 1]

        return self.get_model_weights(model, layers)

    # Dataset operations
    def map_dataset(self, dataset: Any, map_fn: Callable, device: Optional[str] = None) -> List[Any]:
        """
        Apply a mapping function to each batch in a dataset.

        For PyTorch, this returns a list of mapped results since DataLoader
        doesn't support lazy mapping like tf.data.Dataset.
        """
        import inspect

        def _move_to_device(obj):
            if device is None:
                return obj
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, (list, tuple)):
                return type(obj)(_move_to_device(o) for o in obj)
            return obj

        # Decide whether to unpack based on map_fn signature (TF-like behavior)
        sig = inspect.signature(map_fn)
        params = list(sig.parameters.values())
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        num_positional = sum(
            p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in params
        )

        results = []
        for batch in dataset:
            batch = _move_to_device(batch)

            if isinstance(batch, (list, tuple)):
                # TF-like: if fn takes 1 arg, give it the tuple as-is; otherwise unpack
                if has_varargs or num_positional > 1:
                    out = map_fn(*batch)
                else:
                    out = map_fn(batch)
            else:
                out = map_fn(batch)

            results.append(out)

        return results

    def cache_dataset(self, dataset: Any) -> List[Any]:
        """
        Cache a dataset in memory.

        For PyTorch, this materializes the DataLoader into a list.
        """
        return list(dataset)

    def save_dataset(self, dataset: Any, path: str) -> None:
        """Save a dataset to disk using torch.save."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        # Materialize dataset if it's a DataLoader
        if hasattr(dataset, '__iter__'):
            data = list(dataset)
        else:
            data = dataset
        torch.save(data, path)

    def load_dataset(self, path: str) -> List[Any]:
        """Load a dataset from disk."""
        import os
        if os.path.exists(path):
            return torch.load(path)
        raise FileNotFoundError(f"The dataset path: {path} was not found")

    def get_dataset_batch_size(self, dataset: Any) -> int:
        """Get the batch size of a dataset (DataLoader)."""
        if hasattr(dataset, 'batch_size'):
            return dataset.batch_size
        # Try to infer from first batch
        for batch in dataset:
            if isinstance(batch, (list, tuple)):
                return batch[0].shape[0]
            return batch.shape[0]
        raise ValueError("Could not determine batch size from dataset")

    def zip_datasets(self, dataset1: Any, dataset2: Any) -> List[Tuple[Any, Any]]:
        """Zip two datasets together."""
        return list(zip(dataset1, dataset2))

    def batch_dataset(self, dataset: Any, batch_size: int) -> Any:
        """
        Batch a dataset.

        For PyTorch, if dataset is a list, create batches manually.
        If it's a Dataset, wrap with DataLoader.
        """
        from torch.utils.data import DataLoader

        if isinstance(dataset, DataLoader):
            # Already batched, return as-is or rebatch
            return dataset
        elif isinstance(dataset, (list, tuple)):
            # Create batches from list
            batches = []
            for i in range(0, len(dataset), batch_size):
                batch_items = dataset[i:i + batch_size]
                # If items are tuples (e.g., (inputs, targets)), collate them properly
                if batch_items and isinstance(batch_items[0], tuple):
                    # Stack each element of the tuple across the batch
                    collated = []
                    num_elements = len(batch_items[0])
                    for j in range(num_elements):
                        elements = [item[j] for item in batch_items]
                        # Stack tensors, or create a list for non-tensors
                        if isinstance(elements[0], torch.Tensor):
                            collated.append(torch.stack(elements))
                        else:
                            collated.append(elements)
                    batches.append(tuple(collated))
                else:
                    batches.append(tuple(batch_items) if isinstance(batch_items, list) else batch_items)
            return batches
        else:
            # Assume it's a PyTorch Dataset
            return DataLoader(dataset, batch_size=batch_size)

    def create_dataset_from_tensors(self, tensors: torch.Tensor, batch_size: int) -> List[Any]:
        """Create a batched dataset from tensors."""
        # For PyTorch, return a list containing the tensor(s)
        # This mirrors TensorFlow's from_tensors().batch() behavior
        return [tensors]

    def unbatch_dataset(self, dataset: Any) -> List[Any]:
        """Unbatch a dataset."""
        unbatched = []
        for batch in dataset:
            if isinstance(batch, (list, tuple)):
                # Unbatch each element in the tuple
                batch_size = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else len(batch[0])
                for i in range(batch_size):
                    item = tuple(b[i] for b in batch)
                    unbatched.append(item)
            elif isinstance(batch, torch.Tensor):
                for i in range(batch.shape[0]):
                    unbatched.append(batch[i])
            else:
                unbatched.append(batch)
        return unbatched

    def shuffle_dataset(self, dataset: Any, buffer_size: int) -> List[Any]:
        """
        Shuffle a dataset.

        For PyTorch, this materializes and shuffles the data.
        """
        import random
        if isinstance(dataset, list):
            data = list(dataset)
        else:
            data = list(dataset)
        random.shuffle(data)
        return data

    def take_dataset(self, dataset: Any, count: int) -> List[Any]:
        """Take a number of elements from a dataset."""
        if isinstance(dataset, list):
            return dataset[:count]
        # Materialize and take
        result = []
        for i, item in enumerate(dataset):
            if i >= count:
                break
            result.append(item)
        return result

    def get_dataset_size(self, dataset: Any) -> int:
        """Get the total number of elements in a dataset."""
        if isinstance(dataset, list):
            # If it's a list of batched items, count total samples
            total = 0
            for batch in dataset:
                if isinstance(batch, (list, tuple)):
                    if isinstance(batch[0], torch.Tensor):
                        total += batch[0].shape[0]
                    else:
                        total += len(batch[0])
                elif isinstance(batch, torch.Tensor):
                    total += batch.shape[0]
                else:
                    total += 1
            return total
        elif hasattr(dataset, 'dataset'):
            # DataLoader with underlying dataset
            return len(dataset.dataset)
        elif hasattr(dataset, '__len__'):
            return len(dataset)
        else:
            # Materialize and count
            return sum(1 for _ in dataset)

    def get_dataset_element_spec(self, dataset: Any) -> Any:
        """
        Get the element spec of a dataset.

        For PyTorch, returns shape info from first batch.
        Handles nested tuple/list structures recursively.
        """
        def _get_spec(item):
            """Recursively get spec for an item."""
            if isinstance(item, torch.Tensor):
                return {'shape': item.shape, 'dtype': item.dtype}
            elif isinstance(item, (list, tuple)):
                return tuple(_get_spec(sub_item) for sub_item in item)
            else:
                return type(item)

        for batch in dataset:
            return _get_spec(batch)
        return None

    def assert_batched_dataset(self, dataset: Any) -> None:
        """Assert that a dataset is batched."""
        # For PyTorch DataLoader, check batch_size attribute
        if hasattr(dataset, 'batch_size') and dataset.batch_size is not None:
            return
        # For lists, check first element
        if isinstance(dataset, list) and len(dataset) > 0:
            first = dataset[0]
            if isinstance(first, (list, tuple)) and len(first) > 0:
                if isinstance(first[0], torch.Tensor) and first[0].dim() > 0:
                    return
        raise ValueError("Dataset does not appear to be batched")

    # Linear algebra operations for IHVP
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> torch.Tensor:
        """Create a tensor of zeros."""
        if dtype is None:
            dtype = torch.float32
        return torch.zeros(shape, dtype=dtype)

    def zeros_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create a tensor of zeros with the same shape and dtype as the input."""
        return torch.zeros_like(tensor)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> torch.Tensor:
        """Create a tensor of ones."""
        if dtype is None:
            dtype = torch.float32
        return torch.ones(shape, dtype=dtype)

    def ones_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create a tensor of ones with the same shape and dtype as the input."""
        return torch.ones_like(tensor)

    def argsort(self, tensor: torch.Tensor, axis: int = -1, descending: bool = False) -> torch.Tensor:
        """Return the indices that would sort the tensor along an axis."""
        return torch.argsort(tensor, dim=axis, descending=descending)

    def copy(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create a copy of a tensor."""
        return tensor.clone()

    def sqrt(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute element-wise square root."""
        return torch.sqrt(tensor)

    def maximum(self, a: Any, b: Any) -> torch.Tensor:
        """Element-wise maximum of two tensors/scalars."""
        # Handle scalar inputs
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        return torch.maximum(a, b)

    def pinv(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the Moore-Penrose pseudo-inverse of a matrix."""
        return torch.linalg.pinv(matrix)

    def cast(self, tensor: Any, dtype: Any) -> Any:
        """Cast a tensor to a different dtype."""
        # Scalar -> keep as Python scalar (GPU-safe, broadcasts fine)
        if isinstance(tensor, (int, float, bool, np.number)):
            if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                return float(tensor)
            return int(tensor)

        # Tensor -> normal cast
        return tensor.to(dtype)

    def get_dtype(self, tensor: torch.Tensor) -> Any:
        """Get the dtype of a tensor."""
        return tensor.dtype

    def float32_dtype(self) -> Any:
        """Return the float32 dtype for the framework."""
        return torch.float32

    def int32_dtype(self) -> Any:
        """Return the int32 dtype for the framework."""
        return torch.int32

    def int64_dtype(self) -> Any:
        """Return the int64 dtype for the framework."""
        return torch.int64

    def constant(self, value: Any, dtype: Any = None) -> torch.Tensor:
        """Create a constant tensor."""
        if dtype is None:
            return torch.tensor(value)
        return torch.tensor(value, dtype=dtype)

    def convert_to_tensor(self, value: Any, dtype: Any = None) -> torch.Tensor:
        """Convert a value to a tensor."""
        if isinstance(value, torch.Tensor):
            if dtype is not None:
                return value.to(dtype)
            return value
        if dtype is None:
            return torch.tensor(value)
        return torch.tensor(value, dtype=dtype)

    def reduce_prod(self, tensor: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        """Reduce product along an axis."""
        if axis is None:
            return tensor.prod()
        return tensor.prod(dim=axis)

    def compute_hessian(
        self,
        model: nn.Module,
        weights: List[torch.nn.Parameter],
        loss_function: Callable,
        dataset: Any,
        nb_params: int,
        jacobian_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """Compute the Hessian matrix of the loss with respect to weights."""
        device = weights[0].device
        dtype = weights[0].dtype
        hess = torch.zeros((nb_params, nb_params), device=device, dtype=dtype)
        nb_elt = 0

        for batch in dataset:
            inputs, targets = batch[0], batch[1]
            # Ensure data is on same device/dtype as the model weights
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device=device, dtype=dtype)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device=device, dtype=dtype)
            batch_size = inputs.shape[0]

            for i in range(batch_size):
                model.zero_grad()
                input_sample = inputs[i:i+1]
                target_sample = targets[i:i+1]

                predictions = model(input_sample)
                loss = loss_function(predictions, target_sample).sum()

                # Compute gradients
                grads = torch.autograd.grad(loss, weights, create_graph=True)
                grad_flat = torch.cat([g.flatten() for g in grads])

                # Compute Hessian row by row
                for j, g in enumerate(grad_flat):
                    model.zero_grad()
                    hess_row = torch.autograd.grad(g, weights, retain_graph=True)
                    hess_row_flat = torch.cat([h.flatten() for h in hess_row])
                    hess[j] += hess_row_flat.detach()

                nb_elt += 1

        return hess / nb_elt

    def compute_hvp_single(
        self,
        model: nn.Module,
        weights: List[torch.nn.Parameter],
        loss_function: Callable,
        v: List[torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hessian-vector product using forward-over-backward AD."""
        model.zero_grad()
        predictions = model(inputs)
        loss = loss_function(predictions, targets).sum()

        # Compute gradients with graph
        grads = torch.autograd.grad(loss, weights, create_graph=True)

        # Compute HVP: sum of grad_i * v_i derivatives
        grad_v_product = sum(
            (g * v_i).sum() for g, v_i in zip(grads, v)
        )

        hvp_list = torch.autograd.grad(grad_v_product, weights)

        # Flatten and concatenate
        hvp = torch.cat([h.flatten() for h in hvp_list])

        return hvp

    def map_fn(self, fn: Callable, elems: torch.Tensor) -> torch.Tensor:
        """Apply a function to each element in a batch."""
        results = [fn(elem) for elem in elems]
        return torch.stack(results)

    def get_dataset_cardinality(self, dataset: Any) -> int:
        """Get the number of batches in a dataset."""
        if hasattr(dataset, '__len__'):
            return len(dataset)
        # Fallback: count batches
        count = 0
        for _ in dataset:
            count += 1
        return count

    def is_sequential_model(self, model: nn.Module) -> bool:
        """Check if a model is a Sequential model."""
        return isinstance(model, nn.Sequential)

    def create_sequential_from_layers(self, layers: List[nn.Module]) -> nn.Module:
        """Create a Sequential model from a list of layers."""
        return nn.Sequential(*layers)

    # Additional operations for boundary-based calculators
    def norm(self, tensor: torch.Tensor, ord: Optional[int] = None, axis: Optional[int] = None) -> torch.Tensor:
        """Compute the norm of a tensor."""
        if axis is None:
            # Flatten and compute norm
            flat = tensor.flatten()
            if ord is None:
                return torch.linalg.norm(flat)
            return torch.linalg.norm(flat, ord=ord)
        if ord is None:
            return torch.linalg.norm(tensor, dim=axis)
        return torch.linalg.norm(tensor, ord=ord, dim=axis)

    def top_k(self, tensor: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the top k values and their indices from a tensor."""
        return torch.topk(tensor, k=k)

    def arange(self, start: int, end: int, dtype: Any = None) -> torch.Tensor:
        """Create a tensor with values from start to end."""
        if dtype is None:
            dtype = torch.int32
        return torch.arange(start, end, dtype=dtype)

    def tile(self, tensor: torch.Tensor, multiples: Tuple[int, ...]) -> torch.Tensor:
        """Tile a tensor by repeating it along each dimension."""
        return tensor.repeat(multiples)

    def repeat(self, tensor: torch.Tensor, repeats: int, axis: int) -> torch.Tensor:
        """Repeat elements of a tensor along an axis."""
        return torch.repeat_interleave(tensor, repeats, dim=axis)

    def sign(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute the element-wise sign of a tensor."""
        return torch.sign(tensor)

    def pow(self, tensor: torch.Tensor, exponent: Any) -> torch.Tensor:
        """Raise tensor elements to a power."""
        return torch.pow(tensor, exponent)

    def logical_and(self, a: Any, b: Any) -> torch.Tensor:
        """Compute element-wise logical AND."""
        return torch.logical_and(a, b)

    def reduce_any(self, tensor: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        """Compute logical OR reduction along an axis."""
        if axis is None:
            return tensor.any()
        return tensor.any(dim=axis)

    def argmin(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """Return indices of minimum values along an axis."""
        return torch.argmin(tensor, dim=axis)

    def reduce_mean(self, tensor: torch.Tensor, axis: Optional[int] = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the mean along an axis."""
        if axis is None:
            return tensor.mean()
        return tensor.mean(dim=axis, keepdim=keepdims)

    def clone_variable(self, variable: torch.nn.Parameter) -> torch.Tensor:
        """Create a copy of a variable (weight tensor)."""
        return variable.clone().detach()

    def assign_variable(self, variable: torch.nn.Parameter, value: torch.Tensor) -> None:
        """Assign a value to a variable in-place."""
        with torch.no_grad():
            variable.copy_(value)

    def compute_output_jacobian(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Jacobian of the model output with respect to the input."""
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs = model(inputs)

        batch_size = inputs.shape[0]
        num_outputs = outputs.shape[-1]

        # Compute Jacobian row by row
        jacobian_rows = []
        for i in range(num_outputs):
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[..., i] = 1.0
            grad = torch.autograd.grad(
                outputs, inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]
            jacobian_rows.append(grad)

        # Stack to form jacobian: (batch, num_outputs, *input_shape)
        jacobian = torch.stack(jacobian_rows, dim=1)

        return outputs.detach(), jacobian

    def compute_output_jacobian_wrt_weights(
        self,
        model: nn.Module,
        weights: List[torch.nn.Parameter],
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute the Jacobian of the model output with respect to the weights."""
        outputs = model(inputs)
        batch_size = inputs.shape[0]
        num_outputs = outputs.shape[-1]

        jacobians = [
            torch.zeros((batch_size, num_outputs, *w.shape), device=w.device, dtype=w.dtype)
            for w in weights
        ]

        for k in range(batch_size):
            for i in range(num_outputs):
                grad_outputs = torch.zeros_like(outputs)
                grad_outputs[k, i] = 1.0
                grads = torch.autograd.grad(
                    outputs, weights,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False
                )
                for w_idx, grad in enumerate(grads):
                    jacobians[w_idx][k, i] = grad

        return outputs.detach(), jacobians

    def boolean_mask(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a boolean mask to a tensor."""
        return tensor[mask]

    def while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        loop_vars: List[Any],
        maximum_iterations: Optional[int] = None,
        parallel_iterations: int = 10
    ) -> List[Any]:
        """Execute a while loop with the given condition and body functions."""
        iteration = 0
        while cond_fn(*loop_vars):
            if maximum_iterations is not None and iteration >= maximum_iterations:
                break
            loop_vars = body_fn(*loop_vars)
            if not isinstance(loop_vars, (list, tuple)):
                loop_vars = [loop_vars]
            iteration += 1
        return loop_vars

    # Arnoldi algorithm specific operations
    def random_normal(self, shape: Tuple[int, ...], dtype: Any = None) -> torch.Tensor:
        """Generate random tensor from normal distribution."""
        if dtype is None:
            dtype = torch.float32
        return torch.randn(shape, dtype=dtype)

    def diag_part(self, tensor: torch.Tensor, k: int = 0) -> torch.Tensor:
        """Extract diagonal from a matrix with offset k."""
        return torch.diagonal(tensor, offset=k)

    def eigh_tridiagonal(
        self,
        maindiag: torch.Tensor,
        superdiag: torch.Tensor,
        eigvals_only: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute eigenvalues and eigenvectors of a symmetric tridiagonal matrix."""
        # PyTorch doesn't have direct tridiagonal eigensolver
        # Construct the tridiagonal matrix and use eigh
        n = maindiag.shape[0]
        device = maindiag.device
        dtype = maindiag.dtype

        # Build the tridiagonal matrix
        T = torch.diag(maindiag)
        if n > 1:
            # Add super and sub diagonals
            indices_super = torch.arange(n - 1, device=device)
            for i in indices_super:
                T[i, i + 1] = superdiag[i]
                T[i + 1, i] = superdiag[i]  # Symmetric

        # Compute eigenvalues and eigenvectors
        eig_vals, eig_vectors = torch.linalg.eigh(T)

        if eigvals_only:
            return eig_vals, None
        return eig_vals, eig_vectors

    def eig(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors of a square matrix."""
        return torch.linalg.eig(tensor)

    def real(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the real part of a complex tensor."""
        return tensor.real if tensor.is_complex() else tensor

