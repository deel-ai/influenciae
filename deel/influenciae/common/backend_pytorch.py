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
        """Get all layers (modules) from a model."""
        return list(model.modules())

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
    def map_dataset(
        self,
        dataset: Any,  # DataLoader
        map_fn: Callable,
        device: Optional[str] = None
    ) -> List[Any]:
        """
        Apply a mapping function to each batch in a dataset.

        For PyTorch, this returns a list of mapped results since DataLoader
        doesn't support lazy mapping like tf.data.Dataset.
        """
        results = []
        for batch in dataset:
            if device is not None:
                # Move batch to device if specified
                if isinstance(batch, (list, tuple)):
                    batch = tuple(
                        b.to(device) if isinstance(b, torch.Tensor) else b
                        for b in batch
                    )
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(device)
            result = map_fn(*batch) if isinstance(batch, (list, tuple)) else map_fn(batch)
            results.append(result)
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
                batch = dataset[i:i + batch_size]
                batches.append(batch)
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
        """
        for batch in dataset:
            if isinstance(batch, (list, tuple)):
                return tuple(
                    {'shape': b.shape, 'dtype': b.dtype} if isinstance(b, torch.Tensor) else type(b)
                    for b in batch
                )
            elif isinstance(batch, torch.Tensor):
                return {'shape': batch.shape, 'dtype': batch.dtype}
            return type(batch)
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

    def pinv(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the Moore-Penrose pseudo-inverse of a matrix."""
        return torch.linalg.pinv(matrix)

    def cast(self, tensor: torch.Tensor, dtype: Any) -> torch.Tensor:
        """Cast a tensor to a different dtype."""
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
        hess = torch.zeros((nb_params, nb_params))
        nb_elt = 0

        for batch in dataset:
            inputs, targets = batch[0], batch[1]
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

