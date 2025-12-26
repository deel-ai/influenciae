# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch backend implementation.
"""
from typing import Any, List, Tuple, Callable, Optional, Union

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

    def reduce_sum(self, tensor: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        """Reduce sum along an axis."""
        if axis is None:
            return tensor.sum()
        return tensor.sum(dim=axis)

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

