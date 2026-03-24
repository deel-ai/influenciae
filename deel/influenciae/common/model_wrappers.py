# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Framework-agnostic model wrappers for influence functions.
Supports both TensorFlow and PyTorch models.
"""
from typing import Any, Callable, List, Optional, Tuple, Union

from .._optional_imports import import_optional_attr, import_optional_module
from .backend import BaseBackend, Framework, get_backend_for_model
from ..types import DatasetLike, Layer, LossFunction, Model, Tensor, WeightVariable

# Type aliases
ProcessBatchTypeAlias = Callable[[Tuple[Any, ...]], Tuple[Tensor, Tensor, Optional[Tensor]]]


def default_process_batch(batch: Tuple[Any, ...]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Default batch processing function.

    Parameters
    ----------
    batch
        A batch tuple of (inputs, labels) or (inputs, labels, sample_weights).

    Returns
    -------
    inputs
        The input data.
    labels
        The labels/targets.
    sample_weight
        Sample weights (None if not provided).
    """
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    raise ValueError(f"Batch should have 2 or 3 elements, got {len(batch)}")


class BaseInfluenceModel:
    """
    A framework-agnostic model wrapper for Influence functions that facilitates the access to the
    weights for which these quantities are to be computed.

    Supports both TensorFlow (tf.keras.Model) and PyTorch (nn.Module) models.

    Attributes
    ----------
    model
        Model used for computing influence score.
    weights
        List of the model weights to watch when computing gradients, jacobians & hessians.
    loss_function
        Loss function to calculate influence. Make sure not to apply any reduction.
    process_batch_for_loss_fn
        A callable for preprocessing the batch to transform it into a format that can be treated
        by the algorithm: (inputs, label, sample weight).
    backend
        The framework-specific backend for operations.
    """

    def __init__(
        self,
        model: Model,
        weights_to_watch: Optional[List[WeightVariable]] = None,
        loss_function: Optional[LossFunction] = None,
        process_batch_for_loss_fn: ProcessBatchTypeAlias = default_process_batch,
        weights_processed: bool = False
    ):
        """
        Initialize the BaseInfluenceModel.

        Parameters
        ----------
        model
            The model (tf.keras.Model or nn.Module) used for computing influence.
        weights_to_watch
            List of weights to watch. If None, all trainable weights are used.
        loss_function
            Loss function to use. If None, a default cross-entropy loss is used.
        process_batch_for_loss_fn
            Function to preprocess batches into (inputs, labels, sample_weights).
        weights_processed
            Whether weights_to_watch is already in the correct format.
        """
        self.model = model
        self.backend: BaseBackend = get_backend_for_model(model)
        self.process_batch_for_loss_fn = process_batch_for_loss_fn
        self.weights_processed = weights_processed

        # Set default loss function based on framework
        if loss_function is None:
            loss_function = self._get_default_loss_function()
        self.loss_function = loss_function

        # Validate loss function reduction (TensorFlow specific)
        self._validate_loss_function(loss_function)

        # Get weights to watch
        if weights_to_watch is None:
            weights_to_watch = self.backend.get_model_weights(model)
            self.weights_processed = True

        if not self.weights_processed:
            # Flatten nested weight lists
            self.weights = self._process_weights_list(weights_to_watch)
        else:
            self.weights = weights_to_watch

        self.nb_params = self.backend.get_num_params(self.weights)

    def _get_default_loss_function(self) -> LossFunction:
        """Get the default loss function for the framework."""
        if self.backend.framework == Framework.TENSORFLOW:
            tf = import_optional_module("tensorflow", extra="tensorflow")
            reduction = import_optional_attr("tensorflow.keras.losses", "Reduction", extra="tensorflow")
            return tf.keras.losses.CategoricalCrossentropy(
                from_logits=False, reduction=reduction.NONE
            )

        nn = import_optional_module("torch.nn", extra="pytorch")
        return nn.CrossEntropyLoss(reduction='none')

    def _validate_loss_function(self, loss_function: LossFunction) -> None:
        """Validate that the loss function doesn't have reduction."""
        if self.backend.framework == Framework.TENSORFLOW:
            reduction = import_optional_attr("tensorflow.keras.losses", "Reduction", extra="tensorflow")
            loss_reduction = getattr(loss_function, 'reduction', None)
            if loss_reduction is not None and loss_reduction is not reduction.NONE:
                raise ValueError('The loss function must not have reduction (use Reduction.NONE).')
        # For PyTorch, we could check loss_function.reduction == 'none' but it's less standardized

    def __call__(self, inputs: Tensor) -> Tensor:
        """
        Computes the forward pass of the original model.

        Parameters
        ----------
        inputs
            Inputs on which to make the inference.

        Returns
        -------
        outputs
            Outputs of the original model.
        """
        return self.backend.forward(self.model, inputs)

    def _process_weights_list(
        self,
        weights_to_watch: Union[WeightVariable, List[WeightVariable]],
    ) -> List[WeightVariable]:
        """
        Ensure a proper formatting of the weights (flatten nested lists).

        Parameters
        ----------
        weights_to_watch
            A collection of weights in potentially nested format.

        Returns
        -------
        processed_weights
            A flat list of weights.
        """
        if self.weights_processed:
            if isinstance(weights_to_watch, (list, tuple)):
                return list(weights_to_watch)
            return [weights_to_watch]

        if not isinstance(weights_to_watch, (list, tuple)):
            return [weights_to_watch]

        processed_weights: List[WeightVariable] = []
        for weights in weights_to_watch:
            if isinstance(weights, (list, tuple)):
                processed_weights.extend(weights)
            else:
                processed_weights.append(weights)

        return processed_weights

    @property
    def layers(self) -> List[Layer]:
        """
        Access the layers of the original model.

        Returns
        -------
        layers
            The layers of the original model.
        """
        return self.backend.get_layers(self.model)

    def _compute_loss(self, batch: Tuple[Any, ...]) -> Tensor:
        """
        Computes the model's loss for a single batch of samples.

        Parameters
        ----------
        batch
            A batch of tuples of sets of inputs and their corresponding outputs.

        Returns
        -------
        loss_values
            The loss values for each input (i.e. not reduced).
        """
        model_inp, y_true, sample_weight = self.process_batch_for_loss_fn(batch)
        return self.backend.compute_loss(
            self.model, self.loss_function, model_inp, y_true, sample_weight
        )

    def _compute_jacobian(self, batch: Tuple[Any, ...]) -> Tensor:
        """
        Computes the model's Jacobian for a single batch of samples.

        Parameters
        ----------
        batch
            A batch of tuples of sets of inputs and their corresponding outputs.

        Returns
        -------
        jacobian
            The Jacobian matrix for the set of inputs.
        """
        model_inp, y_true, sample_weight = self.process_batch_for_loss_fn(batch)
        return self.backend.compute_jacobian(
            self.model, self.weights, self.loss_function,
            model_inp, y_true, sample_weight
        )

    def _compute_gradient(self, batch: Tuple[Any, ...]) -> Tensor:
        """
        Computes the model's gradient for a single batch of samples.

        Parameters
        ----------
        batch
            A batch of tuples of sets of inputs and their corresponding outputs.

        Returns
        -------
        gradient
            The gradient vector for the set of inputs.
        """
        model_inp, y_true, sample_weight = self.process_batch_for_loss_fn(batch)
        return self.backend.compute_gradient(
            self.model, self.weights, self.loss_function,
            model_inp, y_true, sample_weight
        )

    def _loss_tensor(self, batch: Tuple[Any, ...]) -> Tensor:
        """
        Computes the model's loss on the batched tensor.

        Parameters
        ----------
        batch
            A batch of tuples of sets of inputs and their corresponding outputs.

        Returns
        -------
        loss_values
            Loss values for each of the points of the batch.
        """
        return self._compute_loss(batch)

    def batch_loss(self, dataset: DatasetLike) -> Tensor:
        """
        Computes the model's loss on the whole batched dataset.

        Parameters
        ----------
        dataset
            Batched dataset (tf.data.Dataset or PyTorch DataLoader).

        Returns
        -------
        loss_values
            Loss values for each of the points in the dataset.
        """
        losses = [self._compute_loss(batch) for batch in dataset]
        return self.backend.concat(losses, axis=0)

    def batch_jacobian_tensor(self, batch: Tuple[Any, ...]) -> Tensor:
        """
        Computes the Jacobian of the loss wrt the weights on a Tensor.

        Parameters
        ----------
        batch
            A batch of tuples of sets of inputs and their corresponding outputs.

        Returns
        -------
        jacobians
            Matrix of the first-order partial derivative of the loss function wrt weights.
        """
        return self._compute_jacobian(batch)

    def batch_jacobian(self, dataset: DatasetLike) -> Tensor:
        """
        Computes the Jacobian of the loss wrt the weights on the whole batched dataset.

        Parameters
        ----------
        dataset
            Batched dataset (tf.data.Dataset or PyTorch DataLoader).

        Returns
        -------
        jacobians
            Matrix of the first-order partial derivative of the loss function wrt weights.
        """
        jacobians = [self._compute_jacobian(batch) for batch in dataset]
        return self.backend.concat(jacobians, axis=0)

    def batch_gradient_tensor(self, batch: Tuple[Any, ...]) -> Tensor:
        """
        Computes the gradient of the loss wrt the weights on a Tensor.

        Parameters
        ----------
        batch
            A batch of tuples of sets of inputs and their corresponding outputs.

        Returns
        -------
        gradients
            Gradient values of the loss function wrt weights.
        """
        return self._compute_gradient(batch)

    def batch_gradient(self, dataset: DatasetLike) -> Tensor:
        """
        Computes the gradient of the loss wrt the weights on the whole batched dataset.

        Parameters
        ----------
        dataset
            Batched dataset (tf.data.Dataset or PyTorch DataLoader).

        Returns
        -------
        gradients
            Gradient values of the loss function wrt weights.
        """
        gradients = [self._compute_gradient(batch) for batch in dataset]
        return self.backend.stack(gradients, axis=0)


class InfluenceModel(BaseInfluenceModel):
    """
    A framework-agnostic model wrapper for Influence functions which allows specifying
    layer ranges from which to watch weights (e.g. ignoring a feature extractor).

    Supports both TensorFlow (tf.keras.Model) and PyTorch (nn.Module) models.

    Parameters
    ----------
    model
        Model used for computing influence score (tf.keras.Model or nn.Module).
    start_layer
        Starting layer name or index for the weights and bias collection. If set to None,
        will search for the last layer with weights before logits.
    last_layer
        Last layer name or index for the weights and biases collection.
        If set to None, only the layer indicated in the start_layer parameter will be used.
    loss_function
        Loss function to calculate influence. Make sure not to apply any reduction.
    process_batch_for_loss_fn
        A callable for preprocessing the batch to transform it into a format that can be treated
        by the algorithm: (inputs, label, sample weight).
    """

    def __init__(
        self,
        model: Model,
        start_layer: Optional[Union[str, int]] = None,
        last_layer: Optional[Union[str, int]] = None,
        loss_function: Optional[LossFunction] = None,
        process_batch_for_loss_fn: ProcessBatchTypeAlias = default_process_batch
    ):
        self.start_layer = start_layer
        self.last_layer = last_layer

        # Get the backend first to determine weights
        backend = get_backend_for_model(model)

        # Get weights for the specified layer range
        weights_to_watch = self._get_weights_of_interest(backend, model, start_layer, last_layer)

        super().__init__(
            model,
            weights_to_watch,
            loss_function,
            process_batch_for_loss_fn,
            weights_processed=True
        )

    @staticmethod
    def _get_weights_of_interest(
        backend: BaseBackend,
        model: Model,
        start_layer: Optional[Union[str, int]],
        last_layer: Optional[Union[str, int]]
    ) -> List[WeightVariable]:
        """
        Gets the list of trainable weights from layer 'start_layer' to layer 'last_layer'.

        Parameters
        ----------
        backend
            The framework-specific backend.
        model
            Model to get weights from.
        start_layer
            Starting layer for the weights collection. If None, auto-detects the last weight layer.
        last_layer
            Last layer for the weights collection. If None, only start_layer is used.

        Returns
        -------
        weights
            A flat list of weights between start_layer and last_layer.
        """
        return backend.get_weights_for_layer_range(model, start_layer, last_layer)


# Backwards compatibility aliases
TensorFlowInfluenceModel = InfluenceModel
PyTorchInfluenceModel = InfluenceModel
