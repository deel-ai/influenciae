# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module containing the base class for representer point theorem-based influence calculators.

Supports both TensorFlow and PyTorch models through the backend abstraction layer.
"""
from abc import abstractmethod

from .._optional_imports import import_optional_attr, import_optional_module
from ..common import BaseInfluenceCalculator, BaseBackend, Framework, get_backend_for_model
from ..types import Tuple, Callable, Union, Any, Optional, DatasetLike


class BaseRepresenterPoint(BaseInfluenceCalculator):
    """
    Base interface for representer point theorem-based influence calculators.

    Supports both TensorFlow and PyTorch models through the backend abstraction layer.

    Disclaimer: This method only works on classification problems!

    Parameters
    ----------
    model
        A model that has already been trained (TensorFlow or PyTorch).
    train_set
        A batched dataset with the points with which the model was trained.
    loss_function
        The loss function with which the model was trained. This loss function MUST NOT be reduced.
    target_layer
        Layer name or index to split the model at (the last layer by default).
    """

    def __init__(
            self,
            model: Any,
            train_set: DatasetLike,
            loss_function: Union[Callable, Any],
            target_layer: Union[str, int] = -1
    ):
        # Get the backend for the model
        self.backend: BaseBackend = get_backend_for_model(model)

        # Make sure that the dataset is batched
        self.backend.assert_batched_dataset(train_set)
        self.train_set = train_set

        # Validate loss function reduction
        self._validate_loss_function(loss_function)

        # Validate that the model's last layer is appropriate for representer point methods
        self._validate_last_layer(model)

        self.loss_function = loss_function
        self.model = model
        self.target_layer = target_layer

        # Cut the model in two (feature extractor and head)
        self.feature_extractor, self.original_head = self.backend.split_model(model, target_layer)

    def _validate_loss_function(self, loss_function: Any) -> None:
        """
        Validate that the loss function doesn't have reduction.

        Parameters
        ----------
        loss_function
            The loss function to validate.
        """
        if self.backend.framework == Framework.TENSORFLOW:
            reduction = import_optional_attr("tensorflow.keras.losses", "Reduction", extra="tensorflow")
            if hasattr(loss_function, 'reduction'):
                assert loss_function.reduction == reduction.NONE, \
                    "The loss function must not have reduction (use Reduction.NONE)."
        else:  # PyTorch
            if hasattr(loss_function, 'reduction'):
                assert loss_function.reduction == 'none', \
                    "The loss function must not have reduction (use reduction='none')."

    def _validate_last_layer(self, model: Any) -> None:
        """
        Validate that the model's last layer is a Dense/Linear layer with no bias.

        Parameters
        ----------
        model
            The model to validate.
        """
        if self.backend.framework == Framework.TENSORFLOW:
            tf = import_optional_module("tensorflow", extra="tensorflow")
            layers = self.backend.get_layers(model)
            if not isinstance(layers[-1], tf.keras.layers.Dense):
                raise ValueError('The last layer of the model must be a Dense layer with no bias.')
            if layers[-1].use_bias:
                raise ValueError('The last layer of the model must be a Dense layer with no bias.')
        else:  # PyTorch
            # Get the last layer - for PyTorch we need to find the last Linear layer
            last_layer = self._get_last_linear_layer(model)
            if last_layer is None:
                raise ValueError('The last layer of the model must be a Linear layer with no bias.')
            if last_layer.bias is not None:
                raise ValueError('The last layer of the model must be a Linear layer with no bias.')

    def _get_last_linear_layer(self, model: Any) -> Optional[Any]:
        """
        Get the last Linear layer from a PyTorch model.

        Parameters
        ----------
        model
            The PyTorch model.

        Returns
        -------
        layer
            The last Linear layer, or None if not found.
        """
        nn = import_optional_module("torch.nn", extra="pytorch")
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        return last_linear

    @abstractmethod
    def _compute_alpha(self, z_batch: Any, y_batch: Any) -> Any:
        """
        Compute the alpha vector for a given input-output pair (z, y).

        Parameters
        ----------
        z_batch
            A tensor containing the latent representation of an input point.
        y_batch
            The labels corresponding to the representations z.

        Returns
        -------
        alpha
            A tensor with the alpha coefficients of the kernel given by the representer point theorem.
        """
        raise NotImplementedError()

    def _preprocess_samples(self, samples: Tuple[Any, ...]) -> Tuple[Any, Any]:
        """
        Preprocess a single batch of samples.

        Parameters
        ----------
        samples
            A single batch of tensors containing the samples.

        Returns
        -------
        x_batch
            The preprocessed feature maps.
        y_t
            The labels.
        """
        # Handle both TensorFlow (where samples[:-1] gives inputs) and PyTorch (where samples[0] gives inputs)
        if len(samples) == 2:
            inputs = samples[0]
        else:
            inputs = samples[:-1]
            if isinstance(inputs, tuple) and len(inputs) == 1:
                inputs = inputs[0]

        x_batch = self.backend.forward(self.feature_extractor, inputs)
        y_t = samples[-1]

        return x_batch, y_t

    def _compute_influence_vector(self, train_samples: Tuple[Any, ...]) -> Tuple[Any, Any]:
        """
        Compute an equivalent of the influence vector for a sample of training points.

        Disclaimer: this vector is not an estimation of the difference between the actual
        model and the perturbed model without the samples (like it is the case with what is
        calculated using deel.influenciae.influence).

        Parameters
        ----------
        train_samples
            A tensor with a group of training samples of which we wish to compute the influence.

        Returns
        -------
        influence_vectors
            A tuple containing the alpha weights and the feature maps for each sample.
            This allows for optimizations to be put in place but is not really an influence vector
            of any kind.
        """
        # Handle both TensorFlow and PyTorch input formats
        if len(train_samples) == 2:
            inputs = train_samples[0]
        else:
            inputs = train_samples[:-1]
            if isinstance(inputs, tuple) and len(inputs) == 1:
                inputs = inputs[0]

        x_batch = self.backend.forward(self.feature_extractor, inputs)
        alpha = self._compute_alpha(x_batch, train_samples[-1])

        return alpha, x_batch

    def _estimate_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[Any, ...],
            samples_to_evaluate: Tuple[Any, ...]
    ) -> Any:
        """
        Estimate the (individual) influence scores of a single batch of samples with respect to
        a batch of samples belonging to the model's training dataset.

        Parameters
        ----------
        train_samples
            A single batch of training samples (and their target values).
        samples_to_evaluate
            A single batch of samples of which we wish to compute the influence of removing the training
            samples.

        Returns
        -------
        influence_values
            A tensor containing the individual influence scores.
        """
        return self._estimate_influence_value_from_influence_vector(
            self._preprocess_samples(samples_to_evaluate),
            self._compute_influence_vector(train_samples)
        )

    def _estimate_influence_value_from_influence_vector(
            self,
            preproc_test_sample: Tuple[Any, Any],
            influence_vector: Tuple[Any, Any]
    ) -> Any:
        """
        Compute the influence score for a (batch of) preprocessed test sample(s) and a training "influence vector".

        Parameters
        ----------
        preproc_test_sample
            A tuple with (feature_maps_test, labels) for the test sample.
        influence_vector
            A tuple with (alpha, feature_maps_train) for the training influence vector.

        Returns
        -------
        influence_values
            A tensor with influence values for the (batch of) test samples.
        """
        # Extract the different information inside the tuples
        feature_maps_test, _ = preproc_test_sample
        alpha, feature_maps_train = influence_vector

        alpha_shape = self.backend.tensor_shape(alpha)
        alpha_ndim = len(alpha_shape)
        kernel_matrix = self.backend.matmul(feature_maps_train, self.backend.transpose(feature_maps_test))
        kernel_dtype = self.backend.get_dtype(kernel_matrix)

        if alpha_ndim == 1 or (alpha_ndim == 2 and alpha_shape[1] == 1):
            # Binary classification case
            influence_values = self.backend.multiply(
                self.backend.cast(alpha, kernel_dtype),
                kernel_matrix
            )
        else:
            # Multiclass case - gather alpha values based on predictions
            head_output = self.backend.forward(self.original_head, feature_maps_test)
            indices = self.backend.argmax(head_output, axis=1)
            gathered_alpha = self.backend.gather_along_axis(alpha, indices, axis=1, batch_dims=1)

            # Reshape gathered_alpha to (n_train, 1) for proper broadcasting with K (n_train, n_test)
            # This ensures each row j of K is multiplied by gathered_alpha[j]
            gathered_alpha = self.backend.reshape(gathered_alpha, (-1, 1))
            influence_values = self.backend.multiply(
                self.backend.cast(gathered_alpha, kernel_dtype),
                kernel_matrix
            )

        influence_values = self.backend.transpose(influence_values)

        return influence_values

    def _compute_influence_value_from_batch(self, train_samples: Tuple[Any, ...]) -> Any:
        """
        Compute the influence score for a batch of training samples (i.e. self-influence).

        Parameters
        ----------
        train_samples
            A tensor containing a batch of training samples.

        Returns
        -------
        influence_values
            A tensor with the self-influence of the training samples.
        """
        # Handle both TensorFlow and PyTorch input formats
        if len(train_samples) == 2:
            inputs = train_samples[0]
        else:
            inputs = train_samples[:-1]
            if isinstance(inputs, tuple) and len(inputs) == 1:
                inputs = inputs[0]

        x_batch = self.backend.forward(self.feature_extractor, inputs)
        alpha = self._compute_alpha(x_batch, train_samples[-1])

        # If the problem is binary classification, take all the alpha values
        # If multiclass, take only those that correspond to the prediction
        out_shape = self.backend.get_output_shape(self.model)
        if len(out_shape) == 1:
            influence_values = alpha
        elif len(out_shape) == 2 and out_shape[1] == 1:
            influence_values = alpha
        else:
            head_output = self.backend.forward(self.original_head, x_batch)
            if len(out_shape) > 2:
                head_output = self.backend.squeeze(head_output, axis=-1)
            indices = self.backend.argmax(head_output, axis=1)
            influence_values = self.backend.gather_along_axis(alpha, indices, axis=1, batch_dims=1)

        return self.backend.abs(influence_values)
