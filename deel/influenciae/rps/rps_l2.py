# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing the representer point theorem for kernels for estimating the
influence of training data-points, as per:
https://arxiv.org/abs/1811.09720

Supports both TensorFlow and PyTorch models through the backend abstraction layer.
"""
from typing import Any, Callable, Optional, Tuple, Union

from .base_representer_point import BaseRepresenterPoint
from ..types import DatasetLike
from ..utils.model_surgery import (
    compute_l2_alpha,
    create_surrogate_linear_model,
    train_surrogate_linear_model,
)


class RepresenterPointL2(BaseRepresenterPoint):
    """
    A class implementing a method to compute the influence of training points through
    the representer point theorem for kernels.

    It builds a kernel that approximates the model's last layer such that for a training
    point x_i and a test point x_t with label y_t:

    y_t = sum_i k(alpha_i, x_i, x_t)

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
    lambda_regularization
        The coefficient for the regularization of the surrogate last layer that needs
        to be trained for this method.
    scaling_factor
        A float with the scaling factor for the SGD backtracking line-search optimizer
        for fitting the surrogate linear model.
    epochs
        An integer for the amount of epochs to fit the linear model.
    layer_index
        Layer index of the logits (-1 for last layer by default).
    """

    def __init__(
            self,
            model: Any,
            train_set: DatasetLike,
            loss_function: Union[Callable, Any],
            lambda_regularization: float,
            scaling_factor: float = 0.1,
            epochs: int = 100,
            layer_index: int = -1,
    ):
        super().__init__(model, train_set, loss_function, layer_index)
        self.n_train = self.backend.get_dataset_size(train_set)
        self.train_set = train_set
        self.lambda_regularization = lambda_regularization
        self.scaling_factor = scaling_factor
        self.epochs = epochs
        self.linear_layer: Optional[Any] = None
        self._train_last_layer(self.epochs)

    def _train_last_layer(self, epochs: int) -> None:
        """
        Train an L2-regularized surrogate linear model to predict like the original head.

        Parameters
        ----------
        epochs
            An integer with the amount of epochs to train the surrogate model.
        """
        self.linear_layer = self._create_surrogate_model()
        self.linear_layer = train_surrogate_linear_model(
            backend=self.backend,
            surrogate_model=self.linear_layer,
            feature_extractor=self.feature_extractor,
            original_head=self.original_head,
            train_set=self.train_set,
            loss_function=self.loss_function,
            scaling_factor=self.scaling_factor,
            epochs=epochs,
        )

    def _create_surrogate_model(self) -> Any:
        """
        Create an L2-regularized linear surrogate with the right input and output sizes.

        Returns
        -------
        surrogate_model
            An L2-regularized linear model.
        """
        return create_surrogate_linear_model(
            self.backend,
            self.feature_extractor,
            self.original_head,
            self.lambda_regularization,
        )

    def _compute_alpha(self, z_batch: Any, y_batch: Any) -> Any:
        """
        Compute the alpha factor for the kernel approximation.

        Parameters
        ----------
        z_batch
            A training sample wrt to which we wish to compute the gradients.
        y_batch
            Label associated to the training sample.

        Returns
        -------
        alpha
            The alpha coefficients representing the influence score.
        """
        assert self.linear_layer is not None
        return compute_l2_alpha(
            backend=self.backend,
            linear_layer=self.linear_layer,
            loss_function=self.loss_function,
            z_batch=z_batch,
            y_batch=y_batch,
            n_train=self.n_train,
            lambda_regularization=self.lambda_regularization,
        )

    def predict_with_kernel(self, samples_to_evaluate: Tuple[Any, ...]) -> Any:
        """
        Use the learned kernel to approximate the model's predictions on a batch of samples.

        Parameters
        ----------
        samples_to_evaluate
            A single batch of tensors with the samples for which we wish to approximate the model's
            predictions.

        Returns
        -------
        predictions
            A tensor with an approximation of the model's predictions.
        """
        influence_vectors = self.compute_influence_vector(self.train_set)
        _, dataset_influence = self._estimate_inf_values_with_inf_vect_dataset(influence_vectors, samples_to_evaluate)

        predictions = None
        for _, influence_values in dataset_influence:
            batch_pred = self.backend.reduce_sum(influence_values, axis=1)
            if predictions is None:
                predictions = batch_pred
            else:
                predictions = predictions + batch_pred

        return predictions
