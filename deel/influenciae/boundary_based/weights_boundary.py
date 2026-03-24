# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a data-point-influence estimation method based on how intensely
the model's weights must be adversarially attacked to make it change its prediction
for each sample. Intuitively, outliers and atypical examples will need a lighter
deformation of the boundary for it to place them in the wrong class.

Unlike other influence calculators, this one cannot be used to estimate
the influence of a point on another.

This boundary deformation process is performed using deep fool on the target weights.

Supports both TensorFlow and PyTorch models through the backend abstraction layer.
"""
from typing import Any, List, Tuple

from ..common import SelfInfluenceCalculator, BaseBackend, get_backend_for_model
from ._base_boundary import _BaseBoundaryCalculatorMixin


class WeightsBoundaryCalculator(_BaseBoundaryCalculatorMixin, SelfInfluenceCalculator):
    """
    A class implementing an influence score based on the distance of a sample to the boundary of its classifier.
    The distance to the boundary is estimated by deforming the boundary of the model to move a given sample
    to the closest adversarial class.
    To compute this distance, the deep fool method is used on the weights of the model (deep fool originally compute
    the distance on the sample space).
    [https://arxiv.org/abs/1511.04599]

    Supports both TensorFlow and PyTorch models through the backend abstraction layer.

    Notes
    -----
    This method has better mislabeled-sample-detection performance when the model overfits.

    Parameters
    ----------
    model
        A TensorFlow or PyTorch model that has already been trained
    step_nbr
        Number of the iterations to find the closest adversarial problem
    norm_type
        The distance norm used to compute the distance to the boundary
    eps
        Difference between two logits to assume that the logits have the same values
    """

    def __init__(self, model: Any, step_nbr: int = 100, norm_type: int = 2, eps: float = 1E-6):
        self.backend: BaseBackend = get_backend_for_model(model)
        self.weights_init = [self.backend.clone_variable(w)
                            for w in self.backend.get_model_weights(model)]
        self.model = model

        self.step_nbr = step_nbr
        self.norm_type = norm_type
        self.eps = eps

    def _compute_norm(self, weights: List[Any]) -> Any:
        """
        Compute the norm of a list of weights

        Parameters
        ----------
        weights
            The list of weights

        Returns
        -------
        weights_norm
            The norm of the weights
        """
        weights_flatten = self.backend.concat(
            [self.backend.reshape(w, (-1,)) for w in weights],
            axis=0
        )
        weights_norm = self.backend.norm(weights_flatten, ord=self.norm_type)

        return weights_norm

    def _step(self, x: Any, y_pred: Any) -> Tuple[Any, Any]:
        """
        The optimization step to find the distance between the boundary and a given sample x.
        To see more details about the optimization procedure for multi-class classifiers,
        please refer to [https://arxiv.org/abs/1511.04599]

        Notes
        -----
        This function updates the weights of the model at each step.

        Parameters
        ----------
        x
            The current sample used to compute the distance to the boundary of the model
        y_pred
            The one-hot labels predicted by the current model for the x sample

        Returns
        -------
        computation
            Boolean to determine if the optimization process should continue.
            True if the sample didn't change of class, False if not.
        loss_value
            The loss of the optimization procedure
        """
        y_pred_class = self.backend.argmax(y_pred, axis=1)

        model_weights = self.backend.get_model_weights(self.model)

        # Compute output and jacobian with respect to weights
        y, jac = self.backend.compute_output_jacobian_wrt_weights(self.model, model_weights, x)

        y_shape = self.backend.tensor_shape(y)
        computation = self._compute_step_condition(y, y_pred_class, self.eps)

        # Default loss value
        x_dtype = self.backend.get_dtype(x)
        default_loss = self.backend.constant(0.0, dtype=x_dtype)

        if computation:
            loss_value = self._compute_weight_update(y, jac, y_pred_class, y_shape, model_weights)
        else:
            loss_value = default_loss

        return computation, loss_value

    def _compute_weight_update(
        self,
        y: Any,
        jac: List[Any],
        y_pred_class: Any,
        y_shape: Tuple,
        model_weights: List[Any]
    ) -> Any:
        """
        Compute the DeepFool update step for weights.

        Parameters
        ----------
        y
            Model output logits
        jac
            List of Jacobians of model output with respect to each weight tensor
        y_pred_class
            Original predicted class
        y_shape
            Shape of the output tensor
        model_weights
            Current model weights

        Returns
        -------
        loss
            The loss value for this step
        """
        indexes_other = self._build_other_class_indices(y_pred_class, y_shape)

        # Compute delta in logits
        delta_y = self._compute_delta_y(indexes_other, y_pred_class, y)

        # Compute delta in jacobian for each weight tensor
        jac_delta = [
            self.backend.reduce_mean(self._delta_to_index(indexes_other, y_pred_class, j), axis=0)
            for j in jac
        ]

        # Compute norm of jacobian difference
        jac_norm = self.backend.concat(
            [self.backend.reshape(j, (self.backend.tensor_shape(j)[0], -1)) for j in jac_delta],
            axis=1
        )
        jac_norm = self.backend.norm(jac_norm, ord=self.norm_type, axis=1)

        # Compute coefficient for each class
        coeff = delta_y / jac_norm

        # Find best class to attack
        best_class = self.backend.argmin(coeff, axis=0)

        # Compute loss using gather (graph-compatible, no to_numpy)
        loss = self.backend.gather_along_axis(
            coeff / self.backend.pow(jac_norm, self.norm_type - 1),
            self.backend.expand_dims(best_class, axis=0),
            axis=0
        )
        loss = self.backend.squeeze(loss)

        # Update weights using gather for best class
        for w, g in zip(model_weights, jac_delta):
            g_best = self.backend.gather_along_axis(
                g,
                self.backend.expand_dims(best_class, axis=0),
                axis=0
            )
            g_best = self.backend.squeeze(g_best, axis=0)
            new_value = (
                w + loss
                * self.backend.pow(self.backend.abs(g_best), self.norm_type - 1)
                * self.backend.sign(g_best)
            )
            self.backend.assign_variable(w, new_value)

        return loss

    def _delta_weights(self) -> Any:
        """
        Compute the norm between the trained weights of the model and the current modified weights of the model.

        Returns
        -------
        norm
            The distance between the initial model and the current model
        """
        model_weights = self.backend.get_model_weights(self.model)
        weights = [w1 - w2 for w1, w2 in zip(self.weights_init, model_weights)]
        norm = self._compute_norm(weights)

        return norm

    def _reset_weights(self) -> None:
        """
        Set the weights of the model to the initial trained weights.
        """
        model_weights = self.backend.get_model_weights(self.model)
        for w, w_init in zip(model_weights, self.weights_init):
            self.backend.assign_variable(w, w_init)

    def _compute_single_sample_score(self, x: Any) -> Any:
        """
        Computes the influence score (self-influence) for a single training samples.

        Parameters
        ----------
        x
            A tensor with a single training sample.

        Returns
        -------
        score
            The influence score of the sample.
        """
        x = self.backend.expand_dims(x, axis=0)
        y_pred = self.backend.forward(self.model, x)

        # Use while_loop for graph-compatible iteration
        def cond_fn(cond, idx):
            return self.backend.logical_and(cond, idx < self.step_nbr)

        def body_fn(_cond, idx):
            new_cond, _ = self._step(x, y_pred)
            return [new_cond, idx + 1]

        # Initial loop variables
        init_cond = self.backend.constant(True)
        init_idx = self.backend.constant(0, dtype=self.backend.int32_dtype())

        # Run the loop
        self.backend.while_loop(
            cond_fn,
            body_fn,
            [init_cond, init_idx],
            maximum_iterations=self.step_nbr
        )

        score = self._delta_weights()

        self._reset_weights()

        return score

    def _compute_influence_value_from_batch(self, train_samples: Tuple[Any, ...]) -> Any:
        """
        Computes the influence score (self-influence) for a single batch of training samples.

        Parameters
        ----------
        train_samples
            A tensor with a single batch of training sample.

        Returns
        -------
        influence_values
            The influence score of each sample in the batch train_samples.
        """
        # Get the input samples (first element of tuple, excluding last which is typically labels)
        inputs = train_samples[:-1][0] if len(train_samples) > 1 else train_samples[0]

        scores = self.backend.map_fn(self._compute_single_sample_score, inputs)
        scores = - self.backend.expand_dims(scores, axis=1)

        return scores
