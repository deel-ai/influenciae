# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a data-point-influence estimation method based on how intensely
each sample must be adversarially attacked to make it change its prediction. Intuitively,
we would expect outliers and atypical examples to need a lesser attack for the model to
misclassify it.

Unlike other influence calculators, this one cannot be used to estimate
the influence of a point on another.

The adversarial attacks are performed via deep fool.

Supports both TensorFlow and PyTorch models through the backend abstraction layer.
"""
from ..common import SelfInfluenceCalculator, BaseBackend, get_backend_for_model
from ..types import Tuple, Any


class SampleBoundaryCalculator(SelfInfluenceCalculator):
    """
    A class implementing an influence score based on the distance of a sample to the
    boundary of the classifier.
    The distance to the boundary is estimated using the deep fool method.
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
    eps
        Difference between two logits to assume that they have the same values
    """

    def __init__(self, model: Any, step_nbr: int = 100, eps: float = 1E-6):
        self.backend: BaseBackend = get_backend_for_model(model)
        self.weights_init = [self.backend.clone_variable(w)
                            for w in self.backend.get_model_weights(model)]
        self.model = model

        self.step_nbr = step_nbr
        self.eps = eps

    def _delta_to_index(self, indexes_1: Any, indexes_2: Any, x: Any) -> Any:
        """
        Compute the difference between the logit of a given class and the other logits

        Parameters
        ----------
        indexes_1
            The indices of other classes
        indexes_2
            The indices of the predicted class
        x
            The logits

        Returns
        -------
        delta_x
            The difference between the logits
        """
        x1 = self.backend.gather_along_axis(x, indexes_1, axis=1, batch_dims=1)
        x2 = self.backend.gather_along_axis(
            x,
            self.backend.expand_dims(indexes_2, axis=1),
            axis=1,
            batch_dims=1
        )

        x1_shape = self.backend.tensor_shape(x1)
        delta_x = x1 - self.backend.repeat(x2, x1_shape[1], axis=1)

        return delta_x

    def _step(self, x: Any, y_pred: Any) -> Tuple[Any, Any, Any]:
        """
        The optimization step to find the distance between the boundary and a given sample x.

        Notes
        -----
        To see more details about the optimization procedure for multi-class classifier, please
        refer to [https://arxiv.org/abs/1511.04599]

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
        x_new
            The sample updated by the optimization procedure
        """
        y_pred_class = self.backend.argmax(y_pred, axis=1)

        # Compute output and jacobian with respect to input
        y, jac = self.backend.compute_output_jacobian(self.model, x)

        y_computed = self.backend.argmax(y, axis=1)
        y_shape = self.backend.tensor_shape(y)

        # Check if prediction changed
        computation = self.backend.reduce_any(y_computed == y_pred_class)

        # Check if top 2 logits are close enough
        top_k_values, _ = self.backend.top_k(self.backend.squeeze(y, axis=0), k=2)
        enough_close = self.backend.abs(top_k_values[0] - top_k_values[1]) > self.eps

        computation = self.backend.logical_and(computation, enough_close)

        # Default values if we don't need to update
        x_dtype = self.backend.get_dtype(x)
        default_loss = self.backend.constant(0.0, dtype=x_dtype)
        default_x = x

        # Only compute update if we should continue
        if computation:
            loss_value, x_updated = self._compute_update(x, y, jac, y_pred_class, y_shape)
        else:
            loss_value, x_updated = default_loss, default_x

        return computation, loss_value, x_updated

    def _compute_update(self, x: Any, y: Any, jac: Any, y_pred_class: Any, y_shape: Tuple) -> Tuple[Any, Any]:
        """
        Compute the DeepFool update step.

        Parameters
        ----------
        x
            Current input sample
        y
            Model output logits
        jac
            Jacobian of model output with respect to input
        y_pred_class
            Original predicted class
        y_shape
            Shape of the output tensor

        Returns
        -------
        loss
            The loss value for this step
        x_new
            The updated sample
        """
        batch_size = y_shape[0]
        num_classes = y_shape[1]

        # Create indices for all classes and other classes
        indexes_all = self.backend.tile(
            self.backend.expand_dims(self.backend.arange(0, num_classes), axis=0),
            (batch_size, 1)
        )
        indexes_class = self.backend.cast(
            self.backend.tile(
                self.backend.expand_dims(y_pred_class, axis=1),
                (1, num_classes)
            ),
            dtype=self.backend.int32_dtype()
        )

        # Get indices of other classes (not the predicted class)
        mask = indexes_all != indexes_class
        indexes_other = self.backend.reshape(
            self.backend.boolean_mask(indexes_all, mask),
            (-1, num_classes - 1)
        )

        # Compute delta in logits
        delta_y = self._delta_to_index(indexes_other, y_pred_class, y)
        delta_y = self.backend.abs(self.backend.reduce_mean(delta_y, axis=0))

        # Compute delta in jacobian
        jac_delta = self.backend.reduce_mean(
            self._delta_to_index(indexes_other, y_pred_class, jac),
            axis=0
        )

        # Compute norm of jacobian difference
        jac_delta_shape = self.backend.tensor_shape(jac_delta)
        jac_norm = self.backend.reshape(jac_delta, (jac_delta_shape[0], -1))
        jac_norm = self.backend.norm(jac_norm, axis=1)

        # Compute coefficient for each class
        coeff = delta_y / jac_norm

        # Find best class to attack
        best_class = self.backend.argmin(coeff, axis=0)

        # Compute loss and update using tensor operations (no to_numpy - graph-compatible)
        # Use gather to get the value at best_class index
        loss = self.backend.gather_along_axis(
            coeff / jac_norm,
            self.backend.expand_dims(best_class, axis=0),
            axis=0
        )
        loss = self.backend.squeeze(loss)

        # Gather the best jacobian delta
        jac_delta_best = self.backend.gather_along_axis(
            jac_delta,
            self.backend.expand_dims(best_class, axis=0),
            axis=0
        )
        jac_delta_best = self.backend.squeeze(jac_delta_best, axis=0)

        x_new = x + loss * jac_delta_best

        return loss, x_new

    def _compute_single_sample_score(self, x: Any) -> Any:
        """
        Computes the influence score (self-influence) for a single training sample.

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
        def cond_fn(cond, idx, x_current):
            return self.backend.logical_and(cond, idx < self.step_nbr)

        def body_fn(cond, idx, x_current):
            new_cond, _, x_new = self._step(x_current, y_pred)
            return [new_cond, idx + 1, x_new]

        # Initial loop variables
        init_cond = self.backend.constant(True)
        init_idx = self.backend.constant(0, dtype=self.backend.int32_dtype())

        # Run the loop
        _, _, x_final = self.backend.while_loop(
            cond_fn,
            body_fn,
            [init_cond, init_idx, x],
            maximum_iterations=self.step_nbr
        )

        score = self.backend.norm(x - x_final)

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
