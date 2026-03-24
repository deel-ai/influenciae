# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Backtracking Line Search optimizer algorithm with Stochastic Gradient Descent steps.
This code was based on an implementation by Louis Bethune (ANITI) -- https://github.com/Algue-Rythme

Supports both TensorFlow and PyTorch backends.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from .._optional_imports import import_optional_attr, import_optional_module


@dataclass()
class BTLSParameters:
    """
    Data class containing the Backtracking Line Search optimizer's internal parameters
    """
    beta: float
    gamma: float
    eta: float
    max_eta: float
    min_eta: float


# =============================================================================
# TensorFlow Implementation
# =============================================================================

# Conditionally define TensorFlow classes only when TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras import Model  # pylint: disable=E0611
    from tensorflow.keras.optimizers import Optimizer, SGD  # pylint: disable=E0611
    _HAS_TENSORFLOW = True
except (ImportError, ModuleNotFoundError):
    _HAS_TENSORFLOW = False
    # Create placeholder to avoid NameError
    Optimizer = object
    Model = None
    SGD = None
    tf = None

_BACKTRACKING_LINE_SEARCH_TF: Optional[type] = None

if _HAS_TENSORFLOW:
    class TensorFlowBacktrackingLineSearch(Optimizer):
        """
        Implementation of a batched Backtracking Line Search optimizer with SGD steps.
        TensorFlow-specific implementation using Keras Optimizer interface.

        Parameters
        ----------
        batches_per_epoch
            An integer indicating the amount of batches that constitute a whole epoch.
            This information is used for scaling the optimizer updates.
        scaling_factor
            A scaling factor for the Wolfe condition
        """
        def __init__(
                self,
                batches_per_epoch: int,
                scaling_factor: float,
                **kwargs
        ):
            learning_rate = kwargs.pop("learning_rate", 1.0)
            try:
                super().__init__(learning_rate=learning_rate, name="backtracking_line_search", **kwargs)
            except TypeError as exc:
                if "learning_rate is not a valid argument" not in str(exc):
                    raise
                super().__init__(name="backtracking_line_search", **kwargs)
            self.optimizer = SGD(learning_rate=learning_rate)
            self.scaling_factor = scaling_factor
            self.batches_per_epoch = int(batches_per_epoch)
            self.parameters = BTLSParameters(
                beta=0.9,
                gamma=2 ** (1 / self.batches_per_epoch),
                eta=1.,
                max_eta=10.,
                min_eta=1e-6
            )

        def step(self, model, current_loss, x_inputs, labels, gradients):
            """
            Performs a step of the line-search optimizer by attempting stochastic gradient descents until the Wolfe
            condition is met.

            Parameters
            ----------
            model
                A compiled TF model with accessible weights
            current_loss
                A tensor with the current loss value for the batch
            x_inputs
                The inputs for the batch
            labels
                The corresponding labels for the batch
            gradients
                The gradients of the loss wrt the model's weights for the provided loss tensor
            """
            # Save the original weights for the batch
            curr_weights = model.get_weights()

            # Attempt a first direction
            norm = self.c_gradnorm(gradients)

            def closure():
                predictions = model(x_inputs, training=True)
                regularization_losses = list(getattr(model, "losses", []))

                # Compatibility with Keras 3, which changed the signature of compute_loss and compiled_loss
                compute_loss = getattr(model, "compute_loss", None)
                if compute_loss is not None:
                    try:
                        return compute_loss(
                            x=x_inputs,
                            y=labels,
                            y_pred=predictions,
                            sample_weight=None,
                            training=True,
                        )
                    except TypeError:
                        try:
                            return compute_loss(
                                x=x_inputs,
                                y=labels,
                                y_pred=predictions,
                                sample_weight=None,
                            )
                        except TypeError:
                            pass

                compiled_loss = getattr(model, "compiled_loss", None)
                if compiled_loss is None:
                    raise AttributeError(
                        "The model must expose either `compute_loss` or `compiled_loss` "
                        "to be optimized with BacktrackingLineSearch."
                    )

                try:
                    return compiled_loss(
                        labels,
                        predictions,
                        sample_weight=None,
                        regularization_losses=regularization_losses,
                    )
                except TypeError:
                    try:
                        return compiled_loss(
                            labels,
                            predictions,
                            regularization_losses=regularization_losses,
                        )
                    except TypeError:
                        try:
                            base_loss = compiled_loss(labels, predictions)
                        except TypeError:
                            base_loss = compiled_loss(y_true=labels, y_pred=predictions)

                # Compatibility with Keras 3
                # In case the loss function does not support regularization losses, we add them manually if they exist
                if regularization_losses:
                    reg_loss = tf.add_n([tf.cast(loss_term, base_loss.dtype) for loss_term in regularization_losses])
                    return base_loss + reg_loss
                return base_loss

            self.parameters.eta *= self.parameters.gamma
            direction = self.attempt_step(model, curr_weights, gradients, closure)

            # Repeat progressively smaller steps until the (approximate) Wolfe condition is verified
            max_backtracking_steps = 1000
            n_steps = 0
            while True:
                wolfe_ok = self.wolfe_condition(direction, current_loss, norm, self.parameters.eta)
                if hasattr(wolfe_ok, "numpy"):
                    wolfe_ok = bool(wolfe_ok.numpy())
                else:
                    wolfe_ok = bool(wolfe_ok)

                if wolfe_ok:
                    break

                self.parameters.eta *= self.parameters.beta
                n_steps += 1
                if n_steps >= max_backtracking_steps:
                    break
                if self.parameters.eta < self.parameters.min_eta or self.parameters.eta > self.parameters.max_eta:
                    break
                direction = self.attempt_step(model, curr_weights, gradients, closure)

            # Save the most suitable learning rate in the permitted range
            self.parameters.eta = min(max(self.parameters.min_eta, self.parameters.eta), self.parameters.max_eta)

        @staticmethod
        def wolfe_condition(target, source, norm, eta):
            """
            Verifies whether the Wolfe condition is being met for a set of loss values (before vs after an SGD step).

            Parameters
            ----------
            target
                A tensor with the value of the loss at the current inner step of the line-search
            source
                A tensor with the value of the loss at the beginning of the optimizer's step
            norm
                A tensor with the norm of the gradients for the current loss
            eta
                The current step size in the inner step of the line-search

            Returns
            -------
            wolfe_condition
                A boolean indicating whether the Wolfe condition is verified for the current learning rate
            """
            return tf.less_equal(target, source - eta * norm)

        def attempt_step(self, model: Model, curr_weights: tf.Tensor, gradients: tf.Tensor, closure: Callable):
            """
            Performs a step of SGD using the updated learning rate in the direction of the gradient and returns the new
            value of the loss function with the new weights.

            Parameters
            ----------
            model
                A TF model with accessible weights and variables
            curr_weights
                A tensor with the current model's weights
            gradients
                A tensor with the loss' gradients wrt the model's weights on the batch
            closure
                An object whose call returns the loss function's value using the updated weights

            Returns
            -------
            called_closure
                The new value of the loss for the model with the updated weights
            """
            self.optimizer.learning_rate.assign(self.parameters.eta)
            model.set_weights(curr_weights)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return closure()

        def c_gradnorm(self, gradients):
            """
            Compute the squared norm of the loss function's gradients

            Parameters
            ----------
            gradients
                A tensor with the loss function's gradients

            Returns
            -------
            norm
                A tensor with the norm of the loss function's gradients scaled by the parameter scaling_factor
            """
            return self.scaling_factor * tf.linalg.global_norm(gradients) ** 2

        def _resource_apply_dense(self, grad, handle, apply_state):
            pass

        def _resource_apply_sparse(self, grad, handle, indices, apply_state):
            pass

        def get_config(self):
            """
            Implementation of configuration setter method, as required for the Optimizer interface.
            """
            base_config = super().get_config()
            base_config["batches_per_epoch"] = self.batches_per_epoch
            base_config["scaling_factor"] = self.scaling_factor
            return base_config

    _BACKTRACKING_LINE_SEARCH_TF = TensorFlowBacktrackingLineSearch


BacktrackingLineSearch = _BACKTRACKING_LINE_SEARCH_TF


# =============================================================================
# PyTorch Implementation
# =============================================================================

class BacktrackingLineSearchPyTorch:
    """
    Implementation of a batched Backtracking Line Search optimizer with SGD steps.
    PyTorch-specific implementation using torch.optim.Optimizer interface.

    Parameters
    ----------
    params
        An iterable of parameters to optimize or dicts defining parameter groups.
    batches_per_epoch
        An integer indicating the amount of batches that constitute a whole epoch. This information is used for scaling
        the optimizer updates.
    scaling_factor
        A scaling factor for the Wolfe condition (default: 0.1).
    beta
        The backtracking factor for reducing step size (default: 0.9).
    gamma
        The growth factor for the learning rate per batch (default: computed from batches_per_epoch).
    max_eta
        Maximum allowed learning rate (default: 10.0).
    min_eta
        Minimum allowed learning rate (default: 1e-6).
    """
    def __init__(
            self,
            params,
            batches_per_epoch: int,
            scaling_factor: float = 0.1,
            beta: float = 0.9,
            gamma: Optional[float] = None,
            max_eta: float = 10.,
            min_eta: float = 1e-6,
    ):
        self._torch = import_optional_module("torch", extra="pytorch")
        torch_sgd = import_optional_attr("torch.optim", "SGD", extra="pytorch")

        self.params = list(params)
        self.scaling_factor = scaling_factor
        self.batches_per_epoch = int(batches_per_epoch)

        # Compute gamma if not provided
        if gamma is None:
            gamma = 2 ** (1 / self.batches_per_epoch)

        self.parameters = BTLSParameters(
            beta=beta,
            gamma=gamma,
            eta=1.,
            max_eta=max_eta,
            min_eta=min_eta
        )

        # Internal SGD optimizer for applying gradients
        self._sgd = torch_sgd(self.params, lr=self.parameters.eta)

    def step(  # pylint: disable=unused-argument
        self,
        model,
        current_loss,
        x_inputs,
        labels,
        gradients,
        closure: Callable,
    ):
        """
        Performs a step of the line-search optimizer by attempting stochastic gradient descents until the Wolfe
        condition is met.

        Parameters
        ----------
        model
            A PyTorch model (nn.Module) with accessible parameters
        current_loss
            A tensor with the current loss value for the batch
        x_inputs
            The inputs for the batch
        labels
            The corresponding labels for the batch
        gradients
            The gradients of the loss wrt the model's weights for the provided loss tensor (list of tensors)
        closure
            A callable that recomputes the loss with current weights
        """
        # Save the original weights for the batch
        curr_weights = [p.data.clone() for p in self.params]

        # Compute gradient norm
        norm = self.c_gradnorm(gradients)

        # Grow the step size
        self.parameters.eta *= self.parameters.gamma
        direction = self._attempt_step(curr_weights, gradients, closure)

        # Repeat progressively smaller steps until the (approximate) Wolfe condition is verified
        while not self._wolfe_condition(direction, current_loss, norm, self.parameters.eta):
            self.parameters.eta *= self.parameters.beta
            if self.parameters.eta < self.parameters.min_eta:
                break
            direction = self._attempt_step(curr_weights, gradients, closure)

        # Save the most suitable learning rate in the permitted range
        self.parameters.eta = min(max(self.parameters.min_eta, self.parameters.eta), self.parameters.max_eta)

    @staticmethod
    def _wolfe_condition(target, source, norm, eta):
        """
        Verifies whether the Wolfe condition is being met for a set of loss values (before vs after an SGD step).

        Parameters
        ----------
        target
            A tensor with the value of the loss at the current inner step of the line-search
        source
            A tensor with the value of the loss at the beginning of the optimizer's step
        norm
            A tensor with the norm of the gradients for the current loss
        eta
            The current step size in the inner step of the line-search

        Returns
        -------
        wolfe_condition
            A boolean indicating whether the Wolfe condition is verified for the current learning rate
        """
        return target <= source - eta * norm

    def _attempt_step(self, curr_weights, gradients, closure: Callable):
        """
        Performs a step of SGD using the updated learning rate in the direction of the gradient and returns the new
        value of the loss function with the new weights.

        Parameters
        ----------
        curr_weights
            A list of tensors with the current model's weights
        gradients
            A list of tensors with the loss' gradients wrt the model's weights on the batch
        closure
            A callable that returns the loss function's value using the updated weights

        Returns
        -------
        called_closure
            The new value of the loss for the model with the updated weights
        """
        torch = self._torch

        # Restore weights to original state
        with torch.no_grad():
            for param, weight in zip(self.params, curr_weights):
                param.data.copy_(weight)

        # Apply SGD step with current learning rate
        with torch.no_grad():
            for param, grad in zip(self.params, gradients):
                if grad is not None:
                    param.data.add_(grad, alpha=-self.parameters.eta)

        # Evaluate the new loss
        with torch.no_grad():
            return closure()

    def c_gradnorm(self, gradients):
        """
        Compute the squared norm of the loss function's gradients

        Parameters
        ----------
        gradients
            A list of tensors with the loss function's gradients

        Returns
        -------
        norm
            A tensor with the norm of the loss function's gradients scaled by the parameter scaling_factor
        """
        # Compute global norm (similar to tf.linalg.global_norm)
        total_norm_sq = sum(g.pow(2).sum() for g in gradients if g is not None)
        return self.parameters.eta * self.scaling_factor * total_norm_sq

    def zero_grad(self):
        """Clear gradients for all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def state_dict(self):
        """Return optimizer state."""
        return {
            'eta': self.parameters.eta,
            'beta': self.parameters.beta,
            'gamma': self.parameters.gamma,
            'max_eta': self.parameters.max_eta,
            'min_eta': self.parameters.min_eta,
            'scaling_factor': self.scaling_factor,
            'batches_per_epoch': self.batches_per_epoch,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.parameters.eta = state_dict['eta']
        self.parameters.beta = state_dict['beta']
        self.parameters.gamma = state_dict['gamma']
        self.parameters.max_eta = state_dict['max_eta']
        self.parameters.min_eta = state_dict['min_eta']
        self.scaling_factor = state_dict['scaling_factor']
        self.batches_per_epoch = state_dict['batches_per_epoch']
