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

import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=E0611
from tensorflow.keras.optimizers import Optimizer, SGD  # pylint: disable=E0611

from ..types import Callable


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

class BacktrackingLineSearch(Optimizer):
    """
    Implementation of a batched Backtracking Line Search optimizer with SGD steps.
    TensorFlow-specific implementation using Keras Optimizer interface.

    Parameters
    ----------
    batches_per_epoch
        An integer indicating the amount of batches that constitute a whole epoch. This information is used for scaling
        the optimizer updates.
    scaling_factor
        A scaling factor for the Wolfe condition
    """
    def __init__(
            self,
            batches_per_epoch: int,
            scaling_factor: float,
            **kwargs
    ):
        super().__init__(name="backtracking_line_search", **kwargs)
        self.optimizer = SGD()
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
            return model.compiled_loss(labels, model(x_inputs, training=True))
        # closure = lambda: model.compiled_loss(labels, model(x_inputs, training=True))
        self.parameters.eta *= self.parameters.gamma
        direction = self.attempt_step(model, curr_weights, gradients, closure)

        # Repeat progressively smaller steps until the (approximate) Wolfe condition is verified
        while not BacktrackingLineSearch.wolfe_condition(direction, current_loss, norm, self.parameters.eta):
            self.parameters.eta *= self.parameters.beta
            if self.parameters.max_eta < self.parameters.eta < self.parameters.min_eta:
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
            gamma: float = None,
            max_eta: float = 10.,
            min_eta: float = 1e-6,
    ):
        from torch.optim import SGD

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
        self._sgd = SGD(self.params, lr=self.parameters.eta)

    def step(self, model, current_loss, x_inputs, labels, gradients, closure: Callable):
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
        import torch

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

