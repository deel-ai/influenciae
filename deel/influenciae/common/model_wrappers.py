"""
Custom wrappers for tensorflow model
"""

import tensorflow as tf
from tensorflow.keras.losses import Reduction # pylint: disable=E0611

from .tf_operations import find_layer, assert_batched_dataset
from ..types import Callable, Optional, Union


class InfluenceModel:
    """
    Tensorflow model wrapper for Influence functions.

    Parameters
    ----------
    model
        Model used for computing influence score.
    target_layer
        Layer to target for influence calculation (e.g. before logits). Can be an int (layer
        index) or a string (layer_name). It is recommended to use the layer before logits.
        Defaults to the last layer with weights.
    loss_function
        Loss function to calculate influence (e.g. keras CategoricalCrossentropy). Make sure not to
        apply any reduction (Reduction.NONE), and specify correctly if the output is `from_logits`
        for example.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 target_layer: Optional[Union[str, int]] = None,
                 loss_function: Callable = tf.keras.losses.CategoricalCrossentropy(
                     from_logits=False, reduction=Reduction.NONE)):

        if hasattr(loss_function, 'reduction') and loss_function.reduction is not Reduction.NONE:
            raise ValueError('The loss function must not have reduction.')

        if target_layer is None:
            target_layer = self._find_last_weight_layer(model)
        self.target_layer = target_layer

        self.model = model
        self.weights_layer = find_layer(model, target_layer)
        self.weights = self.weights_layer.weights[0]
        self.loss_function = loss_function

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward of the original model

        Parameters
        ----------
        x
            Inputs on which to make the inference.

        Returns
        -------
        y
            Outputs of the original model.
        """
        return self.model(x)

    @property
    def layers(self):
        """
        Access the layers of the original model.

        Returns
        -------
        layers
            The layers of the original model
        """
        return self.model.layers

    def batch_loss(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Computes the model's loss on the whole batched dataset.

        Parameters
        ----------
        dataset
            Batched dataset on which to compute the loss.

        Returns
        -------
        loss_values
            Loss values for each of the points in the dataset.
        """
        assert_batched_dataset(dataset)

        loss_values = tf.concat([
            InfluenceModel._loss(self.model, self.loss_function, batch_x, batch_y)
            for batch_x, batch_y in dataset
        ], axis=0)

        return loss_values

    def batch_jacobian(self, dataset) -> tf.Tensor:
        """
        Computes the jacobian of the loss wrt the weights of the target_layer on the whole
        batched dataset.

        Parameters
        ----------
        dataset
            Batched dataset on which to compute the jacobian.

        Returns
        -------
        jacobians
            Matrix of the first-order partial derivative of the loss function wrt the
            target_layer weights.
        """
        assert_batched_dataset(dataset)

        jacobians = tf.concat([
            InfluenceModel._jacobian(self.model, self.weights, self.loss_function,
                                     batch_x, batch_y)
            for batch_x, batch_y in dataset
        ], axis=0)

        return jacobians

    def batch_gradient(self, dataset) -> tf.Tensor:
        """
        Computes the gradient of the loss wrt the weights of the target_layer on the whole
        batched dataset.

        Parameters
        ----------
        dataset
            Batched dataset on which to compute the gradient.

        Returns
        -------
        gradients
            Gradient values of the loss function wrt the target_layer's weights.
        """
        assert_batched_dataset(dataset)

        gradients = tf.concat([
            InfluenceModel._gradient(self.model, self.weights, self.loss_function,
                                     batch_x, batch_y)
            for batch_x, batch_y in dataset
        ], axis=0)

        return gradients

    @staticmethod
    def _find_last_weight_layer(model: tf.keras.Model) -> int:
        """
        Find and return the id of the last layer before the logits with weights.

        Parameters
        ----------
        model
            Model used for computing influence score.

        Returns
        -------
        layer_id
            Id (e.g -2, -3...)of the layer found.
        """
        for layer_id in range(2, len(model.layers)):
            layer = model.layers[-layer_id]
            if hasattr(layer, 'weights') and layer.weights:
                return -layer_id
        raise ValueError('No layers with weights found for the model.')

    @staticmethod
    @tf.function
    def _loss(model: tf.keras.Model, loss_function: Callable,
              x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Computes the model's loss for a batch of samples.

        Parameters
        ----------
        model
            Model used for computing influence score.
        loss_function
            Reduction-less loss function to calculate influence (e.g. cross-entropy).
        x
            Batch of inputs on which to compute the loss.
        y
            Batch of target used to compute the loss.

        Returns
        -------
        loss_values
            Loss values for each inputs.
        """
        return loss_function(y, model(x))

    @staticmethod
    @tf.function
    def _jacobian(model: tf.keras.Model, weights: tf.Tensor, loss_function: Callable,
                  x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Computes the model's jacobian for a batch of samples.

        Parameters
        ----------
        model
            Model used for computing influence score.
        loss_function
            Reduction-less loss function to calculate influence (e.g. cross-entropy).
        x
            Batch of inputs on which to compute the jacobian.
        y
            Batch of target used to compute the jacobian.

        Returns
        -------
        jacobian
            Jacobian matrix for the set of inputs.
        """
        with tf.GradientTape() as tape:
            tape.watch(weights)
            y_pred = loss_function(y, model(x))

        jacobian = tape.jacobian(y_pred, weights)
        jacobian = tf.reshape(jacobian, (len(jacobian), -1))

        return jacobian

    @staticmethod
    @tf.function
    def _gradient(model: tf.keras.Model, weights: tf.Variable, loss_function: Callable,
                  x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Computes the model gradients for a batch of sample.

        Parameters
        ----------
        model
            Model used for computing influence score.
        loss_function
            Reduction-less loss function to calculate influence (e.g. cross-entropy).
        x
            Batch of inputs on which to compute the gradient.
        y
            Batch of target used to compute the gradient.

        Returns
        -------
        gradient
            Gradient vector for the set of inputs.
        """
        with tf.GradientTape() as tape:
            tape.watch(weights)
            y_pred = loss_function(y, model(x))

        gradients = tape.gradient(y_pred, weights)
        gradients = tf.reshape(gradients, (len(gradients), -1))

        return gradients
