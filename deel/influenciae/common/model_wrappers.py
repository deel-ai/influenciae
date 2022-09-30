# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Custom wrappers for tensorflow model
"""
import itertools

import tensorflow as tf
from tensorflow.keras.losses import Reduction  # pylint: disable=E0611

from ..utils import assert_batched_dataset, from_layer_name_to_layer_idx, default_process_batch
from ..types import Callable, Optional, Union, List, Tuple

ProcessBatchTypeAlias = Callable[[Tuple[tf.Tensor, ...]], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]

class BaseInfluenceModel:
    """
    A generic Tensorflow model wrapper for Influence functions.

    Parameters
    ----------
    model
        Model used for computing influence score.
    weights_to_watch
        List of the model weights to watch when computing gradients, jacobians & hessians
    loss_function
        Loss function to calculate influence (e.g. keras CategoricalCrossentropy). Make sure not to
        apply any reduction (Reduction.NONE), and specify correctly if the output is `from_logits`
        for example.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 weights_to_watch: Optional[List[tf.Variable]] = None,
                 loss_function: Callable = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=False, reduction=Reduction.NONE),
                 process_batch_for_loss_fn: ProcessBatchTypeAlias = default_process_batch,
                 weights_processed: bool = False):

        if hasattr(loss_function, 'reduction') and loss_function.reduction is not Reduction.NONE:
            raise ValueError('The loss function must not have reduction.')

        self.model = model
        self.weights_processed = weights_processed
        if weights_to_watch is None:
            weights_to_watch =[layer.weights for layer in model.layers]
            self.weights_processed = False
        # "flatten" the list of weights and remove empty weights
        self.weights = self.__process_weights_list(weights_to_watch)

        self.nb_params = tf.reduce_sum([tf.size(w) for w in self.weights])
        self.loss_function = loss_function
        self.process_batch_for_loss_fn = process_batch_for_loss_fn

    def __call__(self, inps: tf.Tensor) -> tf.Tensor:
        """
        Forward of the original model

        Parameters
        ----------
        inps
            Inputs on which to make the inference.

        Returns
        -------
        y
            Outputs of the original model.
        """
        return self.model(inps)

    def __process_weights_list(self, weights_to_watch):
        """
        Ensure a proper formatting of the weights
        TODO: Improve it, cause list(itertools.chain(*weights_to_watch)) is not idempotent
        """
        if self.weights_processed:
            return weights_to_watch

        return list(itertools.chain(*weights_to_watch))

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

    @staticmethod
    @tf.function
    def _loss(
        model: tf.keras.Model,
        loss_function: Callable,
        batch: Tuple[tf.Tensor, ...],
        process_batch_for_loss_fn: ProcessBatchTypeAlias) -> tf.Tensor:
        """
        Computes the model's loss for a single batch of samples.

        Parameters
        ----------
        model
            Model used for computing influence score.
        loss_function
            Reduction-less loss function to calculate influence (e.g. cross-entropy).
        batch
            TODO: rewrite doc
        Returns
        -------
        loss_values
            Loss values for each inputs (i.e not reduced).
        """
        model_inp, y_true, sample_weight = process_batch_for_loss_fn(batch)
        return loss_function(y_true, model(model_inp), sample_weight)

    @staticmethod
    @tf.function
    def _jacobian(model: tf.keras.Model, weights: tf.Tensor, loss_function: Callable,
                  batch: Tuple[tf.Tensor, ...], process_batch_for_loss_fn: ProcessBatchTypeAlias) -> tf.Tensor:
        """
        Computes the model's jacobian for a single batch of samples.

        Parameters
        ----------
        model
            Model used for computing influence score.
        loss_function
            Reduction-less loss function to calculate influence (e.g. cross-entropy).
        batch
            TODO

        Returns
        -------
        jacobian
            Jacobian matrix for the set of inputs.
        """
        model_inp, y_true, sample_weight = process_batch_for_loss_fn(batch)
        batch_size = tf.shape(y_true)[0]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(weights)
            y_pred = loss_function(y_true, model(model_inp), sample_weight)

        jacobian = tape.jacobian(y_pred, weights)

        jacobian = [tf.reshape(j, (batch_size, -1,)) for j in jacobian]
        jacobian = tf.concat(jacobian, axis=1)

        return jacobian

    @staticmethod
    @tf.function
    def _gradient(model: tf.keras.Model, weights: tf.Variable, loss_function: Callable,
                  batch: Tuple[tf.Tensor, ...], process_batch_for_loss_fn: ProcessBatchTypeAlias) -> tf.Tensor:
        """
        Computes the model gradients for a single batch of sample.

        Parameters
        ----------
        model
            Model used for computing influence score.
        loss_function
            Reduction-less loss function to calculate influence (e.g. cross-entropy).
        batch
            TODO: Docs

        Returns
        -------
        gradient
            Gradient vector for the set of inputs.
        """
        model_inp, y_true, sample_weight = process_batch_for_loss_fn(batch)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(weights)
            y_pred = tf.expand_dims(loss_function(y_true, model(model_inp), sample_weight), axis=-1)

        gradients = tape.gradient(y_pred, weights)
        # note that it is the accumulated gradients for all inputs in the batch
        gradients = [tf.reshape(g, (-1,)) for g in gradients]
        gradients = tf.concat(gradients, axis=0)

        return gradients

    def _loss_tensor(self, batch: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Computes the model's loss on the batched tensor

        Parameters
        ----------
        batch
            TODO: Docs

        Returns
        -------
        loss_values
            Loss values for each of the points of the batch.
        """
        loss_values = BaseInfluenceModel._loss(self.model, self.loss_function, batch, self.process_batch_for_loss_fn)

        return loss_values

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
            BaseInfluenceModel._loss(self.model, self.loss_function, batch, self.process_batch_for_loss_fn)
            for batch in dataset
        ], axis=0)

        return loss_values

    @tf.function
    def batch_jacobian_tensor(self, batch: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Computes the jacobian of the loss wrt the weights of the start_layer on a Tensor

        Parameters
        ----------
        batch
            TODO: Docs

        Returns
        -------
        jacobians
            Matrix of the first-order partial derivative of the loss function wrt the
            start_layer weights.
        """

        jacobians = BaseInfluenceModel._jacobian(self.model, self.weights, self.loss_function,
                                             batch, self.process_batch_for_loss_fn)

        return jacobians

    def batch_jacobian(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Computes the jacobian of the loss wrt the weights of the start_layer on the whole
        batched dataset.

        Parameters
        ----------
        dataset
            Batched dataset on which to compute the jacobian.

        Returns
        -------
        jacobians
            Matrix of the first-order partial derivative of the loss function wrt the
            start_layer weights.
        """
        assert_batched_dataset(dataset)

        jacobians = tf.concat([
            BaseInfluenceModel._jacobian(self.model, self.weights, self.loss_function,
                                         batch, self.process_batch_for_loss_fn)
            for batch in dataset
        ], axis=0)

        return jacobians

    @tf.function
    def batch_gradient_tensor(self, batch: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Computes the gradient of the loss wrt the weights of the start_layer on a Tensor

        Parameters
        ----------
        batch
            TODO: Docs

        Returns
        -------
        gradients
            Gradient values of the loss function wrt the start_layer's weights.
        """
        gradients = BaseInfluenceModel._gradient(self.model, self.weights, self.loss_function,
                                             batch, self.process_batch_for_loss_fn)

        return gradients

    def batch_gradient(self, dataset) -> tf.Tensor:
        """
        Computes the gradient of the loss wrt the weights of the start_layer on the whole
        batched dataset.

        Parameters
        ----------
        dataset
            Batched dataset on which to compute the gradient.

        Returns
        -------
        gradients
            Gradient values of the loss function wrt the start_layer's weights.
        """
        assert_batched_dataset(dataset)

        gradients = tf.stack([
            BaseInfluenceModel._gradient(self.model, self.weights, self.loss_function,
                                         batch, self.process_batch_for_loss_fn)
            for batch in dataset
        ])

        return gradients

class InfluenceModel(BaseInfluenceModel):
    """
    A Tensorflow model wrapper for Influence functions which only require the first layer
    index or name from which we will watch the weights (e.g you ignore the feature extractor).

    Parameters
    ----------
    model
        Model used for computing influence score.
    start_layer
        Starting layer name or index for the weights and bias collection. If set to None,
        will search for the last layer with weights before logits.
    last_layer
        Last layer name or index for the weights and biases collection.
        If set to None, only the layer indicated in the start_layer parameter will be used.
    loss_function
        Loss function to calculate influence (e.g. keras CategoricalCrossentropy). Make sure not to
        apply any reduction (Reduction.NONE), and specify correctly if the output is `from_logits`
        for example.
    process_batch_for_loss_fn
        TODO: docs
    """
    def __init__(self,
                 model: tf.keras.Model,
                 start_layer: Optional[Union[str, int]] = None,
                 last_layer: Optional[Union[str, int]] = None,
                 loss_function: Callable = tf.keras.losses.CategoricalCrossentropy(
                     from_logits=False, reduction=Reduction.NONE),
                 process_batch_for_loss_fn: ProcessBatchTypeAlias = default_process_batch):

        weights_to_watch = InfluenceModel._get_weights_of_interest(model, start_layer, last_layer)
        super().__init__(model, weights_to_watch, loss_function, process_batch_for_loss_fn, weights_processed=True)

    @staticmethod
    def _get_weights_of_interest(model: tf.keras.Model,
                                 start_layer: Optional[Union[str, int]],
                                 last_layer: Optional[Union[str, int]]) -> list:
        """
        Gets the list of trainable weights from layer 'start_layer' to layer 'last_layer' in model

        Parameters
        ----------
        model
            Model we want to get the weights from.
        start_layer
            Starting layer for the weights and bias collection. If set to None, will search for the
            last layer with weights before logits.
        last_layer
            Last layer for the weights and biases collection used in hessian computation. If set to
            None, only the layer indicated in the start_layer parameter will be used.

        Returns
        -------
        weights
            A flatten list of weights between the start_layer and the last_layer layers in model
        """
        # get an id value for the start_layer parameter
        if start_layer is None:
            start_layer = InfluenceModel._find_last_weight_layer(model)
            start_layer = len(model.layers) + start_layer
        elif isinstance(start_layer, str):
            start_layer = from_layer_name_to_layer_idx(model, start_layer)
        else:
            assert(isinstance(start_layer, int)), "start_layer should be None, a string or an int"

        # get the list of layers of interest
        if last_layer is None:
            layers_for_influence = [model.layers[start_layer]]
        elif isinstance(last_layer, str):
            last_layer = from_layer_name_to_layer_idx(model, last_layer)
            assert last_layer >= start_layer, \
                f"last_layer id: {last_layer} should be greater than start_layer id: {start_layer}"
            layers_for_influence = model.layers[start_layer : last_layer+1]
            start_layer = last_layer
        else:
            assert(isinstance(last_layer, int)), "last_layer should be None, a string or an int"
            if last_layer < 0:
                last_layer += len(model.layers)
                assert last_layer >= start_layer, \
                    f"last_layer id: {last_layer} should be greater than start_layer id: {start_layer}"
            elif last_layer == 0:
                assert last_layer == start_layer, \
                    f"last_layer id: {last_layer} should be greater than start_layer id: {start_layer}"
            layers_for_influence = model.layers[start_layer : last_layer+1]

        # get the list of weights of interest
        weights = [lay.weights for lay in layers_for_influence]
        weights = list(itertools.chain(*weights))
        return weights

    @staticmethod
    def _find_last_weight_layer(model: tf.keras.Model) -> int:
        """
        Find and return the id of the last layer before logits with weights.

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
