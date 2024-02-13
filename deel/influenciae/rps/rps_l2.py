# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing the representer point theorem for kernels for estimating the
influence of training data-points, as per:
https://arxiv.org/abs/1811.09720
"""
import tensorflow as tf
from tensorflow.keras import Model #pylint:  disable=E0611
from tensorflow.keras.layers import Input, Dense #pylint:  disable=E0611
from tensorflow.keras.losses import MeanSquaredError, Loss #pylint:  disable=E0611
from tensorflow.keras.regularizers import L2 #pylint:  disable=E0611

from .base_representer_point import BaseRepresenterPoint
from ..types import Tuple, Callable, Union

from ..utils import BacktrackingLineSearch, dataset_size


class RepresenterPointL2(BaseRepresenterPoint):
    """
    A class implementing a method to compute the influence of training points through
    the representer point theorem for kernels.

    It builds a kernel that approximates the model's last layer such that for a training
    point x_i and a test point x_t with label y_t:

    y_t = sum_i k(alpha_i, x_i, x_t)

    Disclaimer: This method only works on classification problems!

    Parameters
    ----------
    model
        A TF2 model that has already been trained
    train_set
        A batched TF dataset with the points with which the model was trained
    loss_function
        The loss function with which the model was trained. This loss function MUST NOT be reduced.
    lambda_regularization
        The coefficient for the regularization of the surrogate last layer that needs
        to be trained for this method
    scaling_factor
        A float with the scaling factor for the SGD backtracking line-search optimizer
        for fitting the surrogate linear model
    epochs
        An integer for the amount of epochs to fit the linear model
    layer_index
        layer of the logits
    """

    def __init__(
            self,
            model: Model,
            train_set: tf.data.Dataset,
            loss_function: Union[Callable[[tf.Tensor, tf.Tensor], tf.Tensor], Loss],
            lambda_regularization: float,
            scaling_factor: float = 0.1,
            epochs: int = 100,
            layer_index: int = -1,
    ):
        super().__init__(model, train_set, loss_function, layer_index)
        self.n_train = dataset_size(train_set)
        self.train_set = train_set
        self.lambda_regularization = lambda_regularization
        self.scaling_factor = scaling_factor
        self.epochs = epochs
        self.linear_layer = None
        self._train_last_layer(self.epochs)

    def _train_last_layer(self, epochs: int):
        """
        Trains an L2-regularized surrogate linear model to predict like the model on the
        training dataset. The optimization is done using a Backtracking Line-Search
        algorithm with the Armijo condition and SGD as the optimizer as it was done
        in the original implementation.

        Parameters
        ----------
        epochs
            An integer with the amount of epochs to train the surrogate model
        """
        self.linear_layer = self._create_surrogate_model()
        optimizer = BacktrackingLineSearch(batches_per_epoch=self.n_train / self.train_set._batch_size,  # pylint: disable=W0212
                                           scaling_factor=self.scaling_factor)  # the optimizer used in the paper's code
        loss_function = self.loss_function
        mse_loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        self.linear_layer.compile(optimizer=optimizer, loss=mse_loss)
        for _ in range(epochs):
            for x_batch, _ in self.train_set:
                loss, grads, z_batch, y_target = self._learn_step_last_layer(x_batch, mse_loss)
                optimizer.step(self.linear_layer, loss, z_batch, y_target, grads)

        self.linear_layer.compile(optimizer=optimizer, loss=loss_function)

    @tf.function
    def _learn_step_last_layer(
            self,
            x_batch: tf.Tensor,
            mse_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Trains the L2-regularized surrogate linear model to predict like the model on the training
        dataset for one optimizer step on a single batch.

        Parameters
        ----------
        x_batch
            A training sample wrt to which we wish to compute the gradients
        mse_loss
            A callable that computes the MSE loss

        Returns
        -------
        A tuple with (value of the loss, gradients of the linear model, the latent space of the batch, the prediction)
        """
        z_batch = self.feature_extractor(x_batch)
        y_target = self.model.layers[-1](z_batch)
        with tf.GradientTape() as tape:
            logits = self.linear_layer(z_batch, training=True)
            loss = mse_loss(y_target, logits)
        gradients = tape.gradient(loss, self.linear_layer.trainable_weights)
        return loss, gradients, z_batch, y_target

    def _create_surrogate_model(self) -> Model:
        """
        Instances an L2-regularized linear model to use as surrogate with the
        right input and output shapes.

        Returns
        -------
        surrogate_model
            A TF2 L2-regularized linear model
        """
        inputs = Input(shape=self.feature_extractor.output_shape[1:], dtype=self.model.output.dtype)
        last_layer = Dense(self.model.output_shape[-1], use_bias=False,
                           kernel_regularizer=L2(self.lambda_regularization),
                           dtype=self.model.output.dtype)
        outputs = last_layer(inputs)
        surrogate_model = Model(inputs=inputs, outputs=outputs)
        surrogate_model.layers[-1].trainable = True
        surrogate_model.compile(loss=self.model.compiled_loss)

        return surrogate_model

    def _compute_alpha(self, z_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        """
        Computes the alpha factor for the kernel approximation. This element gives a notion of
        the resistance that each training data-point towards minimizing the norm of the linear
        layer's weight matrix. This is essentially this method's notion of influence score.

        Parameters
        ----------
        z_batch
            a training sample wrt to which we wish to compute the gradients
        y_batch
            label associated to the training sample

        Returns
        -------
        alpha
            mean of the gradient
        """
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(self.linear_layer.weights)
            logits = self.linear_layer(z_batch)
            loss = self.linear_layer.compiled_loss(y_batch, logits)
        alpha = tape.jacobian(loss, self.linear_layer.weights)[0]
        alpha = tf.divide(
            alpha,
            -2. * self.lambda_regularization * tf.cast(self.n_train, alpha.dtype) + tf.constant(1e-5, dtype=alpha.dtype)
        )

        # Now, divide each of the alpha_i by their feature maps
        alpha = tf.multiply(
            alpha,
            tf.repeat(
                tf.expand_dims(
                    tf.divide(tf.ones_like(z_batch), z_batch + tf.constant(1e-5, dtype=alpha.dtype)),
                    axis=-1),
                alpha.shape[-1], axis=-1
            )
        )  # Do the multiplication part of the inner product
        alpha = tf.reduce_sum(alpha, axis=1)  # Now do the sum

        return alpha

    def predict_with_kernel(self, samples_to_evaluate: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Uses the learned kernel to approximate the model's predictions on a group of samples.

        Parameters
        ----------
        samples_to_evaluate
            A single batch of tensors with the samples for which we wish to approximate the model's
            predictions

        Returns
        -------
        predictions
            A tensor with an approximation of the model's predictions
        """
        influence_vectors = self.compute_influence_vector(self.train_set)
        _, dataset_influence = self._estimate_inf_values_with_inf_vect_dataset(influence_vectors, samples_to_evaluate)
        dataset_influence = dataset_influence.map(lambda x, v: v)
        dataset_iterator = iter(dataset_influence)

        def body_fun(i, value):
            v = next(dataset_iterator)
            i = i + 1
            value = tf.cast(value, v.dtype) + tf.reduce_sum(v, axis=1)
            return i, value

        _, predictions = tf.while_loop(lambda i, value: i < dataset_influence.cardinality(), body_fun,
                                            [tf.constant(0, dtype=tf.int64),
                                             tf.zeros((tf.shape(samples_to_evaluate[-1])[0],), dtype=tf.float32)])

        return predictions
