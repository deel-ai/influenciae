# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf

from ..common import VectorBasedInfluenceCalculator
from ..types import Tuple
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.regularizers import L2

from ..utils import assert_batched_dataset, BacktrackingLineSearch, dataset_size


class RepresenterPointL2(VectorBasedInfluenceCalculator):
    """
    A class implementing a method to compute the influence of training points through
    the representer point theorem for kernels.

    Parameters
    ----------
    model
        A TF2 model that has already been trained
    train_set
        A batched TF dataset with the points with which the model was trained
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
            lambda_regularization: float,
            scaling_factor: float = 0.1,
            epochs: int = 100,
            layer_index: int = -2,
    ):
        assert_batched_dataset(train_set)
        self.n_train = dataset_size(train_set)
        self.feature_extractor = Model(inputs=model.input, outputs=model.layers[layer_index].output)
        self.model = model
        self.train_set = train_set
        self.lambda_regularization = lambda_regularization
        self.scaling_factor = scaling_factor
        self.epochs = epochs
        self.linear_layer = None
        self.weight_matrix_ds = None
        self._train_last_layer(self.epochs)

    def compute_influence_vector(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Compute the influence vector for a training sample

        Parameters
        ----------
        train_samples
            sample to evaluate
        Returns
        -------
        The influence vector for the training sample
        """
        x_batch = self.feature_extractor(train_samples[:-1])
        alpha = self._compute_gradients(x_batch, train_samples[-1])
        x_batch = tf.reshape(x_batch, (tf.shape(x_batch)[0], -1))
        influence_vectors = tf.repeat(tf.expand_dims(alpha, axis=1), tf.shape(x_batch)[-1], axis=-1) * x_batch

        return influence_vectors

    def preprocess_sample_to_evaluate(self, samples_to_evaluate: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Preprocess a sample to evaluate

        Parameters
        ----------
        samples_to_evaluate
            sample to evaluate
        Returns
        -------
        The preprocessed sample to evaluate
        """
        evaluate_vect = self.feature_extractor(samples_to_evaluate[:-1])
        return evaluate_vect

    def compute_influence_value_from_influence_vector(self, preproc_sample_to_evaluate,
                                                      influence_vector: tf.Tensor) -> tf.Tensor:
        """
        Compute the influence score for a preprocessed sample to evaluate and a training influence vector

        Parameters
        ----------
        preproc_sample_to_evaluate
            Preprocessed sample to evaluate
        influence_vector
            Training influence Vector
        Returns
        -------
        The influence score
        """
        influence_values = tf.matmul(preproc_sample_to_evaluate, tf.transpose(influence_vector))
        return influence_values

    def compute_pairwise_influence_value(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Compute the influence score for a training sample

        Parameters
        ----------
        train_samples
            Training sample
        Returns
        -------
        The influence score
        """
        x_batch = self.feature_extractor(train_samples[:-1])
        influence_values = self._compute_gradients(x_batch, train_samples[-1])

        influence_values = tf.expand_dims(influence_values, axis=1)

        return influence_values

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
        optimizer = BacktrackingLineSearch(batches_per_epoch=self.n_train / self.train_set._batch_size,
                                           scaling_factor=self.scaling_factor)  # the optimizer used in the paper's code
        loss_function = self.model.compiled_loss._losses[0] if isinstance(self.model.compiled_loss._losses, list) \
            else self.model.compiled_loss
        mse_loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        feature_map_ds = self.train_set.map(lambda x, y: self.feature_extractor(x))
        surrogate_train_set = tf.data.Dataset.zip((feature_map_ds,
                                                   feature_map_ds.map(lambda x: self.model.layers[-1](x))))
        self.linear_layer.compile(optimizer=optimizer, loss=mse_loss)
        for epoch in range(epochs):
            for x_batch, y_batch in surrogate_train_set:
                with tf.GradientTape() as tape:
                    logits = self.linear_layer(x_batch, training=True)
                    loss = mse_loss(y_batch, logits)
                gradients = tape.gradient(loss, self.linear_layer.trainable_weights)
                optimizer.step(self.linear_layer, loss, x_batch, y_batch, gradients)
        self.linear_layer.compile(optimizer=optimizer, loss=loss_function)

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

    def _compute_gradients(self, z_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        """
        Computes the gradients of the loss wrt the network's embedding for the point in
        the provided dataset.

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
            tape.watch(z_batch)
            tape.watch(self.linear_layer.trainable_weights)
            logits = self.linear_layer(z_batch)
            loss = self.linear_layer.compiled_loss(y_batch, logits)
        alpha = tf.reduce_mean(tape.gradient(loss, logits), axis=1)
        alpha = tf.divide(alpha, -2. * self.lambda_regularization * tf.cast(self.n_train, alpha.dtype))

        return alpha

    def predict_with_kernel(self, samples_to_evaluate: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        _, dataset_influence = self.compute_influence_values_for_sample_to_evaluate(self.train_set, samples_to_evaluate)
        dataset_influence = dataset_influence.map(lambda x, v: v)
        dataset_iterator = iter(dataset_influence)

        def body_fun(i, value):
            v = next(dataset_iterator)
            i = i + 1
            value = tf.cast(value, v.dtype) + tf.reduce_sum(v, axis=1)
            return i, value

        _, influence_values = tf.while_loop(lambda i, value: i < dataset_influence.cardinality(), body_fun,
                                            [tf.constant(0, dtype=tf.int64),
                                             tf.zeros((tf.shape(samples_to_evaluate[-1])[0],), dtype=tf.float32)])

        return influence_values
