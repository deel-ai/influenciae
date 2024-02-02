# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a technique based on the representer point theorem for kernels,
but using a local jacobian expansion, as per
https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf
"""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential  # pylint: disable=E0611

from ..common import InfluenceModel, InverseHessianVectorProductFactory, BaseInfluenceCalculator
from ..utils import map_to_device, split_model, assert_batched_dataset
from ..types import Union, Optional


class RepresenterPointLJE(BaseInfluenceCalculator):
    """
    Representer Point Selection via Local Jacobian Expansion for Post-hoc Classifier Explanation of Deep Neural
    Networks and Ensemble Models
    https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf

    Disclaimer: This technique requires the last layer of the model to be a Dense layer with no bias.

    Parameters
    ----------
    influence_model
        The TF2.X model implementing the InfluenceModel interface.
    ihvp_calculator_factory
        An InverseHessianVectorProductFactory for creating new instances of the InverseHessianVectorProduct
        class.
    n_samples_for_hessian
        An integer for the amount of samples from the training dataset that will be used for the computation of the
        hessian matrix.
        If None, the whole dataset will be used.
    target_layer
        Either a string or an integer identifying the layer on which to compute the influence-related quantities.
    shuffle_buffer_size
        An integer with the buffer size for the training set's shuffle operation.
    epsilon
        An epsilon value to prevent division by zero.
    """
    def __init__(
            self,
            influence_model: InfluenceModel,
            dataset: tf.data.Dataset,
            ihvp_calculator_factory: InverseHessianVectorProductFactory,
            n_samples_for_hessian: Optional[int] = None,
            target_layer: Union[int, str] = -1,
            shuffle_buffer_size: int = 10000,
            epsilon: float = 1e-5
    ):
        # Make sure that the model's last layer is a Dense layer with no bias
        if not isinstance(influence_model.model.layers[-1], tf.keras.layers.Dense):
            raise ValueError('The last layer of the model must be a Dense layer with no bias.')
        if influence_model.model.layers[-1].use_bias:
            raise ValueError('The last layer of the model must be a Dense layer with no bias.')

        # Make sure that the dataset is batched
        assert_batched_dataset(dataset)

        self.target_layer = target_layer
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)

        # In the paper, the authors explain that in practice, they use a single step of SGD to compute the
        # perturbed model's weights. We will do the same here.
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
        feature_extractor, perturbed_head = split_model(influence_model.model, target_layer)
        target_layer_shape = influence_model.model.layers[target_layer].input.type_spec.shape
        perturbed_head.build(target_layer_shape)
        perturbed_head.compile(optimizer=optimizer, loss=influence_model.loss_function)

        # Get a dataset to compute the SGD step
        if n_samples_for_hessian is None:
            dataset_to_estimate_hessian = map_to_device(dataset, lambda x, y: (feature_extractor(x), y))
        else:
            dataset_to_estimate_hessian = map_to_device(
                dataset.shuffle(shuffle_buffer_size).take(n_samples_for_hessian),
                lambda x, y: (feature_extractor(x), y)
            )

        # Accumulate the gradients for the whole dataset and then update
        trainable_vars = perturbed_head.trainable_variables
        accum_vars = [tf.Variable(tf.zeros_like(t_var.read_value()), trainable=False)
                      for t_var in trainable_vars]
        for x, y in dataset_to_estimate_hessian:
            with tf.GradientTape() as tape:
                y_pred = perturbed_head(x)
                loss = -perturbed_head.loss(y, y_pred)
            gradients = tape.gradient(loss, trainable_vars)
            _ = [accum_vars[i].assign_add(grad) for i, grad in enumerate(gradients)]
        optimizer.apply_gradients(zip(accum_vars, trainable_vars))

        # Keep the feature extractor and the perturbed head
        self.feature_extractor = feature_extractor
        self.perturbed_head = perturbed_head

        # Create the new model with the perturbed weights to compute the hessian matrix
        model = InfluenceModel(self.perturbed_head, 1, loss_function=influence_model.loss_function)  # layer 0 is InputLayer
        self.ihvp_calculator = ihvp_calculator_factory.build(model, dataset_to_estimate_hessian)

    def _reshape_assign(self, weights, influence_vector: tf.Tensor) -> None:
        """
        Updates the model's weights in-place for the Local Jacobian Expansion approximation.

        Parameters
        ----------
        weights
            The weights for which we wish to compute the influence-related quantities.
        influence_vector
            A tensor with the optimizer's stepped weights.
        """
        index = 0
        for w in weights:
            shape = tf.shape(w)
            size = tf.reduce_prod(shape)
            v = influence_vector[index:(index + size)]
            index += size
            v = tf.reshape(v, shape)
            w.assign(w - v)

    def _preprocess_samples(self, samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Preprocess a single batch of samples.

        Parameters
        ----------
        samples
            A single batch of tensors containing the samples.

        Returns
        -------
        evaluate_vect
            The preprocessed sample
        """
        z_batch = self.feature_extractor(samples[:-1])
        y_batch = samples[-1]

        return z_batch, y_batch

    def _compute_alpha(self, z_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        """
        Computes the alpha vector for the Local Jacobian Expansion approximation.

        Parameters
        ----------
        z_batch
            A tensor with the perturbed model's predictions.
        y_batch
            A tensor with the ground truth labels.

        Returns
        -------
        A tensor with the alpha vector for the Local Jacobian Expansion approximation.
        """
        # First, we compute the second term, which contains the Hessian vector product
        weights = self.perturbed_head.trainable_weights
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(weights)
            logits = self.perturbed_head(z_batch)
            loss = self.perturbed_head.compiled_loss(y_batch, logits)
        grads = tape.jacobian(loss, weights)[0]
        grads = tf.multiply(
            grads,
            tf.repeat(
                tf.expand_dims(
                    tf.divide(tf.ones_like(z_batch),
                              tf.cast(tf.shape(z_batch)[0], z_batch.dtype) * z_batch +
                              tf.cast(self.epsilon, z_batch.dtype)),
                    axis=-1),
                grads.shape[-1], axis=-1
            )
        )
        second_term = tf.map_fn(
            lambda v: self.ihvp_calculator._compute_ihvp_single_batch(tf.expand_dims(v, axis=0), use_gradient=False),
            grads
        )  # pylint: disable=protected-access
        second_term = tf.reduce_sum(tf.reshape(second_term, tf.shape(grads)), axis=1)

        # Second, we compute the first term, which contains the weights
        first_term = tf.concat([w for w in weights], axis=0)
        first_term = tf.multiply(
            first_term,
            tf.repeat(
                tf.expand_dims(
                    tf.divide(tf.ones_like(z_batch),
                              tf.cast(tf.shape(z_batch)[0], z_batch.dtype) * z_batch +
                              tf.cast(self.epsilon, z_batch.dtype)),
                    axis=-1),
                first_term.shape[-1], axis=-1
            )
        )
        first_term = tf.reduce_sum(first_term, axis=1)

        return first_term - second_term  # alpha is first term minus second term

    def _compute_influence_value_from_batch(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
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
        z_batch, y_batch = self._preprocess_samples(train_samples)
        alpha = self._compute_alpha(z_batch, y_batch)

        # If the problem is binary classification, take all the alpha values
        # If multiclass, take only those that correspond to the prediction
        out_shape = self.perturbed_head.output_shape
        if len(out_shape) == 1:
            influence_values = alpha
        elif len(out_shape) == 2 and out_shape[1] == 1:
            influence_values = alpha
        else:
            if len(out_shape) > 2:
                indices = tf.argmax(tf.squeeze(self.perturbed_head(z_batch), axis=-1), axis=1)
            else:
                indices = tf.argmax(self.perturbed_head(z_batch), axis=1)
            influence_values = tf.gather(alpha, indices, axis=1, batch_dims=1)

        return influence_values

    def _compute_influence_vector(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
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
            A tensor with a concatenation of the alpha weights and the feature maps for each sample.
            This allows for optimizations to be put in place but is not really an influence vector
            of any kind.
        """
        z_batch = self.feature_extractor(train_samples[:-1])
        alpha = self._compute_alpha(z_batch, train_samples[-1])

        return alpha, z_batch

    def _estimate_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[tf.Tensor, ...],
            samples_to_evaluate: Tuple[tf.Tensor, ...]
    ) -> tf.Tensor:
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
        A tensor containing the individual influence scores.
        """
        return self._estimate_influence_value_from_influence_vector(
            self._preprocess_samples(samples_to_evaluate),
            self._compute_influence_vector(train_samples)
        )

    def _estimate_influence_value_from_influence_vector(
            self,
            preproc_test_sample: tf.Tensor,
            influence_vector: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the influence score for a (batch of) preprocessed test sample(s) and a training "influence vector".

        Parameters
        ----------
        preproc_test_sample
            A tensor with a pre-processed sample to evaluate.
        influence_vector
            A tensor with the training influence vector.

        Returns
        -------
        influence_values
            A tensor with influence values for the (batch of) test samples.
        """
        # Extract the different information inside the tuples
        feature_maps_test, labels_test = preproc_test_sample
        alpha, feature_maps_train = influence_vector

        if len(alpha.shape) == 1 or (len(alpha.shape) == 2 and alpha.shape[1] == 1):
            influence_values = alpha * tf.matmul(feature_maps_train, feature_maps_test, transpose_b=True)
        else:
            if len(self.perturbed_head.output_shape) > 2:
                indices = tf.argmax(tf.squeeze(self.perturbed_head(feature_maps_test), axis=-1), axis=1)
            else:
                indices = tf.argmax(self.perturbed_head(feature_maps_test), axis=1)
            influence_values = tf.gather(alpha, indices, axis=1, batch_dims=1) * \
                               tf.matmul(feature_maps_train, feature_maps_test, transpose_b=True)
        influence_values = tf.transpose(influence_values)

        return influence_values
