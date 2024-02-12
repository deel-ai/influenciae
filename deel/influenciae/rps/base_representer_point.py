# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module containing the base class for representer point theorem-based influence calculators
"""
from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, Reduction

from ..common import BaseInfluenceCalculator
from ..types import Tuple, Callable, Union

from ..utils import assert_batched_dataset, split_model


class BaseRepresenterPoint(BaseInfluenceCalculator):
    """
    Base interface for representer point theorem-based influence calculators.

    Disclaimer: This method only works on classification problems!

    Parameters
    ----------
    model
        A TF2 model that has already been trained
    train_set
        A batched TF dataset with the points with which the model was trained
    loss_function
        The loss function with which the model was trained. This loss function MUST NOT be reduced.
    """
    def __init__(
            self,
            model: Model,
            train_set: tf.data.Dataset,
            loss_function: Union[Callable[[tf.Tensor, tf.Tensor], tf.Tensor], Loss],
            target_layer: Union[str, int] = -1
    ):
        # Make sure that the dataset is batched and that the loss function is not reduced
        assert_batched_dataset(train_set)
        self.train_set = train_set
        if hasattr(loss_function, 'reduction'):
            assert loss_function.reduction == Reduction.NONE

        # Make sure that the model's last layer is a Dense layer with no bias
        if not isinstance(model.layers[-1], tf.keras.layers.Dense):
            raise ValueError('The last layer of the model must be a Dense layer with no bias.')
        if model.layers[-1].use_bias:
            raise ValueError('The last layer of the model must be a Dense layer with no bias.')
        self.loss_function = loss_function

        # Cut the model in two (feature extractor and head)
        self.model = model
        self.target_layer = target_layer
        self.feature_extractor, self.original_head = split_model(model, target_layer)

    @abstractmethod
    def _compute_alpha(self, z_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        """
        Compute the alpha vector for a given input-output pair (z, y)

        Parameters
        ----------
        z_batch
            A tensor containing the latent representation of an input point.
        y_batch
            The labels corresponding to the representations z

        Returns
        -------
        alpha
            A tensor with the alpha coefficients of the kernel given by the representer point theorem
        """
        raise NotImplementedError()

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
        x_batch = self.feature_extractor(samples[:-1])
        y_t = samples[-1]

        return x_batch, y_t

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
        x_batch = self.feature_extractor(train_samples[:-1])
        alpha = self._compute_alpha(x_batch, train_samples[-1])

        return alpha, x_batch

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
        feature_maps_test, _ = preproc_test_sample
        alpha, feature_maps_train = influence_vector

        if len(alpha.shape) == 1 or (len(alpha.shape) == 2 and alpha.shape[1] == 1):
            influence_values = alpha * tf.matmul(feature_maps_train, feature_maps_test, transpose_b=True)
        else:
            influence_values = tf.gather(
                alpha, tf.argmax(self.original_head(feature_maps_test), axis=1), axis=1, batch_dims=1
            ) * tf.matmul(feature_maps_train, feature_maps_test, transpose_b=True)
        influence_values = tf.transpose(influence_values)

        return influence_values

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
        x_batch = self.feature_extractor(train_samples[:-1])
        alpha = self._compute_alpha(x_batch, train_samples[-1])

        # If the problem is binary classification, take all the alpha values
        # If multiclass, take only those that correspond to the prediction
        out_shape = self.model.output_shape
        if len(out_shape) == 1:
            influence_values = alpha
        elif len(out_shape) == 2 and out_shape[1] == 1:
            influence_values = alpha
        else:
            if len(out_shape) > 2:
                indices = tf.argmax(tf.squeeze(self.original_head(x_batch), axis=-1), axis=1)
            else:
                indices = tf.argmax(self.original_head(x_batch), axis=1)
            influence_values = tf.gather(alpha, indices, axis=1, batch_dims=1)

        return tf.abs(influence_values)
