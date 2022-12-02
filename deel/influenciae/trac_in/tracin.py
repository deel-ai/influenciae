# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing the TracIn method originally presented in
https://arxiv.org/pdf/2002.08484.pdf
It computes the influence of each training point by tracing the gradients during the
training phase. In practice, we will use the model at different checkpoints, achieving
a more efficient estimation at the cost of a little precision.
"""
import tensorflow as tf

from ..common import InfluenceModel, BaseInfluenceCalculator
from ..types import Union, List, Tuple


class TracIn(BaseInfluenceCalculator):
    """
    A class implementing an influence score based on TracIn method proposed in
    [https://arxiv.org/pdf/2002.08484.pdf](https://arxiv.org/pdf/2002.08484.pdf)

    Notes
    -----
    This method traces the gradients of each of the training data-points as a means to
    compute their influence in the model. In practice, instead of saving the gradients
    at each epoch, the model's checkpoints are used, with the checkpoint frequency
    allowing for a trade-off between the precision of the computed influence and its
    computational cost.

    Parameters
    ----------
    models
        A list of TF2.X models implementing the InfluenceModel interface at different steps (epochs)
        of the training
    learning_rates
        Learning rate or list of learning rates used during the training.
        If learning_rates is a list, it should have the same size as the amount of models
    """
    def __init__(self, models: List[InfluenceModel], learning_rates: Union[float, List[float]]):
        self.models = models

        if isinstance(learning_rates, List):
            assert len(models) == len(learning_rates)
            self.learning_rates = learning_rates
        else:
            self.learning_rates = [learning_rates for _ in range(len(models))]

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
            A tensor with the influence for each sample.
        """
        influence_vectors = []
        for model, lr in zip(self.models, self.learning_rates):
            g_train = model.batch_jacobian_tensor(train_samples)
            influence_vectors.append(g_train * tf.cast(tf.sqrt(lr), g_train.dtype))
        influence_vectors = tf.concat(influence_vectors, axis=1)

        return influence_vectors

    def _preprocess_samples(self, samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Pre-process a sample to facilitate evaluation afterwards. In this case, it amounts to transforming
        it into it's "influence vector".

        Parameters
        ----------
        samples
            A tensor with the group of samples we wish to evaluate.
        Returns
        -------
        evaluate_vect
            A tensor with the pre-processed samples.
        """
        evaluate_vect = self._compute_influence_vector(samples)
        return evaluate_vect

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
        Compute the influence score of a (pre-processed) sample and an "influence vector" from a training
        data-point

        Parameters
        ----------
        preproc_test_sample
            A tensor with a (pre-processed) test sample
        influence_vector
            A tensor with an "influence vector" calculated using a training point

        Returns
        -------
        influence_values
            A tensor with the influence scores
        """
        influence_values = tf.matmul(preproc_test_sample, tf.transpose(influence_vector))
        return influence_values

    def _compute_influence_value_from_batch(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
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
        influence_vector = self._compute_influence_vector(train_samples)
        influence_values = tf.reduce_sum(influence_vector * influence_vector, axis=1, keepdims=True)
        return influence_values
