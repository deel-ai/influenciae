# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TODO: Insert short introduction
"""
import tensorflow as tf

from ..common import InfluenceModel, VectorBasedInfluenceCalculator
from ..types import Union, List, Tuple

class TracIn(VectorBasedInfluenceCalculator):
    """
    A class implementing an influence score based on Tracin method
    https://arxiv.org/pdf/2002.08484.pdf

    The method evaluates the influence values of samples over a set of training points

    Parameters
    ----------
    models
        A list of TF2.X models implementing the InfluenceModel interface at different step (epoch)
        of the training
    learning_rates: Learning rate or list of learning used during the training.
        If learning_rates is a list, it shall be have the same size of the models argument
    """
    def __init__(self, models: List[InfluenceModel], learning_rates: Union[float, List[float]]):
        self.models = models

        if isinstance(learning_rates, List):
            assert len(models) == len(learning_rates)
            self.learning_rates = learning_rates
        else:
            self.learning_rates = [learning_rates for _ in range(len(models))]


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
        influence_vectors = []
        for model, lr in zip(self.models, self.learning_rates):
            g_train = model.batch_jacobian_tensor(train_samples)
            influence_vectors.append(g_train * tf.cast(tf.sqrt(lr), g_train.dtype))
        influence_vectors = tf.concat(influence_vectors, axis=1)

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
        evaluate_vect = self.compute_influence_vector(samples_to_evaluate)
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
            Training influence Vvctor
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
        influence_vector = self.compute_influence_vector(train_samples)
        influence_values = tf.reduce_sum(influence_vector * influence_vector, axis=1, keepdims=True)
        return influence_values
