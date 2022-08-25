import tensorflow as tf
from typing import Tuple

from .influence_calculator import InverseHessianVectorProduct
from ..common import InfluenceModel
from tensorflow.keras.models import Sequential


class RPSLJE:
    """
    Representer Point Selection via Local Jacobian Expansion for Post-hoc Classifier Explanation of Deep Neural
    Networks and Ensemble Models
    https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    ihvp_calculator
        Either a string containing the IHVP method ('exact' or 'cgd'), an IHVPCalculator
        object or an InverseHessianVectorProduct object.
    """

    def __init__(
            self,
            model: InfluenceModel,
            ihvp_calculator: InverseHessianVectorProduct
    ):
        self.model = model
        self.ihvp_calculator = ihvp_calculator
        self.feature_model = Sequential(self.model.layers[:-self.model.target_layer])

    def compute_influence_vector(self, train_samples: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
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
        batch_size = tf.shape(train_samples[0])[0]

        # TODO - improve: API IHVP shall accept tensor
        ihvp = self.ihvp_calculator.compute_ihvp(
            tf.data.Dataset.from_tensor_slices(train_samples).batch(int(tf.shape(train_samples[0])[0])))

        vec_weight = tf.repeat(tf.reshape(self.model.weights, (1, -1)), batch_size, axis=0)

        ihvp = tf.transpose(vec_weight) - ihvp

        return ihvp

    def preprocess_sample_to_evaluate(self, samples_to_evaluate: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
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
        evaluate_vect = self.feature_model(samples_to_evaluate[0])
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
        influence_values = tf.matmul(preproc_sample_to_evaluate, influence_vector)
        return influence_values

    def compute_pairwise_influence_value(self, train_samples: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
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
        ihvp = self.compute_influence_vector(train_samples)
        evaluate_vect = self.preprocess_sample_to_evaluate(train_samples)
        influence_values = tf.reduce_sum(
            tf.math.multiply(evaluate_vect, tf.transpose(ihvp)), axis=1, keepdims=True)

        return influence_values
