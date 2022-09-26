# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential

from ..common import InfluenceModel, VectorBasedInfluenceCalculator
from ..common import InverseHessianVectorProduct

from ..utils import from_layer_name_to_layer_idx
from ..types import Union, Tuple

class RPSLJE(VectorBasedInfluenceCalculator):
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
    target_layer
        TODO: Define the argument
    """

    def __init__(
            self,
            model: InfluenceModel,
            ihvp_calculator: InverseHessianVectorProduct,
            target_layer: Union[int, str]
    ):
        self.model = model
        self.ihvp_calculator = ihvp_calculator
        if isinstance(target_layer, str):
            target_layer = from_layer_name_to_layer_idx(model, target_layer)

        self.feature_model = Sequential(self.model.layers[:target_layer])

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
        # batch_size = tf.shape(train_samples[0])[0]

        # TODO - improve: API IHVP shall accept tensor
        ihvp = self.ihvp_calculator.compute_ihvp_single_batch(train_samples)

        vec_weight = tf.concat([tf.reshape(w, (1, -1)) for w in self.model.weights], axis=1)
        vec_weight = tf.repeat(vec_weight, tf.shape(ihvp)[1], axis=0)
        #TODO: ask questions here
        ihvp = tf.cast(vec_weight, dtype=ihvp.dtype) - tf.transpose(ihvp)

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
        influence_values = tf.matmul(preproc_sample_to_evaluate, tf.transpose(influence_vector))
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
        evaluate_vect = tf.cast(evaluate_vect, dtype=ihvp.dtype)
        influence_values = tf.reduce_sum(
            tf.math.multiply(evaluate_vect, ihvp), axis=1, keepdims=True)

        return influence_values
