# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a data-point-influence estimation method based on how intensely
the model's weights must be adversarially attacked to make it change its prediction
for each sample. Intuitively, outliers and atypical examples will need a lighter
deformation of the boundary for it to place them in the wrong class.

Unlike other influence calculators, this one cannot be used to estimate
the influence of a point on another.

This boundary deformation process is performed using deep fool on the target weights.
"""
import tensorflow as tf
from tensorflow.keras import Model  # pylint:  disable=E0611

from ..common import SelfInfluenceCalculator
from ..types import Tuple, List


class WeightsBoundaryCalculator(SelfInfluenceCalculator):
    """
    A class implementing an influence score based on the distance of a sample to the boundary of its classifier.
    The distance to the boundary is estimated by deforming the boundary of the model to move a given sample
    to the closest adversarial class.
    To compute this distance, the deep fool method is used on the weights of the model (deep fool originally compute
    the distance on the sample space).
    [https://arxiv.org/abs/1511.04599]

    Notes
    -----
    This method has better mislabeled-sample-detection performance when the model overfits.

    Parameters
    ----------
    model
        A TF2 model that has already been trained
    step_nbr
        Number of the iterations to find the closest adversarial problem
    norm_type
        The distance norm used to compute the distance to the boundary
    eps
        Difference between two logits to assume that the logits have the same values
    """

    def __init__(self, model: Model, step_nbr: int = 100, norm_type: int = 2, eps: float = 1E-6):
        self.weights_init = [tf.identity(w) for w in model.trainable_variables]
        self.model = model

        self.step_nbr = step_nbr
        self.norm_type = norm_type
        self.eps = eps

    def __compute_norm(self, weights: List[tf.Tensor]) -> tf.float32:
        """
        Compute the norm of a list of weights

        Parameters
        ----------
        weights
            The list of weights

        Returns
        -------
        weights_norm
            The norm of the weights
        """
        weights_flatten = tf.concat([tf.reshape(w, (-1,)) for w in weights], axis=0)
        weights_norm = tf.norm(weights_flatten, ord=self.norm_type)

        return weights_norm

    def __delta_to_index(self, indexes_1: tf.Tensor, indexes_2: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the difference between the logit of a given class and the other logits

        Parameters
        ----------
        indexes_1
            The logits of other classes
        indexes_2
            The logits of the predicted class
        x
            The logits

        Returns
        -------
        delta_x
            The difference between the logits
        """
        x1 = tf.gather(x, indexes_1, batch_dims=1)
        x2 = tf.gather(x, tf.expand_dims(indexes_2, axis=1), batch_dims=1)

        delta_x = x1 - tf.repeat(x2, tf.shape(x1)[1], axis=1)

        return delta_x

    @tf.function
    def __step(self, x: tf.Tensor, y_pred: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The optimization step to find the distance between the boundary and a given sample x.
        To see more details about the optimization procedure for multi-class classifiers,
        please refer to [https://arxiv.org/abs/1511.04599]

        Notes
        -----
        This function updates the weights of the model at each step.

        Parameters
        ----------
        x
            The current sample used to compute the distance to the boundary of the model
        y_pred
            The one-hot labels predicted by the current model for the x sample

        Returns
        -------
        computation
            Boolean to determine if the optimization process should continue.
            True if the sample didn't change of class, False if not.
        loss_value
            The loss of the optimization procedure
        """
        y_pred = tf.argmax(y_pred, axis=1)

        with tf.GradientTape() as tape:
            y = self.model(x)

        y_computed = tf.argmax(y, axis=1)

        def update_grads():
            model_weights = self.model.trainable_variables
            jac = tape.jacobian(y, self.model.trainable_variables)

            indexes_all = tf.repeat(tf.expand_dims(tf.range(0, tf.shape(y)[1]), axis=0), tf.shape(y)[0], axis=0)
            indexes_class = tf.cast(tf.repeat(tf.expand_dims(y_pred, axis=1), tf.shape(y)[1], axis=1), dtype=tf.int32)
            indexes_other = tf.reshape(indexes_all[indexes_all != indexes_class], (-1, tf.shape(y)[1] - 1))

            delta_y = self.__delta_to_index(indexes_other, y_pred, y)
            delta_y = tf.abs(tf.reduce_mean(delta_y, axis=0))

            jac_delta = [tf.reduce_mean(self.__delta_to_index(indexes_other, y_pred, j), axis=0) for j in jac]

            jac_norm = tf.concat([tf.reshape(j, (tf.shape(j)[0], -1)) for j in jac_delta], axis=1)
            jac_norm = tf.norm(jac_norm, axis=1, ord=self.norm_type)

            coeff = delta_y / jac_norm

            best_class = tf.argmin(coeff, axis=0)

            loss = (coeff / tf.pow(jac_norm, self.norm_type - 1))[best_class]

            weights = [w + loss * tf.pow(tf.abs(g[best_class]), self.norm_type - 1) * tf.sign(g[best_class]) for w, g in
                       zip(model_weights, jac_delta)]

            for m, w in zip(model_weights, weights):
                m.assign(w)

            return loss

        computation = tf.reduce_any(y_computed == y_pred)

        top_k, _ = tf.math.top_k(tf.squeeze(y, axis=0), k=2)
        is_close_enough = tf.abs(top_k[0] - top_k[1]) > self.eps

        computation = tf.logical_and(computation, is_close_enough)

        loss_value = tf.cond(computation, update_grads, lambda: tf.constant(0.0))

        return computation, loss_value

    def __delta_weights(self) -> tf.float32:
        """
        Compute the norm between the trained weights of the model and the current modified weights of the model.

        Returns
        -------
        norm
            The distance between the initial model and the current model
        """
        weights = [w1 - w2 for w1, w2 in zip(self.weights_init, self.model.trainable_variables)]
        norm = self.__compute_norm(weights)

        return norm

    def __reset_weights(self) -> None:
        """
        Set the weights of the model to the initial trained weights.
        """
        for m, w in zip(self.model.trainable_variables, self.weights_init):
            m.assign(w)

    def __compute_single_sample_score(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the influence score (self-influence) for a single training samples.

        Parameters
        ----------
        x
            A tensor with a single training sample.

        Returns
        -------
        score
            The influence score of the sample.
        """

        x = tf.expand_dims(x, axis=0)
        y_pred = self.model(x)

        tf.while_loop(lambda cond, index: tf.logical_and(cond, index < self.step_nbr),
                      lambda cond, index: (self.__step(x, y_pred)[0], index + 1),
                      [tf.constant(True), tf.constant(0, dtype=tf.int32)])

        score = self.__delta_weights()

        self.__reset_weights()

        return score

    def _compute_influence_value_from_batch(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Computes the influence score (self-influence) for a single batch of training samples.

        Parameters
        ----------
        train_samples
            A tensor with a single batch of training sample.

        Returns
        -------
        influence_values
            The influence score of each sample in the batch train_samples.
        """
        scores = tf.map_fn(self.__compute_single_sample_score, train_samples[:-1][0], parallel_iterations=1)
        scores = - tf.expand_dims(scores, axis=1)

        return scores
