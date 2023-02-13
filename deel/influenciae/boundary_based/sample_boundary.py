# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a data-point-influence estimation method based on how intensely
each sample must be adversarially attacked to make it change its prediction. Intuitively,
we would expect outliers and atypical examples to need a lesser attack for the model to
misclassify it.

Unlike other influence calculators, this one cannot be used to estimate
the influence of a point on another.

The adversarial attacks are performed via deep fool.
"""
import tensorflow as tf
from tensorflow.keras import Model  # pylint:  disable=E0611

from ..common import SelfInfluenceCalculator
from ..types import Tuple


class SampleBoundaryCalculator(SelfInfluenceCalculator):
    """
    A class implementing an influence score based on the distance of a sample to the
    boundary of the classifier.
    The distance to the boundary is estimated using the deep fool method.
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
    eps
        Difference between two logits to assume that they have the same values
    """

    def __init__(self, model: Model, step_nbr: int = 100, eps: float = 1E-6):
        self.weights_init = [tf.identity(w) for w in model.trainable_variables]
        self.model = model

        self.step_nbr = step_nbr
        self.eps = eps

    @staticmethod
    def __delta_to_index(indexes_1: tf.Tensor, indexes_2: tf.Tensor, x: tf.Tensor):
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
    def __step(self, x: tf.Tensor, y_pred: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        The optimization step to find the distance between the boundary and a given sample x.

        Notes
        -----
        To see more details about the optimization procedure for multi-class classifier, please
        refer to [https://arxiv.org/abs/1511.04599]

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
        x_new
            The sample updated by the optimization procedure
        """
        y_pred = tf.argmax(y_pred, axis=1)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            y = self.model(x)

        y_computed = tf.argmax(y, axis=1)

        def update_grads():
            jac = tape.jacobian(y, x)

            indexes_all = tf.repeat(tf.expand_dims(tf.range(0, tf.shape(y)[1]), axis=0), tf.shape(y)[0], axis=0)
            indexes_class = tf.cast(tf.repeat(tf.expand_dims(y_pred, axis=1), tf.shape(y)[1], axis=1), dtype=tf.int32)
            indexes_other = tf.reshape(indexes_all[indexes_all != indexes_class], (-1, tf.shape(y)[1] - 1))

            delta_y = self.__delta_to_index(indexes_other, y_pred, y)
            delta_y = tf.abs(tf.reduce_mean(delta_y, axis=0))

            jac_delta = tf.reduce_mean(self.__delta_to_index(indexes_other, y_pred, jac), axis=0)

            jac_norm = tf.reshape(jac_delta, (tf.shape(jac_delta)[0], -1))
            jac_norm = tf.norm(jac_norm, axis=1)

            coeff = delta_y / jac_norm

            best_class = tf.argmin(coeff, axis=0)

            loss = (coeff / jac_norm)[best_class]

            x_new = x + loss * jac_delta[best_class]

            return loss, x_new

        computation = tf.reduce_any(y_computed == y_pred)

        top_k, _ = tf.math.top_k(tf.squeeze(y, axis=0), k=2)
        enough_close = tf.abs(top_k[0] - top_k[1]) > self.eps

        computation = tf.logical_and(computation, enough_close)

        loss_value, x_updated = tf.cond(computation, update_grads, lambda: (tf.constant(0.0, dtype=x.dtype), x))

        return computation, loss_value, x_updated

    def __compute_single_sample_score(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the influence score (self-influence) for a single training sample.

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

        def body(index, x_current):
            computation, _, x_new = self.__step(x_current, y_pred)
            return computation, index + 1, x_new

        _, _, x_adversarial = tf.while_loop(
            lambda cond, index, x_current: tf.logical_and(cond, index < self.step_nbr),
            lambda cond, index, x_current: body(index, x_current),
            [tf.constant(True), tf.constant(0, dtype=tf.int32), x])

        score = tf.norm(x - x_adversarial)

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
