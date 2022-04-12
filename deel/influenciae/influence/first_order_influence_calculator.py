"""
First order Influence module
"""

import tensorflow as tf

from .influence_calculator import BaseInfluenceCalculator

from ..types import Optional
from ..common import assert_batched_dataset


class FirstOrderInfluenceCalculator(BaseInfluenceCalculator):
    """
    A class implementing the necessary methods to compute the different influence quantities
    using the first-order approximation.

    The methods currently implemented are available to evaluate one or a group of point(s):
    - Influence function vectors: the weights difference when removing point(s)
    - Influence values/Cook's distance: a measure of reliance of the model on the individual
      point(s)

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    dataset
        A batched TF dataset containing the training dataset over which we will estimate the
        inverse-hessian-vector product.
    ihvp_calculator
        Either a string containing the IHVP method ('exact' or 'cgd'), an IHVPCalculator
        object or an InverseHessianVectorProduct object.
    n_samples_for_hessian
        An integer indicating the amount of samples to take from the provided train dataset.
    shuffle_buffer_size
        An integer indicating the buffer size of the train dataset's shuffle operation -- when
        choosing the amount of samples for the hessian.
    """

    def compute_influence(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Computes the influence function vector -- an estimation of the weights difference when
        removing point(s) -- one vector for each point.

        Parameters
        ----------
        dataset
            A batched Tensorflow dataset containing the points from which we aim to compute the
            influence of removal.

        Returns
        -------
        influence_vectors
            A tensor containing one vector per input point

        """
        assert_batched_dataset(dataset)

        influence_vectors = self.ihvp_calculator.compute_ihvp(dataset)
        influence_vectors = tf.transpose(influence_vectors)

        return influence_vectors

    def compute_influence_values(
            self,
            dataset_train: tf.data.Dataset,
            dataset_to_evaluate: Optional[tf.data.Dataset] = None
    ) -> tf.Tensor:
        """
        Computes Cook's distance of each point(s) provided individually, giving measure of the
        influence that each point carries on the model's weights.

        The dataset_train contains the points we will be removing and dataset_to_evaluate,
        those with respect to whom we will be measuring the influence.
        As we will be performing the same operation in batches, we consider that each point
        from one dataset corresponds to one from the other. As such, both datasets must contain
        the same amount of points. In case the dataset_to_evaluate is not given, use by default the
        dataset_train: compute the self influence.

        Parameters
        ----------
        dataset_train
            A batched TF dataset containing the points we wish to remove.
        dataset_to_evaluate.
            A batched TF dataset containing the points with respect to whom we wish to measure
            the influence of removing the training points. Default as dataset_train (self
            influence).

        Returns
        -------
        influence_values
            A tensor containing one influence value per pair of input values (one coming from
            each dataset).
        """
        if dataset_to_evaluate is None:
            # default to self influence
            dataset_to_evaluate = dataset_train

        dataset_size = self.assert_compatible_datasets(dataset_train, dataset_to_evaluate)

        grads = self.model.batch_jacobian(dataset_to_evaluate)
        grads = tf.reshape(grads, (dataset_size, -1))

        ihvp = self.ihvp_calculator.compute_ihvp(dataset_train)

        influence_values = tf.reduce_sum(
            tf.math.multiply(grads, tf.transpose(ihvp)), axis=1, keepdims=True)

        return influence_values

    def compute_influence_group(
            self,
            group: tf.data.Dataset
    ) -> tf.Tensor:
        """
        Computes the influence function vector -- an estimation of the weights difference when
        removing the points -- of the whole group of points.

        Parameters
        ----------
        group
            A batched TF dataset containing the group of points of which we wish to compute the
            influence of removal.

        Returns
        -------
        influence_group
            A tensor containing one vector for the whole group.
        """
        assert_batched_dataset(group)

        ihvp = self.ihvp_calculator.compute_ihvp(group)
        reduced_ihvp = tf.reduce_sum(ihvp, axis=1)

        influence_group = tf.reshape(reduced_ihvp, (1, -1))

        return influence_group

    def compute_influence_values_group(
            self,
            group_train: tf.data.Dataset,
            group_to_evaluate: Optional[tf.data.Dataset] = None
    ) -> tf.Tensor:
        """
        Computes Cook's distance of the whole group of points provided, giving measure of the
        influence that the group carries on the model's weights.

        The dataset_train contains the points we will be removing and dataset_to_evaluate,
        those with respect to whom we will be measuring the influence. As we will be performing
        the same operation in batches, we consider that each point from one dataset corresponds
        to one from the other. As such, both datasets must contain the same amount of points.
        In case the group_to_evaluate is not given, use by default the
        group_to_train: compute the self influence of the group.


        Parameters
        ----------
        group_train
            A batched TF dataset containing the group of points we wish to remove.
        group_to_evaluate
            A batched TF dataset containing the group of points with respect to whom we wish to
            measure the influence of removing the training points.

        Returns
        -------
        influence_values_group
            A tensor containing one influence value for the whole group.
        """
        if group_to_evaluate is None:
            # default to self influence
            group_to_evaluate = group_train

        dataset_size = self.assert_compatible_datasets(group_train, group_to_evaluate)

        reduced_grads = tf.reduce_sum(tf.reshape(self.model.batch_jacobian(group_to_evaluate),
                                      (dataset_size, -1)), axis=0, keepdims=True)

        reduced_ihvp = tf.reduce_sum(self.ihvp_calculator.compute_ihvp(group_train), axis=1, keepdims=True)

        influence_values_group = tf.matmul(reduced_grads, reduced_ihvp)

        return influence_values_group
