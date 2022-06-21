# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Second order Influence module
"""

import tensorflow as tf

from .influence_calculator import BaseInfluenceCalculator, ExactIHVP, ConjugateGradientDescentIHVP

from ..common.tf_operations import assert_batched_dataset, dataset_size
from ..types import Optional


class SecondOrderInfluenceCalculator(BaseInfluenceCalculator):
    """
    A class implementing the necessary methods to compute the different influence quantities
    using the second-order approximation.

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
        raise NotImplementedError('Second order influence functions are not available for single points.')

    def compute_influence_values(
            self,
            dataset_train: tf.data.Dataset,
            dataset_to_evaluate: Optional[tf.data.Dataset] = None
    ) -> tf.Tensor:
        raise NotImplementedError('Second order influence functions are not available for single points.')

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
        fraction = tf.cast(dataset_size(group), dtype=tf.float32) / tf.cast(self.train_size, dtype=tf.float32)
        coeff_additive_term = (1. - 2 * fraction) / \
                              (tf.square(1. - fraction) * tf.cast(self.train_size, dtype=tf.float32))
        coeff_pairwise_term = 1. / tf.square((1. - fraction) * tf.cast(self.train_size, dtype=tf.float32))
        influence_group = coeff_additive_term * self._compute_additive_term(group) + \
                          coeff_pairwise_term * self._compute_pairwise_interactions(group)
        influence_group = tf.transpose(influence_group)  # to get the right shape for the output

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
        ds_size = self.assert_compatible_datasets(group_train, group_to_evaluate)
        influence = tf.transpose(self.compute_influence_group(group_train))
        reduced_grads = tf.reduce_sum(tf.reshape(self.model.batch_jacobian(group_to_evaluate),
                                                 (ds_size, -1)), axis=0, keepdims=True)

        return tf.matmul(reduced_grads, influence)

    def _compute_additive_term(self, dataset: tf.data.Dataset):
        """
        Computes the additive term as per Basu et al.'s article. It accounts for the influence of each of the
        points that we wish to remove, without taking into account the interactions between each other

        Parameters
        ----------
            dataset
                A batched TF dataset containing the points we wish to remove

        Returns
        -------
        interactions
            A tensor containing the addition of the influence of the points in the group
        """
        return tf.reduce_sum(self.ihvp_calculator.compute_ihvp(dataset), axis=1, keepdims=True)

    def _compute_pairwise_interactions(self, dataset: tf.data.Dataset):
        """
        Computes the term corresponding to the pairwise interactions as per Basu et al.'s article. It will
        contain all the interactions between each of the points with each of the other points.

        Disclaimer: this term can be quite computationally intensive to calculate, as there are 3 nested hessian-vector
        products. Thus, it can also be a source of numerical instability when using an approximate version of an
        IHVP calculator.

       Parameters
        ----------
            dataset
                A batched TF dataset containing the points we wish to remove

        Returns
        -------
        interactions
            A tensor containing the sum of all the interactions of each point we are removing with each other point
             of the group
        """
        local_ihvp = ExactIHVP(self.model, dataset) if isinstance(self.ihvp_calculator, ExactIHVP) \
            else ConjugateGradientDescentIHVP(self.model, dataset)

        interactions = self.ihvp_calculator.compute_ihvp(
            tf.data.Dataset.from_tensors(
                tf.squeeze(
                    local_ihvp.compute_hvp(
                        tf.data.Dataset.from_tensors(
                            tf.reduce_sum(self.ihvp_calculator.compute_ihvp(dataset), axis=1)
                        ).batch(1),
                        use_gradient=False),
                    axis=-1)
                ).batch(1), use_gradient=False
        )
        interactions *= tf.cast(dataset_size(dataset), dtype=tf.float32)
        interactions = tf.expand_dims(interactions, axis=1) if len(interactions.shape) == 1 else interactions

        return interactions
