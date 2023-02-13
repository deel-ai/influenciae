# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a second order approximation for groups of data-points, where
pairwise interactions of how holding out one sample affects another in the group
that's being held out are now taken into account. This method was originally
presented in https://arxiv.org/abs/1911.00418 and it is supposed to greatly improve
the accuracy of the influence estimations for big groups of data.

Disclaimer: this method can be very computationally expensive, especially when calculating
the influence for a large number of weights.
"""
import tensorflow as tf

from .base_group_influence import BaseGroupInfluenceCalculator
from ..common import ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP
from ..common import InfluenceModel
from ..common import InverseHessianVectorProduct, IHVPCalculator

from ..utils import assert_batched_dataset, dataset_size
from ..types import Optional, Union


class SecondOrderInfluenceCalculator(BaseGroupInfluenceCalculator):
    """
    A class implementing the necessary methods to compute the different influence quantities
    (only for groups) using a second-order approximation, thus allowing us to take into
    account the pairwise interactions between points inside the group. For small groups of
    points, consider using the first order alternative if the computational cost is
    too high.

    Notes
    -----
    The methods currently implemented are available to evaluate groups of points:
    - Influence function vectors: the weights difference when removing groups of points.
    - Influence values/Cook's distance: a measure of reliance of the model on the group of points.

    This implementation is based on the following paper:
    [https://arxiv.org/abs/1911.00418](https://arxiv.org/abs/1911.00418)

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
    def __init__(
            self,
            model: InfluenceModel,
            dataset: tf.data.Dataset,
            ihvp_calculator: Union[str, InverseHessianVectorProduct, IHVPCalculator] = 'exact',
            n_samples_for_hessian: Optional[int] = None,
            shuffle_buffer_size: Optional[int] = 10000
    ):

        super().__init__(
            model,
            dataset,
            ihvp_calculator,
            n_samples_for_hessian,
            shuffle_buffer_size
        )

        self.train_size = dataset_size(dataset)

    def compute_influence_vector_group(
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

        additive = coeff_additive_term * self._compute_additive_term(group)
        pairwise = coeff_pairwise_term * self._compute_pairwise_interactions(group)

        influence_group = additive + pairwise
        influence_group = tf.transpose(influence_group)  # to get the right shape for the output

        return influence_group

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
        ihvp_ds = self.ihvp_calculator.compute_ihvp(dataset)
        reduced_ihvp = ihvp_ds.map(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))
        return reduced_ihvp.reduce(tf.constant(0, dtype=ihvp_ds.element_spec.dtype), lambda x, y: x + y)

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
        if isinstance(self.ihvp_calculator, ExactIHVP):
            local_ihvp = ExactIHVP(self.model, dataset)
        elif isinstance(self.ihvp_calculator, ConjugateGradientDescentIHVP):
            local_ihvp = ConjugateGradientDescentIHVP(self.model, self.ihvp_calculator.extractor_layer,
                                                      dataset, self.ihvp_calculator.n_opt_iters,
                                                      self.ihvp_calculator.feature_extractor)
        else:
            local_ihvp = LissaIHVP(self.model, self.ihvp_calculator.extractor_layer,
                                   dataset, self.ihvp_calculator.n_opt_iters,
                                   self.ihvp_calculator.feature_extractor)

        ihvp_ds = self.ihvp_calculator.compute_ihvp(dataset)
        ihvp_ds = ihvp_ds.map(lambda x: tf.reduce_sum(x, axis=1))

        reduced_ihvp = ihvp_ds.reduce(tf.constant(0, dtype=ihvp_ds.element_spec.dtype), lambda x, y: x + y)
        reduced_ihvp_ds = tf.data.Dataset.from_tensors(reduced_ihvp).batch(dataset._batch_size)  # pylint: disable=W0212

        local_hvp = local_ihvp.compute_hvp(reduced_ihvp_ds, use_gradient=False).batch(
            dataset._batch_size)  # pylint: disable=W0212

        interactions = self.ihvp_calculator.compute_ihvp(
            local_hvp, use_gradient=False
        )

        ds_size = tf.cast(dataset_size(dataset), dtype=interactions.element_spec.dtype)
        interactions = interactions.map(lambda x: x * ds_size)

        return interactions.get_single_element()

    def estimate_influence_values_group(
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
        influence = tf.transpose(self.compute_influence_vector_group(group_train))
        reduced_grads = tf.reduce_sum(tf.reshape(self.model.batch_jacobian(group_to_evaluate),
                                                 (ds_size, -1)), axis=0, keepdims=True)

        return tf.matmul(reduced_grads, influence)
