# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
First order Influence module implementing computations for all the different influence
related quantities: influence vector (delta of the weights after holding out the
sample and the original model's), Cook's distance (or influence values, a measure of
the model's reliance on the specific sample), both for individual points and whole
groups of data-points.

Disclaimer: This implements only a first order approximation of the influence function,
which does not take into account the pairwise interactions of data-points inside groups.
For a more precise (but much more computationally expensive) alternative, please refer
to the SecondOrderInfluenceCalculator module.
"""
import tensorflow as tf

from .base_group_influence import BaseGroupInfluenceCalculator

from ..common import InfluenceModel
from ..common import BaseInfluenceCalculator
from ..common import InverseHessianVectorProduct, IHVPCalculator

from ..types import Optional, Union, Tuple
from ..utils import assert_batched_dataset


class FirstOrderInfluenceCalculator(BaseInfluenceCalculator, BaseGroupInfluenceCalculator):
    """
    A class implementing the necessary methods to compute the different influence quantities
    using a first-order approximation. This makes it ideal for individual points and small
    groups of data, as it does so (relatively) efficiently.

    Notes
    -----
    For estimating the influence of large groups of data, please refer to the
    SecondOrderInfluenceCalculator class, which also takes into account the pairwise interactions
    between the points inside these groups.

    The methods currently implemented are available to evaluate one or a group of point(s):
    - Influence function vectors: the weights difference when removing points or groups of points
    - Influence values/Cook's distance: a measure of reliance of the model on the individual
      points or groups of points.

    For individual points, the following paper is used:
    [https://arxiv.org/abs/1703.04730](https://arxiv.org/abs/1703.04730).
    For groups of points, the following paper is used:
    [https://arxiv.org/abs/1905.13289](https://arxiv.org/abs/1905.13289)

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
    normalize
        Implement "RelatIF: Identifying Explanatory Training Examples via Relative Influence"
        https://arxiv.org/pdf/2003.11630.pdf
        if True, compute the relative influence by normalizing the influence function.
    """

    def __init__(
            self,
            model: InfluenceModel,
            dataset: tf.data.Dataset,
            ihvp_calculator: Union[str, InverseHessianVectorProduct, IHVPCalculator] = 'exact',
            n_samples_for_hessian: Optional[int] = None,
            shuffle_buffer_size: Optional[int] = 10000,
            normalize=False
    ):
        super().__init__(
            model,
            dataset,
            ihvp_calculator,
            n_samples_for_hessian,
            shuffle_buffer_size
        )

        self.normalize = normalize

    @tf.function
    def _normalize_if_needed(self, v):
        """
        Normalize the input vector if the normalize property is True. If False, do nothing

        Parameters
        ----------
        v
            The vector to normalize of shape [Features_Space, Batch_Size]

        Returns
        -------
        v
            The normalized vector if the normalize property is True, otherwise the input vector
        """
        if self.normalize:
            v = v / tf.norm(v, axis=0, keepdims=True)
        return v

    @tf.function
    def _compute_influence_vector(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Computes the influence vector (i.e. the delta of model's weights after a perturbation on the training
        dataset) for a single batch of training samples.

        Parameters
        ----------
        train_samples
            A tuple with the batch of training samples (with their labels).

        Returns
        -------
        influence_vector
            A tensor with the influence vector for each individual point. Shape will be (batch_size, nb_weights).
        """
        influence_vector = self.ihvp_calculator._compute_ihvp_single_batch(train_samples)  # pylint: disable=W0212
        influence_vector = self._normalize_if_needed(influence_vector)
        influence_vector = tf.transpose(influence_vector)
        return influence_vector

    @tf.function
    def _preprocess_samples(self, samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Preprocess a sample to evaluate

        Parameters
        ----------
        samples
            sample to evaluate

        Returns
        -------
        The preprocessed sample to evaluate
        """
        sample_evaluate_grads = self.model.batch_jacobian_tensor(samples)
        return sample_evaluate_grads

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

    @tf.function
    def _estimate_influence_value_from_influence_vector(self, preproc_test_sample: tf.Tensor,
                                                        influence_vector: tf.Tensor) -> tf.Tensor:
        """
        Estimates the influence score of leaving out the influence vector corresponding to a given training
        data-point on a test sample that has already been pre-processed.

        Parameters
        ----------
        preproc_test_sample
            A single pre-processed test sample we wish to evaluate.
        influence_vector
            A single influence vector corresponding to a data-point from the training dataset.

        Returns
        -------
        influence_values
            A tensor with the resulting influence value.
        """
        influence_values = tf.matmul(preproc_test_sample, tf.transpose(influence_vector))
        return influence_values

    @tf.function
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
        batched_inf_vect = self._compute_influence_vector(train_samples)
        evaluate_vect = self._preprocess_samples(train_samples)
        influence_values = tf.reduce_sum(
            tf.math.multiply(evaluate_vect, batched_inf_vect), axis=1, keepdims=True)
        #TODO: improve IHVP to not compute 2 times the gradient
        return influence_values

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

        ihvp_ds = self.ihvp_calculator.compute_ihvp(group)
        reduced_ihvp = ihvp_ds.map(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))
        reduced_ihvp = reduced_ihvp.reduce(tf.constant(0, dtype=ihvp_ds.element_spec.dtype), lambda x, y: x + y)

        reduced_ihvp = self._normalize_if_needed(reduced_ihvp)

        influence_group = tf.reshape(reduced_ihvp, (1, -1))

        return influence_group

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
        if group_to_evaluate is None:
            # default to self influence
            group_to_evaluate = group_train

        ds_size = self.assert_compatible_datasets(group_train, group_to_evaluate)

        reduced_grads = tf.reduce_sum(tf.reshape(self.model.batch_jacobian(group_to_evaluate),
                                                 (ds_size, -1)), axis=0, keepdims=True)

        ihvp_ds = self.ihvp_calculator.compute_ihvp(group_train)
        reduced_ihvp = ihvp_ds.map(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))
        reduced_ihvp = reduced_ihvp.reduce(tf.constant(0, dtype=ihvp_ds.element_spec.dtype), lambda x, y: x + y)

        reduced_ihvp = self._normalize_if_needed(reduced_ihvp)

        influence_values_group = tf.matmul(reduced_grads, reduced_ihvp)

        return influence_values_group
