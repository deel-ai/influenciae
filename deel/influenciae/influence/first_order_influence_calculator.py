# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
First order Influence module
"""

import tensorflow as tf

from ..common import InfluenceModel
from ..common import VectorBasedInfluenceCalculator
from ..common import InverseHessianVectorProduct, IHVPCalculator

from ..types import Optional, Union, Tuple
from ..utils import assert_batched_dataset, dataset_size


class FirstOrderInfluenceCalculator(VectorBasedInfluenceCalculator):
    """
    TODO: Improve documentation to be more specific
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
        self.model = model

        self.train_size = dataset_size(dataset)

        if n_samples_for_hessian is None:
            dataset_to_estimate_hessian = dataset
        else:
            dataset_to_estimate_hessian = dataset.unbatch().shuffle(shuffle_buffer_size)\
                .take(n_samples_for_hessian).batch(dataset._batch_size)

        self.train_set = dataset_to_estimate_hessian

        # load ivhp calculator from str, IHVPcalculator enum or InverseHessianVectorProduct object
        if isinstance(ihvp_calculator, str):
            self.ihvp_calculator = IHVPCalculator.from_string(ihvp_calculator).value(self.model,
                                                                                     self.train_set)
        elif isinstance(ihvp_calculator, IHVPCalculator):
            self.ihvp_calculator = ihvp_calculator.value(self.model, self.train_set)
        elif isinstance(ihvp_calculator, InverseHessianVectorProduct):
            self.ihvp_calculator = ihvp_calculator
        else:
            raise AttributeError("ihvp_calculator should belong to ['str, IHVPCalculator', 'InverseHessianVectorProduct']")

        self.normalize = normalize

    def __normalize_if_needed(self, v):
        """
        Normalize the input vector if the normalize property is True. If False, do nothing

        Parameters
        ----------
        v:
            The vector to normalize of shape [Features_Space, Batch_Size]

        Returns
        -------
        v:
            The normalized vector if the normalize property is True, otherwise the input vector
        """
        if self.normalize:
            v = v / tf.norm(v, axis=0, keepdims=True)
        return v

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
        #TODO: is it train samples or train_y, train_target ? Should be consistent across the different API
        #TODO: should return (batch, nb_params) or (nb_params, batch) ?
        """
        influence_vector = self.ihvp_calculator.compute_ihvp_single_batch(train_samples)
        influence_vector = self.__normalize_if_needed(influence_vector)
        influence_vector = tf.transpose(influence_vector) #TODO: ensure it is what we want
        return influence_vector

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
        batch_data, batch_label = samples_to_evaluate
        sample_evaluate_grads = self.model.batch_jacobian_tensor(batch_data, batch_label)
        return sample_evaluate_grads


    def compute_influence_value_from_influence_vector(self, preproc_sample_to_evaluate,
                                                      influence_vector: tf.Tensor) -> tf.Tensor:
        """
        Compute the influence score for a preprocessed sample to evaluate and a training influence vector

        Parameters
        ----------
        preproc_sample_to_evaluate
            Preprocessed sample to evaluate
        influence_vector
            Training influence Vector
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
        batched_inf_vect = self.compute_influence_vector(train_samples)
        evaluate_vect = self.preprocess_sample_to_evaluate(train_samples)
        influence_values = tf.reduce_sum(
            tf.math.multiply(evaluate_vect, batched_inf_vect), axis=1, keepdims=True)
        #TODO: improve IHVP to not compute 2 times the gradient
        #TODO: Attention au normalize
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

        reduced_ihvp = self.__normalize_if_needed(reduced_ihvp)

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

        reduced_ihvp = self.__normalize_if_needed(reduced_ihvp)

        influence_values_group = tf.matmul(reduced_grads, reduced_ihvp)

        return influence_values_group
