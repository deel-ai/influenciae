# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing the necessary functions for computing influence functions on
groups of data -- i.e. performing approximate hold-one-out of groups of data-points
as a whole instead of individually. The theoretical validity of this implementation
was discussed in https://arxiv.org/abs/1905.13289 , with positive results.

We propose two classes that implement this interface: the FirstOrderInfluenceCalculator
(using the technique proposed in https://arxiv.org/abs/1905.13289) and the
SecondOrderInfluenceCalculator, that is only available for calculating groups'
influence and does so through the method presented in
https://arxiv.org/abs/1911.00418
"""

from abc import abstractmethod
import tensorflow as tf

from ..common import InfluenceModel
from ..common import InverseHessianVectorProduct, IHVPCalculator, ExactIHVP

from ..utils import dataset_size
from ..types import Optional, Union


class BaseGroupInfluenceCalculator:
    """
    A base class for objects that calculate the different quantities related to influence
    functions for whole groups of data-points.

    Notes
    -----
    The methods currently implemented are available to evaluate groups of points:
    - Influence function vectors: the weights difference when removing entire groups of points.
    - Influence values/Cook's distance: a measure of reliance of the model on the entire group.

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
            ihvp_calculator: Union[str, InverseHessianVectorProduct, IHVPCalculator] = ExactIHVP,
            n_samples_for_hessian: Optional[int] = None,
            shuffle_buffer_size: Optional[int] = 10000
    ):
        self.model = model

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

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @staticmethod
    def assert_compatible_datasets(dataset_a: tf.data.Dataset, dataset_b: tf.data.Dataset) -> int:
        """
        Assert that the datasets are compatible: that they contain the same number of points. Else,
        throw an error.

        Parameters
        ----------
        dataset_a
            First batched tensorflow dataset to check.
        dataset_b
            Second batched tensorflow dataset to check.

        Returns
        -------
        size
            The size of the dataset.
        """
        size_a = dataset_size(dataset_a)
        size_b = dataset_size(dataset_b)

        if size_a != size_b:
            raise ValueError("The amount of points in the train and evaluation groups must match.")

        return size_a
