from abc import ABC, abstractmethod
import tensorflow as tf

from influenciae.common.model_wrappers import InfluenceModel
from influenciae.common.tf_operations import is_dataset_batched
from influenciae.influence.inverse_hessian_vector_product import (
    InverseHessianVectorProduct,
    ExactIHVP,
    ConjugateGradientDescentIHVP
)

from typing import Optional, Union


ihvp_calc_dict = {'exact': ExactIHVP, 'cgd': ConjugateGradientDescentIHVP}


class BaseInfluenceCalculator(ABC):
    def __init__(
            self,
            model: InfluenceModel,
            dataset: tf.data.Dataset,
            ihvp_calculator: Union[str, InverseHessianVectorProduct] = ExactIHVP,
            n_samples_for_hessian: Optional[int] = None,
            shuffle_buffer_size: Optional[int] = 10000
    ):
        """
        A base class for objets that calculate the different quantities related to the influence functions:
        - Influence function vectors (the actual value of the weights when removing the datapoints in question)
        - Influence values/Cook's distance (a measure of reliance of the model on the individual datapoints)
        - Group's influence function vectors (the actual value of weights when removing an entire group of datapoints)
        - Group's influence values (a measure of reliance of the model on an entire group of points)

        Args:
            model: InfluenceModel
                The TF2.X model implementing the InfluenceModel interface
            dataset: tf.data.Dataset
                A batched TF dataset containing the training dataset over which we will estimate the
                inverse-hessian-vector product
            ihvp_calculator: str or InverseHessianVectorProduct
                Either a string containing the IHVP method (only 'exact' and 'cgd' are supported for the moment) or
                the actual IHVP calculator object
            n_samples_for_hessian: Optional[int]
                Optional. An integer indicating the amount of samples to take from the provided train dataset
            shuffle_buffer_size: Optional[int]
                Optional. An integer indicating the buffer size of the train dataset's shuffle operation (when choosing
                the amount of samples for the hessian)
        """
        self.model = model

        if not is_dataset_batched(dataset):
            raise ValueError("The dataset must be batched before performing this operation.")
        self.train_size = dataset.cardinality().numpy() * dataset._batch_size  # calculate the training set's size
        if n_samples_for_hessian is None:
            dataset_to_estimate_hessian = dataset
        else:
            dataset_to_estimate_hessian = dataset.unbatch().shuffle(shuffle_buffer_size)\
                .take(n_samples_for_hessian).batch(dataset._batch_size)
        self.train_set = dataset_to_estimate_hessian

        if isinstance(ihvp_calculator, str):
            if ihvp_calculator not in ['exact', 'cgd']:
                raise ValueError("Only 'exact' and 'cgd' inverse hessian vector product calculators are supported.")
            self.ihvp_calculator = ihvp_calc_dict[ihvp_calculator](self.model, self.train_set)
        elif isinstance(ihvp_calculator, InverseHessianVectorProduct):
            self.ihvp_calculator = ihvp_calculator

    @abstractmethod
    def compute_influence(self, dataset: tf.data.Dataset) -> tf.Tensor:
        pass

    @abstractmethod
    def compute_influence_values(
            self,
            dataset_train: tf.data.Dataset,
            dataset_to_evaluate: tf.data.Dataset
    ) -> tf.Tensor:
        pass

    @abstractmethod
    def compute_influence_group(
            self,
            group: tf.data.Dataset
    ) -> tf.Tensor:
        pass

    @abstractmethod
    def compute_influence_values_group(
            self,
            group_train: tf.data.Dataset,
            group_to_evaluate: tf.data.Dataset
    ) -> tf.Tensor:
        pass
