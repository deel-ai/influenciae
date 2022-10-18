# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TODO: Docs
"""
from abc import abstractmethod

import tensorflow as tf

from ..common import InfluenceModel, BaseInfluenceCalculator
from ..common import ExactIHVP, ConjugateGradientDescentIHVP
from ..common import ExactFactory, CGDFactory

from ..influence import FirstOrderInfluenceCalculator
from ..rps import RPSLJE
from ..trac_in import TracIn
from ..rps import RepresenterPointL2

from ..types import Any, Union


class InfluenceCalculatorFactory:
    """
    TODO: Docs
    """

    @abstractmethod
    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              data_train: Any) -> BaseInfluenceCalculator:
        """
        TODO: Docs
        """
        raise NotImplementedError


class FirstOrderFactory(InfluenceCalculatorFactory):
    """
    TODO: Docs
    """

    def __init__(self, ihvp_mode: str, start_layer=-1, dataset_hessian_size=-1, n_cgd_iters=100,
                 feature_extractor: Union[int, tf.keras.Model] = -1):
        self.start_layer = start_layer
        self.ihvp_mode = ihvp_mode
        self.n_cgd_iters = n_cgd_iters
        self.feature_extractor = feature_extractor
        self.dataset_hessian_size = dataset_hessian_size
        assert self.ihvp_mode in ['exact', 'cgd']

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              data_train: Any) -> BaseInfluenceCalculator:
        influence_model = InfluenceModel(model, start_layer=self.start_layer)
        if self.ihvp_mode == 'exact':
            ihvp_calculator = ExactIHVP(influence_model, training_dataset)
        elif self.ihvp_mode == 'cgd':
            # TODO: feature extractor
            ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, self.feature_extractor, training_dataset,
                                                           self.n_cgd_iters)
        else:
            raise Exception("unknown ihvp calculator=" + self.ihvp_mode)

        if self.dataset_hessian_size > 0:
            training_dataset = training_dataset.take(self.dataset_hessian_size)
        influence_calculator = FirstOrderInfluenceCalculator(influence_model, training_dataset, ihvp_calculator)
        return influence_calculator


class RPSLJEFactory(InfluenceCalculatorFactory):
    """
    TODO: Docs
    """

    def __init__(self, ihvp_mode: str, start_layer=-1, dataset_hessian_size=-1, n_cgd_iters=100,
                 feature_extractor: Union[int, tf.keras.Model] = -1):
        self.start_layer = start_layer
        self.ihvp_mode = ihvp_mode
        self.n_cgd_iters = n_cgd_iters
        self.feature_extractor = feature_extractor
        self.dataset_hessian_size = dataset_hessian_size
        assert self.ihvp_mode in ['exact', 'cgd']

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              data_train: Any) -> BaseInfluenceCalculator:
        influence_model = InfluenceModel(model, start_layer=self.start_layer)
        if self.ihvp_mode == 'exact':
            ihvp_calculator_factory = ExactFactory()
        elif self.ihvp_mode == 'cgd':
            # TODO: feature extractor
            ihvp_calculator_factory = CGDFactory(self.feature_extractor, self.n_cgd_iters)
        else:
            raise Exception("unknown ihvp calculator=" + self.ihvp_mode)

        if self.dataset_hessian_size > 0:
            training_dataset = training_dataset.take(self.dataset_hessian_size)
        influence_calculator = RPSLJE(influence_model, training_dataset, ihvp_calculator_factory)
        return influence_calculator


class TracInFactory(InfluenceCalculatorFactory):
    """
    TODO: Docs
    """

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              data_train: Any) -> BaseInfluenceCalculator:
        models = []
        for model_data in data_train[0]:
            influence_model = InfluenceModel(model_data)
            models.append(influence_model)
        influence_calculator = TracIn(models, data_train[1])
        return influence_calculator


class RPSL2Factory(InfluenceCalculatorFactory):
    """
    TODO: Docs
    """

    def __init__(self, lambda_regularization: float, scaling_factor: float = 0.1, layer_index=-2, epochs: int = 100):
        self.lambda_regularization = lambda_regularization
        self.scaling_factor = scaling_factor
        self.epochs = epochs
        self.layer_index = layer_index

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              data_train: Any) -> BaseInfluenceCalculator:
        influence_calculator = RepresenterPointL2(model,
                                                  training_dataset,
                                                  self.lambda_regularization,
                                                  self.scaling_factor,
                                                  self.epochs,
                                                  self.layer_index)
        return influence_calculator
