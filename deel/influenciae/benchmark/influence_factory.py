# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from abc import abstractmethod

import tensorflow as tf

from ..common import InfluenceModel, BaseInfluenceCalculator
from ..common import ExactIHVP, ConjugateGradientDescentIHVP

from ..influence import FirstOrderInfluenceCalculator
from ..rps import RPSLJE

from ..types import Any

class InfluenceCalculatorFactory:
    """
    TODO: Docs
    """
    @abstractmethod
    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model, data_train: Any) -> BaseInfluenceCalculator:
        raise NotImplementedError

class FirstOrderFactory(InfluenceCalculatorFactory):

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model, data_train) -> BaseInfluenceCalculator:
        influence_model = InfluenceModel(model, start_layer=-1)
        # ihvp_calculator = ExactIHVP(influence_model, training_dataset)
        ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, -1, training_dataset)
        influence_calculator = FirstOrderInfluenceCalculator(influence_model, training_dataset, ihvp_calculator)
        return influence_calculator

class RPSLJEFactory(InfluenceCalculatorFactory):

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model, data_train) -> BaseInfluenceCalculator:
        influence_model = InfluenceModel(model)
        ihvp_calculator = ExactIHVP(influence_model, training_dataset)
        influence_calculator = RPSLJE(
            influence_model,
            ihvp_calculator,
            target_layer=-1)
        return influence_calculator
