# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TODO: Docs
"""
from abc import abstractmethod

import tensorflow as tf

from .model_wrappers import InfluenceModel
from .inverse_hessian_vector_product import InverseHessianVectorProduct, ExactIHVP, ConjugateGradientDescentIHVP

from ..types import Union, Optional

class InverseHessianVectorProductFactory:
    """
    TODO: Docs
    """
    @abstractmethod
    def build(self, model_influence: InfluenceModel, dataset: tf.data.Dataset) -> InverseHessianVectorProduct:
        """
        TODO: Docs
        """
        raise NotImplementedError()

class ExactFactory(InverseHessianVectorProductFactory):
    """
    TODO: Docs
    """
    def build(self, model_influence: InfluenceModel, dataset: tf.data.Dataset) -> InverseHessianVectorProduct:
        """
        TODO: Docs
        """
        return ExactIHVP(model_influence, dataset)


class CGDFactory(InverseHessianVectorProductFactory):
    """
    TODO: Docs
    """
    def __init__(
        self,
        feature_extractor: Union[int, tf.keras.Model] = -1,
        n_cgd_iters: int = 100,
        extractor_layer: Optional[Union[str, int]] = None
    ):
        self.n_cgd_iters = n_cgd_iters
        if isinstance(feature_extractor, int):
            self.extractor_layer = feature_extractor
            self.feature_extractor = None
        else:
            assert extractor_layer is not None, "If you provide a model as a feature extractor you should provide the \
                id of the last extracted layer"
            self.extractor_layer = extractor_layer
            self.feature_extractor = feature_extractor

    def build(self, model_influence: InfluenceModel, dataset: tf.data.Dataset) -> InverseHessianVectorProduct:
        """
        TODO: Docs
        """
        return ConjugateGradientDescentIHVP(
            model_influence,
            self.extractor_layer,
            dataset,
            self.n_cgd_iters,
            self.feature_extractor,
        )
