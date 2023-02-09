# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module defining the interface and classes that implement factories for objects of
class InverseHessianVectorProduct. This will prove itself useful for creating
the objects necessary for computing the different (I)HVPs in second order influence
functions.
"""
from abc import abstractmethod

import tensorflow as tf

from .model_wrappers import InfluenceModel
from .inverse_hessian_vector_product import (
    InverseHessianVectorProduct,
    ExactIHVP,
    ConjugateGradientDescentIHVP,
    LissaIHVP
)

from ..types import Union, Optional


class InverseHessianVectorProductFactory:
    """
    The base interface for InverseHessianVectorProduct factories.
    """
    @abstractmethod
    def build(self, model_influence: InfluenceModel, dataset: tf.data.Dataset) -> InverseHessianVectorProduct:
        """
        Creates an instance of an InverseHessianVectorProduct class with the provided
        parameters.

        Parameters
        ----------
        model_influence
            A TF model implementing the InfluenceModel interface.
        dataset
            A TF dataset object containing the model's (full or partial) training dataset.

        Returns
        -------
        ihvp
            An instance of an InverseHessianVectorProduct class.
        """
        raise NotImplementedError()


class ExactIHVPFactory(InverseHessianVectorProductFactory):
    """
    A factory for instantiating ExactIHVP objects.
    """
    def build(self, model_influence: InfluenceModel, dataset: tf.data.Dataset) -> InverseHessianVectorProduct:
        """
        Creates an instance of the ExactIHVP class for a given model
        implementing the InfluenceModel interface and its (full or partial) training dataset.

        Parameters
        ----------
        model_influence
            A TF model implementing the InfluenceModel interface
        dataset
            A TF dataset containing the model's (full or partial) training dataset.

        Returns
        -------
        exact_ihvp
            An instance of the ExactIHVP class.
        """
        return ExactIHVP(model_influence, dataset)


class CGDIHVPFactory(InverseHessianVectorProductFactory):
    """
    A factory for instantiating ConjugateGradientDescentIHVP objects.

    Attributes
    ----------
    feature_extractor
        Either a TF feature-extractor model or the index of the layer of a whole model
        which will be cut into two for computing the influence vectors and scores.
    n_cgd_iters
        An integer specifying the amount of iterations of the optimizer to run before
        (prematurely) considering the optimization completed.
    extractor_layer
        The cutoff layer for the feature extractor, if specified in TF model format.
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
            assert extractor_layer is not None, "If you provide a model as a feature extractor, you should also" \
                                                "provide the id of the last extracted layer"
            self.extractor_layer = extractor_layer
            self.feature_extractor = feature_extractor

    def build(self, model_influence: InfluenceModel, dataset: tf.data.Dataset) -> InverseHessianVectorProduct:
        """
        Creates an instance of the ConjugateGradientDescentIHVP class for the provided model and its
        corresponding (full or partial) training dataset.

        Parameters
        ----------
        model_influence
            A TF model implementing the InfluenceModel interface.
        dataset
            A TF dataset containing the model's (full or partial) training dataset.

        Returns
        -------
        cgd_ihvp
            An instance of the ConjugateGradientDescentIHVP class
        """
        return ConjugateGradientDescentIHVP(
            model_influence,
            self.extractor_layer,
            dataset,
            self.n_cgd_iters,
            self.feature_extractor,
        )


class LissaIHVPFactory(InverseHessianVectorProductFactory):
    """
    A factory for instantiating LissaIHVP objects.

    Attributes
    ----------
    feature_extractor
        Either a TF feature-extractor model or the index of the layer of a whole model
        which will be cut into two for computing the influence vectors and scores.
    n_cgd_iters
        An integer specifying the amount of iterations of the optimizer to run before
        (prematurely) considering the optimization completed.
    extractor_layer
        The cutoff layer for the feature extractor, if specified in TF model format.
    damping
        A damping parameter to regularize a nearly singular operator.
    scale
        A rescaling factor to verify the hypothesis of norm(operator / scale) < 1.
    """
    def __init__(
        self,
        feature_extractor: Union[int, tf.keras.Model] = -1,
        n_cgd_iters: int = 100,
        extractor_layer: Optional[Union[str, int]] = None,
        damping: float = 1e-4,
        scale: float = 10.
    ):
        self.n_cgd_iters = n_cgd_iters
        self.damping = damping
        self.scale = scale
        if isinstance(feature_extractor, int):
            self.extractor_layer = feature_extractor
            self.feature_extractor = None
        else:
            assert extractor_layer is not None, "If you provide a model as a feature extractor, you should also" \
                                                "provide the id of the last extracted layer"
            self.extractor_layer = extractor_layer
            self.feature_extractor = feature_extractor

    def build(self, model_influence: InfluenceModel, dataset: tf.data.Dataset) -> InverseHessianVectorProduct:
        """
        Creates an instance of the ConjugateGradientDescentIHVP class for the provided model and its
        corresponding (full or partial) training dataset.

        Parameters
        ----------
        model_influence
            A TF model implementing the InfluenceModel interface.
        dataset
            A TF dataset containing the model's (full or partial) training dataset.

        Returns
        -------
        cgd_ihvp
            An instance of the ConjugateGradientDescentIHVP class
        """
        return LissaIHVP(
            model_influence,
            self.extractor_layer,
            dataset,
            self.n_cgd_iters,
            self.feature_extractor,
            self.damping,
            self.scale
        )
