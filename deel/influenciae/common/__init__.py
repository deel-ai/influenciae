# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Common classes and methods
"""

from .model_wrappers import BaseInfluenceModel, InfluenceModel
from .base_influence import SelfInfluenceCalculator, BaseInfluenceCalculator, CACHE
from .inverse_hessian_vector_product import InverseHessianVectorProduct, ExactIHVP, ConjugateGradientDescentIHVP, \
     IHVPCalculator, LissaIHVP, ForwardOverBackwardHVP
from .ihvp_factory import InverseHessianVectorProductFactory, ExactIHVPFactory, CGDIHVPFactory, LissaIHVPFactory
