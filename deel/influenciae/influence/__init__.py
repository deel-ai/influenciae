# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Influence function module
"""
from .first_order_influence_calculator import FirstOrderInfluenceCalculator
from .inverse_hessian_vector_product import ExactIHVP, ConjugateGradientDescentIHVP