# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Utility classes and functions
"""

from .tf_operations import find_layer, is_dataset_batched, assert_batched_dataset, dataset_size
from .model_wrappers import InfluenceModel
from .conjugate_gradients import conjugate_gradients_solve
from .compute_hessian_block import compute_hessian_block
from .backtracking_line_search import BacktrackingLineSearch
