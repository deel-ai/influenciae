# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Utility classes and functions
"""
from .conjugate_gradients import conjugate_gradients_solve
from .backtracking_line_search import BacktrackingLineSearch
from .tf_operations import (
     find_layer,
     from_layer_name_to_layer_idx,
     is_dataset_batched,
     assert_batched_dataset,
     dataset_size,
     default_process_batch,
     dataset_to_tensor,
     array_to_dataset,
     map_to_device
)
from .sorted_dict import BatchSort, ORDER
from .nearest_neighbors import BaseNearestNeighbors, LinearNearestNeighbors
