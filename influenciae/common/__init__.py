"""
Utility classes and functions
"""

from .tf_operations import find_layer, is_dataset_batched, assert_batched_dataset, dataset_size
from .model_wrappers import InfluenceModel
from .conjugate_gradients import conjugate_gradients_solve
