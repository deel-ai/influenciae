# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Utility classes and functions.

This module uses lazy imports to avoid requiring TensorFlow/PyTorch
when only basic utilities are needed, and to avoid circular import issues.
"""
import importlib
from typing import TYPE_CHECKING


# Lazy imports for framework-specific utilities
_LAZY_IMPORTS = {
    # Utilities that cause circular imports
    'BatchSort': '.sorted_dict',
    'ORDER': '.sorted_dict',
    'BaseNearestNeighbors': '.nearest_neighbors',
    'LinearNearestNeighbors': '.nearest_neighbors',
    # Conjugate gradients (imports from common, which imports from utils - circular)
    'conjugate_gradients_solve': '.conjugate_gradients',
    # TensorFlow operations
    'find_layer': '.tf_operations',
    'from_layer_name_to_layer_idx': '.tf_operations',
    'is_dataset_batched': '.tf_operations',
    'assert_batched_dataset': '.tf_operations',
    'dataset_size': '.tf_operations',
    'default_process_batch': '.tf_operations',
    'dataset_to_tensor': '.tf_operations',
    'array_to_dataset': '.tf_operations',
    'map_to_device': '.tf_operations',
    'split_model': '.tf_operations',
    # Backtracking line search (requires backends)
    'BacktrackingLineSearch': '.backtracking_line_search',
    'BacktrackingLineSearchPyTorch': '.backtracking_line_search',
}

def __getattr__(name):
    """Lazy import of framework-specific utilities."""
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        try:
            module = importlib.import_module(module_name, __name__)
            attr = getattr(module, name)
            globals()[name] = attr
            return attr
        except ImportError as e:
            raise ImportError(
                f"Could not import '{name}' from utils. "
                f"Make sure you have the required backend installed. "
                f"Original error: {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes including lazy imports."""
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())


# For type checking, import everything statically
if TYPE_CHECKING:
    from .backtracking_line_search import BacktrackingLineSearch, BacktrackingLineSearchPyTorch
    from .tf_operations import (
        find_layer,
        from_layer_name_to_layer_idx,
        is_dataset_batched,
        assert_batched_dataset,
        dataset_size,
        default_process_batch,
        dataset_to_tensor,
        array_to_dataset,
        map_to_device,
        split_model,
    )
