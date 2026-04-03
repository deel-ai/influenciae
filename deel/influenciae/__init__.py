# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Influenciae
-------

The goal of Influenciae is to provide a simple interface to the various influence functions
techniques. This library supports both TensorFlow and PyTorch backends.

Installation
------------
- Base installation (no backend): pip install influenciae
- With TensorFlow: pip install influenciae[tensorflow]
- With PyTorch: pip install influenciae[pytorch]
- With both backends: pip install influenciae[all]
"""
import importlib
import sys
from typing import TYPE_CHECKING

__version__ = '0.3.0'

# Lazy module loading to avoid ImportError when backends are not installed
# Submodules are only loaded when actually accessed

_SUBMODULES = ['influence', 'common', 'rps', 'trac_in', 'benchmark', 'plots']

def __getattr__(name):
    """Lazy import of submodules."""
    if name in _SUBMODULES:
        try:
            module = importlib.import_module(f'.{name}', __name__)
            globals()[name] = module
            return module
        except ImportError as e:
            raise ImportError(
                f"Could not import '{name}' submodule. "
                f"Make sure you have the required backend installed. "
                f"Install with: pip install influenciae[tensorflow] or pip install influenciae[pytorch]. "
                f"Original error: {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes including lazy submodules."""
    return list(globals().keys()) + _SUBMODULES


# For type checking, import submodules statically
if TYPE_CHECKING:
    from . import influence
    from . import common
    from . import rps
    from . import trac_in
    from . import benchmark
    from . import plots
