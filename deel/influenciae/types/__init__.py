# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Typing module
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, Union


class TensorflowDatasetLike(Protocol):
    """Structural type compatible with ``tf.data.Dataset``."""

    def __iter__(self) -> Iterator[Any]:
        ...


class PyTorchDatasetLike(Protocol):
    """Structural type compatible with PyTorch dataset/data loader objects."""

    def __iter__(self) -> Iterator[Any]:
        ...


DatasetLike = Union[TensorflowDatasetLike, PyTorchDatasetLike]
