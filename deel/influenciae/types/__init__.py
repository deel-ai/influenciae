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


class PyTorchFluentDatasetLike(PyTorchDatasetLike, Protocol):
    """Structural type for PyTorch dataset wrappers exposing fluent transforms."""

    def map(self, map_fn: Callable[[Any], Any]) -> "PyTorchFluentDatasetLike":
        ...

    def cache(self) -> "PyTorchFluentDatasetLike":
        ...

    def zip(self, other: DatasetLike) -> "PyTorchFluentDatasetLike":
        ...

    def batch(self, batch_size: int) -> "PyTorchFluentDatasetLike":
        ...

    def unbatch(self) -> "PyTorchFluentDatasetLike":
        ...

    def shuffle(self, buffer_size: int) -> "PyTorchFluentDatasetLike":
        ...

    def take(self, count: int) -> "PyTorchFluentDatasetLike":
        ...
