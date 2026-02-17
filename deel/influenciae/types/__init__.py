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
        """Return a dataset transformed by ``map_fn``."""
        ...

    def cache(self) -> "PyTorchFluentDatasetLike":
        """Return a cached/materialized dataset wrapper."""
        ...

    def zip(self, other: DatasetLike) -> "PyTorchFluentDatasetLike":
        """Return a dataset zipped with ``other``."""
        ...

    def batch(self, batch_size: int) -> "PyTorchFluentDatasetLike":
        """Return a dataset grouped into batches of ``batch_size``."""
        ...

    def unbatch(self) -> "PyTorchFluentDatasetLike":
        """Return a dataset where batched elements are flattened."""
        ...

    def shuffle(self, buffer_size: int) -> "PyTorchFluentDatasetLike":
        """Return a dataset shuffled with a finite ``buffer_size``."""
        ...

    def take(self, count: int) -> "PyTorchFluentDatasetLike":
        """Return a dataset restricted to the first ``count`` elements."""
        ...
