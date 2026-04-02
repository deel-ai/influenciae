# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Typing module.
"""

from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Protocol, TypeAlias, Union

if TYPE_CHECKING:
    import numpy as np

    from ..common.backend import BaseBackend, Framework


class DatasetLike(Protocol):
    """Structural type compatible with tf.data.Dataset and PyTorch data loaders."""

    def __iter__(self) -> Iterator[Any]:
        ...


class Tensor(Protocol):
    """Structural type for framework tensors."""

    @property
    def shape(self) -> Any:
        """Return the tensor shape metadata."""
        raise NotImplementedError

    @property
    def dtype(self) -> Any:
        """Return the tensor dtype metadata."""
        raise NotImplementedError

    def __getitem__(self, key: Any) -> "Tensor":
        ...

    def __add__(self, other: Any) -> "Tensor":
        ...

    def __radd__(self, other: Any) -> "Tensor":
        ...

    def __sub__(self, other: Any) -> "Tensor":
        ...

    def __rsub__(self, other: Any) -> "Tensor":
        ...

    def __mul__(self, other: Any) -> "Tensor":
        ...

    def __rmul__(self, other: Any) -> "Tensor":
        ...

    def __truediv__(self, other: Any) -> "Tensor":
        ...

    def __rtruediv__(self, other: Any) -> "Tensor":
        ...

    def __neg__(self) -> "Tensor":
        ...

    def __lt__(self, other: Any) -> "Tensor":
        ...

    def __le__(self, other: Any) -> "Tensor":
        ...

    def __gt__(self, other: Any) -> "Tensor":
        ...

    def __ge__(self, other: Any) -> "Tensor":
        ...

    def item(self) -> Union[int, float, bool]:
        """Convert a scalar tensor to its Python value."""
        raise NotImplementedError

    def numpy(self) -> "np.ndarray":
        """Return a NumPy view or copy of the tensor."""
        raise NotImplementedError


class Model(Protocol):
    """Structural type for framework models."""

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        ...


Layer: TypeAlias = Any


class WeightVariable(Tensor, Protocol):
    """Structural type for trainable weight variables."""


DType: TypeAlias = Any
ElementSpec: TypeAlias = Any
LossFunction: TypeAlias = Callable[..., Tensor]
BackendLike: TypeAlias = Union["BaseBackend", "Framework"]


class FluentDatasetLike(DatasetLike, Protocol):
    """Structural type for dataset wrappers exposing fluent transforms."""

    def map(self, map_fn: Callable[[Any], Any]) -> "FluentDatasetLike":
        """Return a dataset transformed by ``map_fn``."""
        raise NotImplementedError

    def cache(self) -> "FluentDatasetLike":
        """Return a cached/materialized dataset wrapper."""
        raise NotImplementedError

    def zip(self, other: DatasetLike) -> "FluentDatasetLike":
        """Return a dataset zipped with ``other``."""
        raise NotImplementedError

    def batch(self, batch_size: int) -> "FluentDatasetLike":
        """Return a dataset grouped into batches of ``batch_size``."""
        raise NotImplementedError

    def unbatch(self) -> "FluentDatasetLike":
        """Return a dataset where batched elements are flattened."""
        raise NotImplementedError

    def shuffle(self, buffer_size: int) -> "FluentDatasetLike":
        """Return a dataset shuffled with a finite ``buffer_size``."""
        raise NotImplementedError

    def take(self, count: int) -> "FluentDatasetLike":
        """Return a dataset restricted to the first ``count`` elements."""
        raise NotImplementedError


__all__ = [
    "BackendLike",
    "DatasetLike",
    "DType",
    "ElementSpec",
    "FluentDatasetLike",
    "Layer",
    "LossFunction",
    "Model",
    "Tensor",
    "WeightVariable",
]
