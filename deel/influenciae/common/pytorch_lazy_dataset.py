"""
Lazy, re-iterable dataset wrappers for the PyTorch backend.
"""
from math import ceil
import random
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional, Protocol, Union, cast
from warnings import warn

from .._optional_imports import import_optional_module

torch = import_optional_module("torch", extra="pytorch")

if TYPE_CHECKING:
    import tensorflow as tf

    class PyTorchDatasetLike(Protocol):
        """Protocol compatible with torch Dataset/DataLoader objects."""

        def __iter__(self) -> Iterator[Any]:
            ...

    DatasetLike = Union[
        tf.data.Dataset,
        PyTorchDatasetLike,
        "LazyDataset",
    ]
else:
    DatasetLike = Any


def _safe_len(dataset: DatasetLike) -> Optional[int]:
    """Return ``len(dataset)`` when available."""
    try:
        return len(cast(Any, dataset))
    except TypeError:
        return None


def _is_iterator(dataset: DatasetLike) -> bool:
    """Check whether an object is a one-pass iterator."""
    try:
        return iter(dataset) is dataset
    except TypeError:
        return False


def ensure_reiterable(
    dataset: DatasetLike,
    context: str = "dataset",
) -> Union[DatasetLike, List[Any]]:
    """
    Ensure a dataset can be iterated multiple times.

    Accepted inputs include ``tf.data.Dataset`` and PyTorch dataset variants
    (``torch.utils.data.Dataset`` / ``torch.utils.data.DataLoader``), plus
    other iterable wrappers used internally.

    If the provided object is a one-pass iterator, materialize it into a list to
    avoid silent exhaustion in multi-pass workflows.
    """
    if _is_iterator(dataset):
        warn(
            f"{context} is a one-pass iterator; materializing it to preserve multi-pass behavior.",
            RuntimeWarning,
            stacklevel=2,
        )
        return list(dataset)
    return dataset


class LazyDataset:
    """Base class for lazy datasets with optional list-style compatibility."""

    def __init__(self):
        self._materialized: Optional[List[Any]] = None

    def _iter_impl(self) -> Iterator[Any]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        if self._materialized is not None:
            return iter(self._materialized)
        return self._iter_impl()

    def materialize(self) -> List[Any]:
        """Materialize the dataset in memory and return it."""
        if self._materialized is None:
            self._materialized = list(self._iter_impl())
        return self._materialized

    def __len__(self) -> int:
        return len(self.materialize())

    def __getitem__(self, index: int) -> Any:
        return self.materialize()[index]


class MappedDataset(LazyDataset):
    """Dataset applying a lazy mapping function."""

    def __init__(self, source: DatasetLike, map_fn: Callable[[Any], Any]):
        super().__init__()
        self.source = ensure_reiterable(source, context="map_dataset input")
        self.map_fn = map_fn
        self.batch_size = getattr(source, "batch_size", None)

    def _iter_impl(self) -> Iterator[Any]:
        for item in self.source:
            yield self.map_fn(item)

    def __len__(self) -> int:
        source_len = _safe_len(self.source)
        if source_len is not None:
            return source_len
        return super().__len__()


class CachedDataset(LazyDataset):
    """Materialized dataset used as an explicit cache boundary."""

    def __init__(self, source: DatasetLike):
        super().__init__()
        source = ensure_reiterable(source, context="cache_dataset input")
        self._cached_data = list(source)
        self._materialized = self._cached_data
        self.batch_size = getattr(source, "batch_size", None)

    def _iter_impl(self) -> Iterator[Any]:
        return iter(self._cached_data)


class ZippedDataset(LazyDataset):
    """Lazy zip of two datasets."""

    def __init__(self, dataset1: DatasetLike, dataset2: DatasetLike):
        super().__init__()
        self.dataset1 = ensure_reiterable(dataset1, context="zip_datasets first input")
        self.dataset2 = ensure_reiterable(dataset2, context="zip_datasets second input")
        self.batch_size = getattr(dataset1, "batch_size", None)

    def _iter_impl(self) -> Iterator[Any]:
        return iter(zip(self.dataset1, self.dataset2))

    def __len__(self) -> int:
        len1 = _safe_len(self.dataset1)
        len2 = _safe_len(self.dataset2)
        if len1 is not None and len2 is not None:
            return min(len1, len2)
        return super().__len__()


def _collate_elements(elements: List[Any]) -> Any:
    """Collate a list of values into a batched value."""
    if not elements:
        return elements
    if isinstance(elements[0], torch.Tensor):
        return torch.stack(elements)
    return elements


def _collate_batch_items(batch_items: List[Any]) -> Any:
    """Collate a list of dataset samples into one batch."""
    first = batch_items[0]
    if isinstance(first, tuple):
        return tuple(
            _collate_elements([item[idx] for item in batch_items])
            for idx in range(len(first))
        )

    if isinstance(first, list):
        return [
            _collate_elements([item[idx] for item in batch_items])
            for idx in range(len(first))
        ]

    if isinstance(first, torch.Tensor):
        return torch.stack(batch_items)

    return tuple(batch_items)


class BatchedDataset(LazyDataset):
    """Lazy batching wrapper."""

    def __init__(self, source: DatasetLike, batch_size: int):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.source = ensure_reiterable(source, context="batch_dataset input")
        self.batch_size = int(batch_size)

    def _iter_impl(self) -> Iterator[Any]:
        buffer = []
        for item in self.source:
            buffer.append(item)
            if len(buffer) == self.batch_size:
                yield _collate_batch_items(buffer)
                buffer = []

        if buffer:
            yield _collate_batch_items(buffer)

    def __len__(self) -> int:
        source_len = _safe_len(self.source)
        if source_len is not None:
            return ceil(source_len / self.batch_size)
        return super().__len__()


def _infer_batch_len(batch: Any) -> Optional[int]:
    """Infer the number of samples contained in a batch."""
    if isinstance(batch, torch.Tensor):
        if batch.dim() == 0:
            return None
        return int(batch.shape[0])

    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        first = batch[0]
        if isinstance(first, torch.Tensor):
            if first.dim() == 0:
                return None
            return int(first.shape[0])
        try:
            return len(first)
        except TypeError:
            return None

    return None


class UnbatchedDataset(LazyDataset):
    """Lazy unbatching wrapper."""

    def __init__(self, source: DatasetLike):
        super().__init__()
        self.source = ensure_reiterable(source, context="unbatch_dataset input")

    def _iter_impl(self) -> Iterator[Any]:
        for batch in self.source:
            batch_len = _infer_batch_len(batch)

            if batch_len is None:
                yield batch
                continue

            if isinstance(batch, tuple):
                for idx in range(batch_len):
                    yield tuple(item[idx] for item in batch)
                continue

            if isinstance(batch, list):
                for idx in range(batch_len):
                    yield [item[idx] for item in batch]
                continue

            if isinstance(batch, torch.Tensor):
                for idx in range(batch_len):
                    yield batch[idx]
                continue

            yield batch


class TakenDataset(LazyDataset):
    """Lazy dataset restricted to a fixed number of items."""

    def __init__(self, source: DatasetLike, count: int):
        super().__init__()
        self.source = source
        self.count = max(0, int(count))
        self.batch_size = getattr(source, "batch_size", None)

    def _iter_impl(self) -> Iterator[Any]:
        taken_items: List[Any] = []
        remaining = self.count
        source_iter = iter(self.source)

        while remaining > 0:
            try:
                item = next(source_iter)
            except StopIteration:
                break
            taken_items.append(item)
            remaining -= 1

        self._materialized = taken_items
        return iter(taken_items)

    def __len__(self) -> int:
        source_len = _safe_len(self.source)
        if source_len is not None:
            return min(source_len, self.count)
        return super().__len__()


class BufferedShuffleDataset(LazyDataset):
    """Lazy shuffle with bounded memory via a finite buffer."""

    def __init__(self, source: DatasetLike, buffer_size: int):
        super().__init__()
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        self.source = ensure_reiterable(source, context="shuffle_dataset input")
        self.buffer_size = int(buffer_size)
        self.batch_size = getattr(source, "batch_size", None)

    def _iter_impl(self) -> Iterator[Any]:
        source_iter = iter(self.source)
        buffer = []

        for _ in range(self.buffer_size):
            try:
                buffer.append(next(source_iter))
            except StopIteration:
                break

        if not buffer:
            return

        for item in source_iter:
            swap_idx = random.randint(0, len(buffer) - 1)
            yield buffer[swap_idx]
            buffer[swap_idx] = item

        random.shuffle(buffer)
        for item in buffer:
            yield item

    def __len__(self) -> int:
        source_len = _safe_len(self.source)
        if source_len is not None:
            return source_len
        return super().__len__()
