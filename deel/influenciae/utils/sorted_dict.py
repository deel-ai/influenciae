# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module containing the implementation of a SortedDictionary, useful for computing the
top-k most influential samples in a large dataset.
"""
from enum import Enum
from typing import Tuple, Any, Optional, Union, cast

import numpy as np

from ..common.backend import (
    BaseBackend,
    Framework,
    get_backend,
    detect_dtype_framework,
)


class ORDER(Enum):
    """
    Enumeration for the two types of ordering for the sorting function.
    ASCENDING puts the elements with the smallest value first.
    DESCENDING puts the elements with the largest value first.
    """
    ASCENDING = 1
    DESCENDING = 2


class BatchSort:
    """
    An implementation of a SortedDictionary that accepts batches of elements and their corresponding scores
    and keeps only the k most/least important ones.

    This class is backend-agnostic and supports both TensorFlow and PyTorch.

    Attributes
    ----------
    batch_shape
        A tuple with the shape of the dictionary's inputs.
    k_shape
        A tuple with the shape of the elements to keep.
    dtype
        The data-type of the input's values. By default, float32 will be used.
    order
        Either ASCENDING or DESCENDING depending on whether the most or the least important elements are to
        be kept.
    backend
        The backend to use for tensor operations. If None, it will be inferred from the first batch added
        or default to TensorFlow if available.
    device
        The device to create tensors on (PyTorch only). If None, tensors will be created on the default device.
    """
    def __init__(
        self,
        batch_shape: Tuple[int, ...],
        k_shape: Tuple[int, ...],
        dtype: Optional[Any] = None,
        order: ORDER = ORDER.DESCENDING,
        backend: Optional[Union[BaseBackend, Framework]] = None,
        device: Optional[Any] = None
    ):
        self._backend = self._resolve_backend(backend, dtype)
        self._dtype = self._backend.float32_dtype() if dtype is None else dtype

        self.k = k_shape[1]
        self.order = order
        self._device = device
        self._initialized = False  # Track if we've seen real data

        # Create shape by concatenating k_shape and batch_shape
        shape = tuple(k_shape) + tuple(batch_shape)
        self._shape = shape
        self._k_shape = k_shape

        self._best_batch = self._backend.zeros(shape, dtype=self._dtype)
        self._best_values = self._initialize_best_values(k_shape)

        if device is not None and self._backend.framework == Framework.PYTORCH:
            self._best_batch = cast(Any, self._best_batch).to(device)
            self._best_values = cast(Any, self._best_values).to(device)

    @staticmethod
    def _default_backend() -> BaseBackend:
        """Pick TensorFlow backend when available, else fallback to PyTorch."""
        try:
            return get_backend(Framework.TENSORFLOW)
        except ImportError:
            return get_backend(Framework.PYTORCH)

    @classmethod
    def _resolve_backend(
        cls,
        backend: Optional[Union[BaseBackend, Framework]],
        dtype: Optional[Any],
    ) -> BaseBackend:
        """Resolve backend from explicit arg, dtype inference, or defaults."""
        if isinstance(backend, BaseBackend):
            return backend

        if backend is not None:
            return get_backend(backend)

        if dtype is not None:
            inferred_framework = detect_dtype_framework(dtype)
            if inferred_framework is not None:
                return get_backend(inferred_framework)

        return cls._default_backend()

    def _initialize_best_values(self, k_shape: Tuple[int, ...]) -> Any:
        """Initialize score storage according to sort order."""
        values = self._backend.ones(k_shape, dtype=self._dtype)
        if self.order == ORDER.DESCENDING:
            return values * (-np.inf)
        return values * np.inf

    @property
    def backend(self) -> BaseBackend:
        """Return the backend used for tensor operations."""
        return self._backend

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the internal tensor shape used by this sorter."""
        return self._shape

    @property
    def dtype(self) -> Any:
        """Return the dtype used for stored values."""
        return self._dtype

    def add_all(self, batch_key: Any, batch_values: Any) -> None:
        """
        Add a new batch of data (element and values) and update the sorted dictionary retaining only the
        top/bottom-k elements.

        Parameters
        ----------
        batch_key
            A batch of new elements in the form of a tensor.
        batch_values
            A batch of their corresponding values in the form of a tensor.
        """
        # For PyTorch, ensure all tensors are on the same device
        if self._backend.framework == Framework.PYTORCH:
            # Move internal tensors to the device of the incoming batch if needed
            if not self._initialized and hasattr(batch_values, 'device'):
                target_device = batch_values.device
                self._best_batch = cast(Any, self._best_batch).to(target_device)
                self._best_values = cast(Any, self._best_values).to(target_device)
                self._device = target_device
                self._initialized = True

        batch_key = self._backend.cast(batch_key, self._backend.get_dtype(self._best_batch))
        batch_values = self._backend.cast(batch_values, self._backend.get_dtype(self._best_values))

        current_score = self._backend.concat([self._best_values, batch_values], axis=1)
        current_batch = self._backend.concat([self._best_batch, batch_key], axis=1)

        descending = self.order == ORDER.DESCENDING
        indexes = self._backend.argsort(current_score, axis=1, descending=descending)
        indexes = indexes[:, :self.k]

        current_best_score = self._backend.gather_along_axis(current_score, indexes, axis=1, batch_dims=1)
        current_best_samples = self._backend.gather_along_axis(current_batch, indexes, axis=1, batch_dims=1)

        self._best_values = current_best_score
        self._best_batch = current_best_samples

    def get(self) -> Tuple[Any, Any]:
        """
        A getter method for the top-k elements and their corresponding scores.

        Returns
        -------
        top_k_tuple
            A tuple with the top_k elements and their scores.
        """
        return self._best_batch, self._best_values

    def reset(self) -> None:
        """
        Resets the values of the whole object and prepares it to start sorting from scratch.
        """
        self._best_batch = self._backend.zeros_like(self._best_batch)

        if self.order == ORDER.DESCENDING:
            self._best_values = self._backend.ones_like(self._best_values) * (-np.inf)
        else:
            self._best_values = self._backend.ones_like(self._best_values) * np.inf
