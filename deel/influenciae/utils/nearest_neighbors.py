# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a Nearest Neighbors interface and a Linear Nearest Neighbors
algorithm. It will prove itself useful for finding the top-k most influential
examples of datasets, as implemented in the influence calculator interface.

This module is backend-agnostic and supports both TensorFlow and PyTorch.
"""
from abc import abstractmethod

from .sorted_dict import BatchSort, ORDER
from ..common import (
    BaseBackend,
    Framework,
    get_backend,
)
from ..types import Any, Callable, Optional, Tuple, Union


class BaseNearestNeighbors:
    """
    A Nearest Neighbors interface for efficiently searching for specific data-points
    in a dataset. This class is backend-agnostic and supports both TensorFlow and PyTorch.
    """

    @abstractmethod
    def build(
        self,
        dataset: Any,
        dot_product_fun: Callable[[Any, Any], Any],
        k: int,
        query_batch_size: int,
        d_type: Optional[Any] = None,
        order: ORDER = ORDER.DESCENDING
    ) -> None:
        """
        Builds the nearest neighbors object that will be used to find the k neighbors
        inside the provided dataset.

        Parameters
        ----------
        dataset
            A dataset containing the points which shall be indexed.
            (tf.data.Dataset for TensorFlow, DataLoader or list of tuples for PyTorch)
        dot_product_fun
            The dot product function used to compute the distance between 2 points
        k
            An integer for the amount of samples to search for
        query_batch_size
            An integer for the query's batch size
        d_type
            The dataset's element's data-type. If None, will be inferred.
        order
            Either descending or ascending for the top or bottom results as per the similarity metric
        """
        raise NotImplementedError()

    @abstractmethod
    def query(self, vector_to_find: Any, batch_size: Optional[int] = None) -> Tuple[Any, Any]:
        """
        Find the k closest points to the provided vector in the object's dataset.

        Parameters
        ----------
        vector_to_find
            A tensor of points to query.
        batch_size
            An integer with the query's batch size

        Returns
        -------
        values_and_samples
            A tuple with the k closest points to the queries and their corresponding distances in the format
            (distances, points)
        """
        raise NotImplementedError()


class LinearNearestNeighbors(BaseNearestNeighbors):
    """
    An implementation of a Linear Nearest Neighbors search algorithm using a given similarity metric and
    doing as much lazy computations as possible for scalability.

    This class is backend-agnostic and supports both TensorFlow and PyTorch.

    Attributes
    ----------
    dataset
        A dataset with the points from which the nearest neighbors will be searched.
    dot_product_fun
        A callable taking in two vectors and returning a notion of similarity between them.
    batched_sorted_dict
        A BatchSort instance that takes care of keeping the top/bottom most similar examples from
        the batch.
    backend
        The backend to use for tensor operations. If None, it will be inferred from the first batch
        or default to TensorFlow if available.
    """

    def __init__(self, backend: Optional[Union[BaseBackend, Framework]] = None):
        """
        Initialize the LinearNearestNeighbors.

        Parameters
        ----------
        backend
            The backend to use for tensor operations. Can be a BaseBackend instance,
            a Framework enum (TENSORFLOW or PYTORCH), or None to infer from data.
        """
        self.dataset = None
        self.dot_product_fun = None
        self.batched_sorted_dict = None
        self._backend_param = backend
        self._backend: Optional[BaseBackend] = None

    @property
    def backend(self) -> BaseBackend:
        """Return the backend used for tensor operations."""
        if self._backend is None:
            # Determine the backend
            if self._backend_param is None:
                # Default to TensorFlow if available, otherwise PyTorch
                try:
                    self._backend = get_backend(Framework.TENSORFLOW)
                except ImportError:
                    self._backend = get_backend(Framework.PYTORCH)
            elif isinstance(self._backend_param, BaseBackend):
                self._backend = self._backend_param
            else:
                # backend_param is a Framework enum
                self._backend = get_backend(self._backend_param)
        return self._backend

    def build(
        self,
        dataset: Any,
        dot_product_fun: Callable[[Any, Any], Any],
        k: int,
        query_batch_size: int,
        d_type: Optional[Any] = None,
        order: ORDER = ORDER.DESCENDING
        ) -> None:
        """
        Builds the linear nearest neighbors object that will be used to find the k neighbors
        inside the provided dataset.

        Parameters
        ----------
        dataset
            A dataset containing the points which shall be indexed.
            (tf.data.Dataset for TensorFlow, DataLoader or list of tuples for PyTorch)
        dot_product_fun
            The dot product function used to compute the distance between 2 points
        k
            An integer for the amount of samples to search for
        query_batch_size
            An integer for the query's batch size
        d_type
            The dataset's element's data-type. If None, will be inferred from the backend's default.
        order
            Either descending or ascending for the top or bottom results as per the similarity metric
        """
        self.dataset = dataset
        self.dot_product_fun = dot_product_fun

        # Get batch shape from dataset element spec
        elt_spec = self.backend.get_dataset_element_spec(dataset)
        batch_shape: Tuple[int, ...] = ()

        # Handle nested specs: we expect ((batch_samples, ...), ihvp) structure
        # and we need shape from the batch_samples
        if isinstance(elt_spec, (list, tuple)):
            first_spec = elt_spec[0]
            if isinstance(first_spec, (list, tuple)):
                # ((x, y), ihvp) structure
                first_spec = first_spec[0]
        else:
            first_spec = elt_spec

        # Extract shape from spec
        if hasattr(first_spec, 'shape'):
            batch_shape = tuple(first_spec.shape[1:])  # Remove batch dimension
        elif isinstance(first_spec, dict) and 'shape' in first_spec:
            batch_shape = tuple(first_spec['shape'][1:])

        # Fallback: iterate to get shape from first batch if not available from spec
        if not batch_shape:
            for item in dataset:
                if isinstance(item, (list, tuple)):
                    first_item = item[0]
                    if isinstance(first_item, (list, tuple)):
                        first_tensor = first_item[0]
                    else:
                        first_tensor = first_item
                else:
                    first_tensor = item
                batch_shape = tuple(self.backend.tensor_shape(first_tensor)[1:])
                break

        # Use backend default dtype if not provided
        if d_type is None:
            d_type = self.backend.float32_dtype()

        self.batched_sorted_dict = BatchSort(
            batch_shape,
            (query_batch_size, k),
            dtype=d_type,
            order=order,
            backend=self._backend
        )

    def query(self, vector_to_find: Any, batch_size: Optional[int] = None) -> Tuple[Any, Any]:
        """
        Find the k closest points to the provided vector in the object's dataset.

        Parameters
        ----------
        vector_to_find
            A tensor of points to query.
        batch_size
            An integer with the query's batch size

        Returns
        -------
        values_and_samples
            A tuple with the k closest points to the queries and their corresponding distances in the format
            (distances, points)
        """
        if batch_size is None:
            batch_size = self.backend.get_batch_size(vector_to_find)

        self.batched_sorted_dict.reset()

        # Iterate through the dataset
        for batch_data in self.dataset:
            # Expected structure: (batch, ihvp) where batch is (samples, labels, ...)
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                batch, ihvp = batch_data[0], batch_data[-1]

                # Handle nested batch structure (samples may be first element of batch)
                if isinstance(batch, (list, tuple)):
                    batch_samples = batch[0]
                else:
                    batch_samples = batch
            else:
                # Fallback for simpler dataset structure
                batch_samples = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
                ihvp = batch_data[-1] if isinstance(batch_data, (list, tuple)) else batch_data

            # Compute influence values using the dot product function
            influence_values = self.dot_product_fun(vector_to_find, ihvp)

            # Expand batch_samples to match query batch size and add to sorted dict
            expanded_batch = self.backend.repeat(
                self.backend.expand_dims(batch_samples, axis=0),
                batch_size,
                axis=0
            )

            self.batched_sorted_dict.add_all(expanded_batch, influence_values)

        training_samples, influences_values = self.batched_sorted_dict.get()

        return influences_values, training_samples
