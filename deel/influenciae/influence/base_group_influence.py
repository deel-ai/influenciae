# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing the necessary functions for computing influence functions on
groups of data -- i.e. performing approximate hold-one-out of groups of data-points
as a whole instead of individually. The theoretical validity of this implementation
was discussed in https://arxiv.org/abs/1905.13289 , with positive results.

We propose two classes that implement this interface: the FirstOrderInfluenceCalculator
(using the technique proposed in https://arxiv.org/abs/1905.13289) and the
SecondOrderInfluenceCalculator, that is only available for calculating groups'
influence and does so through the method presented in
https://arxiv.org/abs/1911.00418
"""

from abc import abstractmethod
from typing import Optional, Union

from ..common import InfluenceModel
from ..common import InverseHessianVectorProduct, InverseHessianVectorProductFactory, IHVPCalculator
from ..common import BaseBackend

from ..types import DatasetLike, Tensor


class BaseGroupInfluenceCalculator:
    """
    A base class for objects that calculate the different quantities related to influence
    functions for whole groups of data-points.

    Notes
    -----
    The methods currently implemented are available to evaluate groups of points:
    - Influence function vectors: the weights difference when removing entire groups of points.
    - Influence values/Cook's distance: a measure of reliance of the model on the entire group.

    Parameters
    ----------
    model
        The model implementing the InfluenceModel interface (TensorFlow or PyTorch).
    dataset
        A batched dataset containing the training dataset over which we will estimate the
        inverse-hessian-vector product.
    ihvp_calculator
        Either a string containing the IHVP method ('exact' or 'cgd'), an IHVPCalculator
        object, an InverseHessianVectorProductFactory object, or an
        InverseHessianVectorProduct object.
    n_samples_for_hessian
        An integer indicating the amount of samples to take from the provided train dataset.
    shuffle_buffer_size
        An integer indicating the buffer size of the train dataset's shuffle operation -- when
        choosing the amount of samples for the hessian.
    """
    # Backend is set by subclasses that have access to a model.
    backend: BaseBackend

    @property
    def _backend(self) -> BaseBackend:
        """Return initialized backend instance."""
        if self.backend is None:
            raise ValueError("Backend is not initialized. Instantiate a calculator with a valid model first.")
        return self.backend

    def __init__(
            self,
            model: InfluenceModel,
            dataset: DatasetLike,
            ihvp_calculator: Union[
                str,
                InverseHessianVectorProduct,
                InverseHessianVectorProductFactory,
                IHVPCalculator,
            ] = 'exact',
            n_samples_for_hessian: Optional[int] = None,
            shuffle_buffer_size: Optional[int] = 10000
    ):
        self.model = model
        self.backend = model.backend

        if n_samples_for_hessian is None:
            dataset_to_estimate_hessian = dataset
        else:
            # Use backend-agnostic dataset operations
            batch_size = self._backend.get_dataset_batch_size(dataset)
            unbatched = self._backend.unbatch_dataset(dataset)
            # Ensure shuffle_buffer_size has a default value
            buffer_size = shuffle_buffer_size if shuffle_buffer_size is not None else 10000
            shuffled = self._backend.shuffle_dataset(unbatched, buffer_size)
            taken = self._backend.take_dataset(shuffled, n_samples_for_hessian)
            dataset_to_estimate_hessian = self._backend.batch_dataset(taken, batch_size)

        self.train_set: DatasetLike = dataset_to_estimate_hessian
        self.ihvp_calculator: InverseHessianVectorProduct

        # load ivhp calculator from str, IHVPcalculator enum or InverseHessianVectorProduct object
        if isinstance(ihvp_calculator, str):
            self.ihvp_calculator = self._build_ihvp(IHVPCalculator.from_string(ihvp_calculator))
        elif isinstance(ihvp_calculator, IHVPCalculator):
            self.ihvp_calculator = self._build_ihvp(ihvp_calculator)
        elif isinstance(ihvp_calculator, InverseHessianVectorProductFactory):
            self.ihvp_calculator = ihvp_calculator.build(self.model, self.train_set)
        elif isinstance(ihvp_calculator, InverseHessianVectorProduct):
            self.ihvp_calculator = ihvp_calculator
        else:
            raise ValueError("Unsupported ihvp_calculator argument type.")

    def _resolve_extractor_layer(self) -> Union[str, int]:
        """Return a valid extractor layer for approximate IHVP calculators."""
        if self.model.start_layer is None:
            raise ValueError(
                "An extractor layer must be provided in the model wrapper when using 'cgd' or 'lissa' IHVP."
            )
        return self.model.start_layer

    def _build_ihvp(self, ihvp_calculator: IHVPCalculator) -> InverseHessianVectorProduct:
        """Instantiate IHVP calculators from enum values."""
        if ihvp_calculator is IHVPCalculator.Exact:
            return ihvp_calculator.value(self.model, self.train_set)

        if ihvp_calculator in (IHVPCalculator.Kfac, IHVPCalculator.Ekfac):
            # K-FAC and EK-FAC don't need an extractor layer; they operate on
            # the full model using hook-based per-layer factor computation.
            return ihvp_calculator.value(self.model, self.train_set)

        extractor_layer = self._resolve_extractor_layer()
        return ihvp_calculator.value(self.model, extractor_layer, self.train_set)

    @abstractmethod
    def compute_influence_vector_group(
            self,
            group: DatasetLike
    ) -> Tensor:
        """
        Computes the influence function vector -- an estimation of the weights difference when
        removing the points -- of the whole group of points.

        Parameters
        ----------
        group
            A batched dataset containing the group of points of which we wish to compute the
            influence of removal.

        Returns
        -------
        influence_group
            A tensor containing one vector for the whole group.
        """
        raise NotImplementedError()

    @abstractmethod
    def estimate_influence_values_group(
            self,
            group_train: DatasetLike,
            group_to_evaluate: Optional[DatasetLike] = None
    ) -> Tensor:
        """
        Computes Cook's distance of the whole group of points provided, giving measure of the
        influence that the group carries on the model's weights.

        The dataset_train contains the points we will be removing and dataset_to_evaluate,
        those with respect to whom we will be measuring the influence. As we will be performing
        the same operation in batches, we consider that each point from one dataset corresponds
        to one from the other. As such, both datasets must contain the same amount of points.
        In case the group_to_evaluate is not given, use by default the
        group_to_train: compute the self influence of the group.


        Parameters
        ----------
        group_train
            A batched dataset containing the group of points we wish to remove.
        group_to_evaluate
            A batched dataset containing the group of points with respect to whom we wish to
            measure the influence of removing the training points.

        Returns
        -------
        influence_values_group
            A tensor containing one influence value for the whole group.
        """
        raise NotImplementedError()

    def assert_compatible_datasets(self, dataset_a: DatasetLike, dataset_b: DatasetLike) -> int:
        """
        Assert that the datasets are compatible: that they contain the same number of points. Else,
        throw an error.

        Parameters
        ----------
        dataset_a
            First batched dataset to check.
        dataset_b
            Second batched dataset to check.

        Returns
        -------
        size
            The size of the dataset.
        """
        size_a = self._backend.get_dataset_size(dataset_a)
        size_b = self._backend.get_dataset_size(dataset_b)

        if size_a != size_b:
            raise ValueError("The amount of points in the train and evaluation groups must match.")

        return size_a

    def _reduce_ihvp_batches(self, ihvp_ds: DatasetLike, keepdims: bool = True) -> Tensor:
        """Sum per-batch IHVP tensors across all batches."""
        reduced_ihvp = None
        for batch in ihvp_ds:
            batch_sum = self._backend.reduce_sum(batch, axis=1, keepdims=keepdims)
            if reduced_ihvp is None:
                reduced_ihvp = batch_sum
            else:
                reduced_ihvp = reduced_ihvp + batch_sum
        assert reduced_ihvp is not None, "ihvp_ds must not be empty"
        return reduced_ihvp
