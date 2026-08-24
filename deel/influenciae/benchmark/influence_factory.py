# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing factories for the different influence calculation techniques.
This will be useful for streamlining the benchmarks.
"""
from abc import abstractmethod
from typing import Any, Callable, Optional, Union

from .._optional_imports import import_optional_attr, import_optional_module
from ..common import (
    InfluenceModel,
    Framework,
    get_backend_for_model,
    ExactIHVP,
    ConjugateGradientDescentIHVP,
    LissaIHVP,
    ExactIHVPFactory,
    CGDIHVPFactory,
    LissaIHVPFactory,
)
from ..common.ihvp_factory import AstraIHVPFactory
from ..common.inverse_hessian_vector_product import AstraIHVP
from ..influence import FirstOrderInfluenceCalculator, ArnoldiInfluenceCalculator
from ..rps import RepresenterPointLJE, RepresenterPointL2
from ..trac_in import TracIn
from ..boundary_based import WeightsBoundaryCalculator, SampleBoundaryCalculator
from ..types import DatasetLike


def _resolve_default_loss_function(model: Any) -> Callable:
    """Resolve default non-reduced loss according to model backend."""
    backend = get_backend_for_model(model)

    if backend.framework == Framework.TENSORFLOW:
        tf = import_optional_module("tensorflow", extra="tensorflow")
        reduction = import_optional_attr("tensorflow.keras.losses", "Reduction", extra="tensorflow")

        return tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=reduction.NONE)

    f = import_optional_module("torch.nn.functional", extra="pytorch")

    def pytorch_cross_entropy_no_reduction(predictions, targets):
        """PyTorch cross-entropy compatible with class indices or one-hot labels."""
        if hasattr(targets, 'ndim') and hasattr(predictions, 'ndim'):
            if targets.ndim == predictions.ndim and predictions.ndim >= 2:
                targets = targets.argmax(dim=-1)
        return f.cross_entropy(predictions, targets, reduction='none')

    return pytorch_cross_entropy_no_reduction


def _get_dataset_subset_for_hessian(
    training_dataset: DatasetLike,
    n_samples: Optional[int],
    model: Any,
) -> DatasetLike:
    """Extract a fixed number of samples from a batched dataset for Hessian estimation."""
    if n_samples is None or n_samples < 0:
        return training_dataset

    backend = get_backend_for_model(model)
    batch_size = backend.get_dataset_batch_size(training_dataset)
    unbatched_dataset = backend.unbatch_dataset(training_dataset)
    subset = backend.take_dataset(unbatched_dataset, n_samples)
    return backend.batch_dataset(subset, batch_size)


class InfluenceCalculatorFactory:
    """
    An interface for factories generating instances of the different influence calculators.
    """

    @abstractmethod
    def build(self, training_dataset: DatasetLike, model: Any,
              train_info: Optional[Any] = None) -> Any:
        """
        Builds an instance of an influence calculator class following the provided model, training dataset
        and additional information.
        """
        raise NotImplementedError


class FirstOrderFactory(InfluenceCalculatorFactory):
    """A factory for creating instances of FirstOrderInfluenceCalculator objects."""

    def __init__(self, ihvp_mode: str, start_layer: int = -1, dataset_hessian_size: int = -1, n_opt_iters: int = 100,
                 feature_extractor: Any = -1, loss_function: Optional[Callable] = None,
                 astra_factory: Optional[AstraIHVPFactory] = None):
        self.start_layer = start_layer
        self.ihvp_mode = ihvp_mode
        self.n_opt_iters = n_opt_iters
        self.feature_extractor = feature_extractor
        self.dataset_hessian_size = dataset_hessian_size
        self.loss_function = loss_function
        self.astra_factory = astra_factory or AstraIHVPFactory()
        assert self.ihvp_mode in ['exact', 'cgd', 'lissa', 'astra']

    def build(self, training_dataset: DatasetLike, model: Any,
              train_info: Any = None) -> FirstOrderInfluenceCalculator:
        del train_info

        loss_function = self.loss_function or _resolve_default_loss_function(model)
        influence_model = InfluenceModel(model, start_layer=self.start_layer, loss_function=loss_function)

        dataset_hessian = _get_dataset_subset_for_hessian(training_dataset, self.dataset_hessian_size, model)

        ihvp_calculator: Union[ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP, AstraIHVP]
        if self.ihvp_mode == 'exact':
            ihvp_calculator = ExactIHVP(influence_model, dataset_hessian)
        elif self.ihvp_mode == 'cgd':
            ihvp_calculator = ConjugateGradientDescentIHVP(
                influence_model,
                self.feature_extractor,
                dataset_hessian,
                self.n_opt_iters
            )
        elif self.ihvp_mode == 'lissa':
            ihvp_calculator = LissaIHVP(
                influence_model,
                self.feature_extractor,
                dataset_hessian,
                self.n_opt_iters,
                damping=1e-4,
                scale=5.
            )
        elif self.ihvp_mode == 'astra':
            ihvp_calculator = self.astra_factory.build(influence_model, dataset_hessian)
        else:
            raise ValueError("unknown ihvp calculator=" + self.ihvp_mode)

        return FirstOrderInfluenceCalculator(
            influence_model,
            training_dataset,
            ihvp_calculator,
            n_samples_for_hessian=self.dataset_hessian_size
        )


class RPSLJEFactory(InfluenceCalculatorFactory):
    """A factory for creating instances of representer point LJE objects."""

    def __init__(self, ihvp_mode: str, start_layer: int = -1, dataset_hessian_size: int = -1, n_opt_iters: int = 100,
                 feature_extractor: Any = -1, loss_function: Optional[Callable] = None):
        self.start_layer = start_layer
        self.ihvp_mode = ihvp_mode
        self.n_opt_iters = n_opt_iters
        self.feature_extractor = feature_extractor
        self.dataset_hessian_size = dataset_hessian_size
        self.loss_function = loss_function
        assert self.ihvp_mode in ['exact', 'cgd', 'lissa']

    def build(self, training_dataset: DatasetLike, model: Any,
              train_info: Any = None) -> RepresenterPointLJE:
        del train_info

        loss_function = self.loss_function or _resolve_default_loss_function(model)
        influence_model = InfluenceModel(model, start_layer=self.start_layer, loss_function=loss_function)

        dataset_hessian = _get_dataset_subset_for_hessian(training_dataset, self.dataset_hessian_size, model)

        ihvp_calculator_factory: Union[ExactIHVPFactory, CGDIHVPFactory, LissaIHVPFactory]
        if self.ihvp_mode == 'exact':
            ihvp_calculator_factory = ExactIHVPFactory()
        elif self.ihvp_mode == 'cgd':
            ihvp_calculator_factory = CGDIHVPFactory(self.feature_extractor, self.n_opt_iters)
        elif self.ihvp_mode == 'lissa':
            ihvp_calculator_factory = LissaIHVPFactory(self.feature_extractor, self.n_opt_iters, damping=1e-4, scale=5.)
        else:
            raise ValueError("unknown ihvp calculator=" + self.ihvp_mode)

        return RepresenterPointLJE(influence_model, dataset_hessian, ihvp_calculator_factory)


class TracInFactory(InfluenceCalculatorFactory):
    """A factory for creating instances of TracIn objects."""

    def __init__(self, loss_function: Optional[Callable] = None):
        self.loss_function = loss_function

    def build(self, training_dataset: DatasetLike, model: Any,
              train_info: Optional[Any] = None) -> TracIn:
        del training_dataset
        if train_info is None:
            raise ValueError("TracInFactory requires `train_info=(models, learning_rates)`.")

        loss_function = self.loss_function or _resolve_default_loss_function(model)

        models = []
        for model_data in train_info[0]:
            influence_model = InfluenceModel(model_data, loss_function=loss_function)
            models.append(influence_model)
        return TracIn(models, train_info[1])


class RPSL2Factory(InfluenceCalculatorFactory):
    """A factory for creating instances of RepresenterPointL2 objects."""

    def __init__(
            self,
            loss_function: Any,
            lambda_regularization: float,
            scaling_factor: float = 0.1,
            layer_index: int = -2,
            epochs: int = 100
    ):
        self.loss_function = loss_function
        self.lambda_regularization = lambda_regularization
        self.scaling_factor = scaling_factor
        self.epochs = epochs
        self.layer_index = layer_index

    def build(self, training_dataset: DatasetLike, model: Any,
              train_info: Any = None) -> RepresenterPointL2:
        del train_info
        return RepresenterPointL2(
            model,
            training_dataset,
            self.loss_function,
            self.lambda_regularization,
            self.scaling_factor,
            self.epochs,
            self.layer_index,
        )


class WeightsBoundaryCalculatorFactory(InfluenceCalculatorFactory):
    """A factory for creating instances of WeightsBoundaryCalculator objects."""

    def __init__(self, step_nbr: int = 100, norm_type: int = 2):
        self.step_nbr = step_nbr
        self.norm_type = norm_type

    def build(self, training_dataset: DatasetLike, model: Any,
              train_info: Any = None) -> WeightsBoundaryCalculator:
        del training_dataset
        del train_info
        return WeightsBoundaryCalculator(model, self.step_nbr, self.norm_type)


class SampleBoundaryCalculatorFactory(InfluenceCalculatorFactory):
    """A factory for creating instances of SampleBoundaryCalculator objects."""

    def __init__(self, step_nbr: int = 100):
        self.step_nbr = step_nbr

    def build(self, training_dataset: DatasetLike, model: Any,
              train_info: Any = None) -> SampleBoundaryCalculator:
        del training_dataset
        del train_info
        return SampleBoundaryCalculator(model, self.step_nbr)


class ArnoldiCalculatorFactory(InfluenceCalculatorFactory):
    """A factory for creating instances of ArnoldiInfluenceCalculator objects."""

    def __init__(
            self,
            subspace_dim: int,
            force_hermitian: bool,
            k_largest_eig_vals: int,
            start_layer: int = -1,
            dataset_hessian_size: int = -1,
            loss_function: Optional[Callable] = None,
            dtype: Any = None
    ):
        self.subspace_dim = subspace_dim
        self.force_hermitian = force_hermitian
        self.k_largest_eig_vals = k_largest_eig_vals
        self.start_layer = start_layer
        self.loss_function = loss_function
        self.dataset_hessian_size = dataset_hessian_size
        self.dtype = dtype

    def build(self, training_dataset: DatasetLike, model: Any,
              train_info: Any = None) -> ArnoldiInfluenceCalculator:
        del train_info

        loss_function = self.loss_function or _resolve_default_loss_function(model)
        influence_model = InfluenceModel(model, start_layer=self.start_layer, loss_function=loss_function)

        dataset_hessian = _get_dataset_subset_for_hessian(training_dataset, self.dataset_hessian_size, model)

        backend = get_backend_for_model(model)
        dtype = self.dtype if self.dtype is not None else backend.float32_dtype()

        return ArnoldiInfluenceCalculator(
            influence_model,
            dataset_hessian,
            self.subspace_dim,
            self.force_hermitian,
            self.k_largest_eig_vals,
            dtype,
        )
