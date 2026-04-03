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
from ..influence import FirstOrderInfluenceCalculator, ArnoldiInfluenceCalculator
from ..rps import RepresenterPointLJE, RepresenterPointL2
from ..trac_in import TracIn
from ..boundary_based import WeightsBoundaryCalculator, SampleBoundaryCalculator
from ..types import DatasetLike, DType, LossFunction, Model


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
    def build(self, model: Model, **kwargs: Any) -> Any:
        """
        Builds an instance of an influence calculator class following the provided model and
        additional keyword arguments.

        Parameters
        ----------
        model
            A model (TF or PyTorch) for which to compute the influence-related quantities.
        **kwargs
            Additional keyword arguments required by the specific factory. May include
            ``training_dataset`` (a batched dataset used to fit the model) and ``train_info``
            (extra training information, e.g. checkpoints and learning rates for TracIn).

        Returns
        -------
        influence_calculator
            The desired influence calculator instance.
        """
        raise NotImplementedError


class FirstOrderFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of FirstOrderInfluenceCalculator objects.

    Attributes
    ----------
    ihvp_mode
        A string indicating whether the IHVPs should be computed using the ``'exact'``,
        ``'cgd'``, or ``'lissa'`` method.
    start_layer
        An integer for the target layer on which to compute the influence. By default, the last
        layer of the model is chosen.
    dataset_hessian_size
        An integer for the amount of samples that should go into the computation of the Hessian
        matrix. By default, the entire training dataset is used.
    n_opt_iters
        An integer indicating how many iterations of the Conjugate Gradient Descent algorithm
        should be run before prematurely stopping the optimization.
    feature_extractor
        Either an integer for the last layer of the feature extractor, or a model (TF or
        PyTorch) for computing the embeddings of the samples. Used if ``ihvp_mode == 'cgd'``
        or ``ihvp_mode == 'lissa'``.
    loss_function
        Loss function to calculate influence. Must not apply any reduction and should match
        the output format of the model (e.g. ``from_logits=True``). When ``None``, a default
        non-reduced categorical cross-entropy is resolved from the model's backend.
    """

    def __init__(self, ihvp_mode: str, start_layer: int = -1, dataset_hessian_size: int = -1,
                 n_opt_iters: int = 100, feature_extractor: Any = -1,
                 loss_function: Optional[LossFunction] = None):
        self.start_layer = start_layer
        self.ihvp_mode = ihvp_mode
        self.n_opt_iters = n_opt_iters
        self.feature_extractor = feature_extractor
        self.dataset_hessian_size = dataset_hessian_size
        self.loss_function = loss_function
        if self.ihvp_mode not in ['exact', 'cgd', 'lissa']:
            raise ValueError(f"ihvp_mode must be one of 'exact', 'cgd', 'lissa'; got '{self.ihvp_mode}'")

    def build(self, model: Model, **kwargs: Any) -> FirstOrderInfluenceCalculator:
        """
        Builds an instance of the FirstOrderInfluenceCalculator class following the provided
        model and training dataset. No additional training information is required.

        Parameters
        ----------
        model
            A model (TF or PyTorch) for which to compute the influence-related quantities.
        **kwargs
            training_dataset
                A batched dataset (tf.data.Dataset or PyTorch DataLoader) containing the data
                on which the model was trained.

        Returns
        -------
        influence_calculator
            The desired FirstOrderInfluenceCalculator instance.
        """
        training_dataset = kwargs.get("training_dataset")
        if training_dataset is None:
            raise ValueError("FirstOrderFactory.build() requires keyword argument 'training_dataset'.")

        loss_function = self.loss_function or _resolve_default_loss_function(model)
        influence_model = InfluenceModel(model, start_layer=self.start_layer, loss_function=loss_function)

        dataset_hessian = _get_dataset_subset_for_hessian(training_dataset, self.dataset_hessian_size, model)

        ihvp_calculator: Union[ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP]
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
        else:
            raise ValueError("unknown ihvp calculator=" + self.ihvp_mode)

        return FirstOrderInfluenceCalculator(
            influence_model,
            training_dataset,
            ihvp_calculator,
            n_samples_for_hessian=self.dataset_hessian_size
        )


class RPSLJEFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of representer point LJE objects.

    Attributes
    ----------
    ihvp_mode
        A string indicating whether the IHVPs should be computed using the ``'exact'``,
        ``'cgd'``, or ``'lissa'`` method.
    start_layer
        An integer for the target layer on which to compute the influence. By default, the last
        layer of the model is chosen.
    dataset_hessian_size
        An integer for the amount of samples that should go into the computation of the Hessian
        matrix. By default, the entire training dataset is used.
    n_opt_iters
        An integer indicating how many iterations of the Conjugate Gradient Descent algorithm
        should be run before prematurely stopping the optimization.
    feature_extractor
        Either an integer for the last layer of the feature extractor, or a model (TF or
        PyTorch) for computing the embeddings of the samples. Used if ``ihvp_mode == 'cgd'``
        or ``ihvp_mode == 'lissa'``.
    loss_function
        Loss function to calculate influence. Must not apply any reduction and should match
        the output format of the model (e.g. ``from_logits=True``). When ``None``, a default
        non-reduced categorical cross-entropy is resolved from the model's backend.
    """

    def __init__(self, ihvp_mode: str, start_layer: int = -1, dataset_hessian_size: int = -1,
                 n_opt_iters: int = 100, feature_extractor: Any = -1,
                 loss_function: Optional[LossFunction] = None):
        self.start_layer = start_layer
        self.ihvp_mode = ihvp_mode
        self.n_opt_iters = n_opt_iters
        self.feature_extractor = feature_extractor
        self.dataset_hessian_size = dataset_hessian_size
        self.loss_function = loss_function
        if self.ihvp_mode not in ['exact', 'cgd', 'lissa']:
            raise ValueError(f"ihvp_mode must be one of 'exact', 'cgd', 'lissa'; got '{self.ihvp_mode}'")

    def build(self, model: Model, **kwargs: Any) -> RepresenterPointLJE:
        """
        Builds an instance of the RepresenterPointLJE class following the provided model and
        training dataset. No additional training information is required.

        Parameters
        ----------
        model
            A model (TF or PyTorch) for which to compute the influence-related quantities.
        **kwargs
            training_dataset
                A batched dataset (tf.data.Dataset or PyTorch DataLoader) containing the data
                on which the model was trained.

        Returns
        -------
        influence_calculator
            The desired RepresenterPointLJE instance.
        """
        training_dataset = kwargs.get("training_dataset")
        if training_dataset is None:
            raise ValueError("RPSLJEFactory.build() requires keyword argument 'training_dataset'.")

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
    """
    A factory for creating instances of TracIn objects.
    As it works by tracking the gradients along the training process, it also requires some
    training information to be able to compute influence values.

    Attributes
    ----------
    loss_function
        Loss function to calculate influence. Must not apply any reduction and should match
        the output format of the model (e.g. ``from_logits=True``). When ``None``, a default
        non-reduced categorical cross-entropy is resolved from the model's backend.
    """

    def __init__(self, loss_function: Optional[LossFunction] = None):
        self.loss_function = loss_function

    def build(self, model: Model, **kwargs: Any) -> TracIn:
        """
        Builds an instance of the TracIn class following the provided model and additional
        training information.

        Parameters
        ----------
        model
            A model (TF or PyTorch) for which to compute the influence-related quantities.
        **kwargs
            train_info
                A tuple whose first element is a list of model checkpoints (TF or PyTorch
                models) and whose second element is the corresponding list of learning rates.

        Returns
        -------
        influence_calculator
            The desired TracIn instance.
        """
        train_info = kwargs.get("train_info")
        if train_info is None:
            raise ValueError("TracInFactory.build() requires keyword argument 'train_info' "
                             "as a tuple of (model_checkpoints, learning_rates).")

        loss_function = self.loss_function or _resolve_default_loss_function(model)

        models = []
        for model_data in train_info[0]:
            influence_model = InfluenceModel(model_data, loss_function=loss_function)
            models.append(influence_model)
        return TracIn(models, train_info[1])


class RPSL2Factory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of RepresenterPointL2 objects.

    Attributes
    ----------
    loss_function
        The loss function with which the model was trained. This loss function must not be
        reduced.
    lambda_regularization
        The strength of the L2 regularization to add to the surrogate last layer.
    scaling_factor
        The backtracking line-search's scaling factor for training the surrogate last layer.
        By default, this value is set to 0.1 and should typically converge quite easily.
    layer_index
        The index for the layer on which to compute the influence values.
    epochs
        An integer indicating for how long the surrogate last layer should be trained. By
        default, a value of 100 is chosen.
    """

    def __init__(
            self,
            loss_function: LossFunction,
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

    def build(self, model: Model, **kwargs: Any) -> RepresenterPointL2:
        """
        Builds an instance of the RepresenterPointL2 class following the provided model and
        training dataset. No additional training information is required.

        Parameters
        ----------
        model
            A model (TF or PyTorch) for which to compute the influence-related quantities.
        **kwargs
            training_dataset
                A batched dataset (tf.data.Dataset or PyTorch DataLoader) containing the data
                on which the model was trained.

        Returns
        -------
        influence_calculator
            The desired RepresenterPointL2 instance.
        """
        training_dataset = kwargs.get("training_dataset")
        if training_dataset is None:
            raise ValueError("RPSL2Factory.build() requires keyword argument 'training_dataset'.")

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
    """
    A factory for creating instances of WeightsBoundaryCalculator objects.

    Attributes
    ----------
    step_nbr
        The number of iterations to search the boundary for. By default, a value of 100 is
        chosen.
    norm_type
        The norm type of the distance between the weights to measure the distance to the
        boundary. By default, the L2 distance is chosen.
    """

    def __init__(self, step_nbr: int = 100, norm_type: int = 2):
        self.step_nbr = step_nbr
        self.norm_type = norm_type

    def build(self, model: Model, **kwargs: Any) -> WeightsBoundaryCalculator:
        """
        Builds an instance of the WeightsBoundaryCalculator class following the provided model.
        No training dataset or additional information is required.

        Parameters
        ----------
        model
            A model (TF or PyTorch) for which to compute the influence-related quantities.
        **kwargs
            Ignored; accepted for interface compatibility.

        Returns
        -------
        influence_calculator
            The desired WeightsBoundaryCalculator instance.
        """
        return WeightsBoundaryCalculator(model, self.step_nbr, self.norm_type)


class SampleBoundaryCalculatorFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of SampleBoundaryCalculator objects.

    Attributes
    ----------
    step_nbr
        The number of iterations to search the boundary for. By default, a value of 100 is
        chosen.
    """

    def __init__(self, step_nbr: int = 100):
        self.step_nbr = step_nbr

    def build(self, model: Model, **kwargs: Any) -> SampleBoundaryCalculator:
        """
        Builds an instance of the SampleBoundaryCalculator class following the provided model.
        No training dataset or additional information is required.

        Parameters
        ----------
        model
            A model (TF or PyTorch) for which to compute the influence-related quantities.
        **kwargs
            Ignored; accepted for interface compatibility.

        Returns
        -------
        influence_calculator
            The desired SampleBoundaryCalculator instance.
        """
        return SampleBoundaryCalculator(model, self.step_nbr)


class ArnoldiCalculatorFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of ArnoldiInfluenceCalculator objects.

    Attributes
    ----------
    subspace_dim
        The dimension of the Krylov subspace for the Arnoldi algorithm.
    force_hermitian
        A boolean indicating if we should force the projected matrix to be Hermitian before
        the eigenvalue computation.
    k_largest_eig_vals
        An integer for the amount of top eigenvalues to keep for the influence estimations.
    start_layer
        An integer for the target layer on which to compute the influence. By default, the last
        layer of the model is chosen.
    dataset_hessian_size
        An integer for the amount of samples that should go into the computation of the Hessian
        matrix. By default, the entire training dataset is used.
    dtype
        Numeric type for the Krylov basis (backend's float32 by default).
    loss_function
        Loss function to calculate influence. Must not apply any reduction and should match
        the output format of the model (e.g. ``from_logits=True``). When ``None``, a default
        non-reduced categorical cross-entropy is resolved from the model's backend.
    """

    def __init__(
            self,
            subspace_dim: int,
            force_hermitian: bool,
            k_largest_eig_vals: int,
            start_layer: int = -1,
            dataset_hessian_size: int = -1,
            loss_function: Optional[LossFunction] = None,
            dtype: Optional[DType] = None
    ):
        self.subspace_dim = subspace_dim
        self.force_hermitian = force_hermitian
        self.k_largest_eig_vals = k_largest_eig_vals
        self.start_layer = start_layer
        self.loss_function = loss_function
        self.dataset_hessian_size = dataset_hessian_size
        self.dtype = dtype

    def build(self, model: Model, **kwargs: Any) -> ArnoldiInfluenceCalculator:
        """
        Builds an instance of the ArnoldiInfluenceCalculator class following the provided model
        and training dataset. No additional training information is required.

        Parameters
        ----------
        model
            A model (TF or PyTorch) for which to compute the influence-related quantities.
        **kwargs
            training_dataset
                A batched dataset (tf.data.Dataset or PyTorch DataLoader) containing the data
                on which the model was trained.

        Returns
        -------
        influence_calculator
            The desired ArnoldiInfluenceCalculator instance.
        """
        training_dataset = kwargs.get("training_dataset")
        if training_dataset is None:
            raise ValueError("ArnoldiCalculatorFactory.build() requires keyword argument 'training_dataset'.")

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
