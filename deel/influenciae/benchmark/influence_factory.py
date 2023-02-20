# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing factories for the different influence calculation techniques.
This will be useful for streamlining the benchmarks.
"""
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction, CategoricalCrossentropy  # pylint: disable=E0611

from ..common import InfluenceModel, BaseInfluenceCalculator
from ..common import ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP
from ..common import ExactIHVPFactory, CGDIHVPFactory, LissaIHVPFactory

from ..influence import FirstOrderInfluenceCalculator, ArnoldiInfluenceCalculator
from ..rps import RepresenterPointLJE
from ..trac_in import TracIn
from ..rps import RepresenterPointL2
from ..boundary_based import WeightsBoundaryCalculator, SampleBoundaryCalculator

from ..types import Any, Union, Callable


class InfluenceCalculatorFactory:
    """
    An interface for factories generating instances of the different influence calculators.
    """

    @abstractmethod
    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              train_info: Any) -> BaseInfluenceCalculator:
        """
        Builds an instance of an influence calculator class following the provided model, training dataset
        and additional information.

        Parameters
        ----------
        training_dataset
            A TF dataset with the data on which the model was trained.
        model
            A TF model for which to compute the influence-related quantities.
        train_info
            An object providing additional information about the training procedure. For example, additional
            information is needed for computing influence values using TracIn.

        Returns
        -------
        The desired influence calculator instance.
        """
        raise NotImplementedError


class FirstOrderFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of FirstOrderInfluenceCalculator objects.

    Attributes
    ----------
    ihvp_mode
        A string indicating whether the IHVPs should be computed using the 'exact' or the 'cgd' method.
    start_layer
        An integer for the target layer on which to compute the influence. By default, the last layer of
        the model is chosen.
    dataset_hessian_size
        An integer for the amount of samples that should go into the computation of the hessian matrix. By
        default, the entire training dataset is used.
    n_opt_iters
        An integer indicating how many iterations of the Conjugate Gradient Descent algorithm should be run
        before prematurely stopping the optimization.
    feature_extractor
        Either an integer for the last layer of the feature extractor, or an entire TF graph for computing the
        embeddings of the samples. Used if ihvp_mode == 'cgd'.
    loss_function
        Loss function to calculate influence (e.g. keras CategoricalCrossentropy). Make sure not to
        apply any reduction (Reduction.NONE), and specify correctly if the output is `from_logits`
        for example.
    """

    def __init__(self, ihvp_mode: str, start_layer=-1, dataset_hessian_size=-1, n_opt_iters=100,
                 feature_extractor: Union[int, tf.keras.Model] = -1,
                 loss_function: Callable = tf.keras.losses.CategoricalCrossentropy(
                     from_logits=True, reduction=Reduction.NONE)
                 ):
        self.start_layer = start_layer
        self.ihvp_mode = ihvp_mode
        self.n_opt_iters = n_opt_iters
        self.feature_extractor = feature_extractor
        self.dataset_hessian_size = dataset_hessian_size
        self.loss_function = loss_function
        assert self.ihvp_mode in ['exact', 'cgd', 'lissa']

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              train_info: Any = None) -> FirstOrderInfluenceCalculator:
        """
        Builds an instance of the FirstOrderInfluenceCalculator class following the provided model and
        training dataset. No additional information is required.

        Parameters
        ----------
        training_dataset
            A TF dataset with the data on which the model was trained.
        model
            A TF model for which to compute the influence-related quantities.
        train_info
            In this case, None, as no additional information is required.

        Returns
        -------
        The desired FirstOrderInfluenceCalculator instance.
        """
        influence_model = InfluenceModel(model, start_layer=self.start_layer, loss_function=self.loss_function)

        if self.dataset_hessian_size is None or self.dataset_hessian_size < 0:
            dataset_hessian = training_dataset
        else:
            batch_size = training_dataset._batch_size.numpy()  # pylint: disable=W0212
            take_size = int(
                np.ceil(float(self.dataset_hessian_size) / batch_size)) * batch_size
            dataset_hessian = training_dataset.take(take_size)

        if self.ihvp_mode == 'exact':
            ihvp_calculator = ExactIHVP(influence_model, dataset_hessian)
        elif self.ihvp_mode == 'cgd':
            ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, self.feature_extractor, dataset_hessian,
                                                           self.n_opt_iters)
        elif self.ihvp_mode == 'lissa':
            ihvp_calculator = LissaIHVP(influence_model, self.feature_extractor, dataset_hessian, self.n_opt_iters,
                                        damping=1e-4, scale=5.)
        else:
            raise ValueError("unknown ihvp calculator=" + self.ihvp_mode)

        influence_calculator = FirstOrderInfluenceCalculator(influence_model, training_dataset, ihvp_calculator,
                                                             n_samples_for_hessian=self.dataset_hessian_size)
        return influence_calculator


class RPSLJEFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of representer point LJE objects.

    Attributes
    ----------
    ihvp_mode
        A string indicating whether the IHVPs should be computed using the 'exact' or the 'cgd' method.
    start_layer
        An integer for the target layer on which to compute the influence. By default, the last layer of
        the model is chosen.
    dataset_hessian_size
        An integer for the amount of samples that should go into the computation of the hessian matrix. By
        default, the entire training dataset is used.
    n_opt_iters
        An integer indicating how many iterations of the Conjugate Gradient Descent algorithm should be run
        before prematurely stopping the optimization.
    feature_extractor
        Either an integer for the last layer of the feature extractor, or an entire TF graph for computing the
        embeddings of the samples. Used if ihvp_mode == 'cgd'.
    loss_function
        Loss function to calculate influence (e.g. keras CategoricalCrossentropy). Make sure not to
        apply any reduction (Reduction.NONE), and specify correctly if the output is `from_logits`
        for example.
    """

    def __init__(self, ihvp_mode: str, start_layer=-1, dataset_hessian_size=-1, n_opt_iters=100,
                 feature_extractor: Union[int, tf.keras.Model] = -1,
                 loss_function: Callable = tf.keras.losses.CategoricalCrossentropy(
                     from_logits=True, reduction=Reduction.NONE)
                 ):
        self.start_layer = start_layer
        self.ihvp_mode = ihvp_mode
        self.n_opt_iters = n_opt_iters
        self.feature_extractor = feature_extractor
        self.dataset_hessian_size = dataset_hessian_size
        self.loss_function = loss_function
        assert self.ihvp_mode in ['exact', 'cgd', 'lissa']

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              train_info: Any = None) -> RepresenterPointLJE:
        """
        Builds an instance of the RepresenterPointLJE class following the provided model and training dataset.
        No additional information is required in this case.

        Parameters
        ----------
        training_dataset
            A TF dataset with the data on which the model was trained.
        model
            A TF model for which to compute the influence-related quantities.
        train_info
            None in this case, as no additional information is required

        Returns
        -------
        The desired RepresenterPointLJE instance.
        """
        influence_model = InfluenceModel(model, start_layer=self.start_layer, loss_function=self.loss_function)

        if self.dataset_hessian_size is None or self.dataset_hessian_size < 0:
            dataset_hessian = training_dataset
        else:
            batch_size = training_dataset._batch_size.numpy()  # pylint: disable=W0212
            take_size = int(
                np.ceil(float(self.dataset_hessian_size) / batch_size)) * batch_size
            dataset_hessian = training_dataset.take(take_size)

        if self.ihvp_mode == 'exact':
            ihvp_calculator_factory = ExactIHVPFactory()
        elif self.ihvp_mode == 'cgd':
            ihvp_calculator_factory = CGDIHVPFactory(self.feature_extractor, self.n_opt_iters)
        elif self.ihvp_mode == 'lissa':
            ihvp_calculator_factory = LissaIHVPFactory(self.feature_extractor, self.n_opt_iters, damping=1e-4, scale=5.)
        else:
            raise ValueError("unknown ihvp calculator=" + self.ihvp_mode)

        influence_calculator = RepresenterPointLJE(influence_model, dataset_hessian, ihvp_calculator_factory)
        return influence_calculator


class TracInFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of TracIn objects.
    As it works by tracking the gradients along the training process, it also requires some
    training information to be able to compute influence values.

    Attributes
    ----------
    loss_function
        Loss function to calculate influence (e.g. keras CategoricalCrossentropy). Make sure not to
        apply any reduction (Reduction.NONE), and specify correctly if the output is `from_logits`
        for example.
    """

    def __init__(self, loss_function: Callable = CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)):
        self.loss_function = loss_function

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              train_info: Any) -> TracIn:
        """
        Builds an instance of the TracIn class following the provided model, training dataset
        and additional information.

        Parameters
        ----------
        training_dataset
            A TF dataset with the data on which the model was trained.
        model
            A TF model for which to compute the influence-related quantities.
        train_info
            A tuple with a list with the model's checkpoints on the first element and the corresponding
            list of learning rates on the second element.

        Returns
        -------
        The desired TracIn instance.
        """
        models = []
        for model_data in train_info[0]:
            influence_model = InfluenceModel(model_data, loss_function=self.loss_function)
            models.append(influence_model)
        influence_calculator = TracIn(models, train_info[1])
        return influence_calculator


class RPSL2Factory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of RepresenterPointL2 objects.

    Attributes
    ----------
    loss_function
        The loss function with which the model was trained. This loss function MUST NOT be reduced.
    lambda_regularization
        The strength of the L2 regularization to add to the surrogate last layer.
    scaling_factor
        The Backtracking line-search's scaling factor for training the surrogate last layer. By default, this
        value is set to 0.1 and should typically converge quite easily.
    layer_index
        The index for the layer on which to compute the influence values.
    epochs
        An integer indicating for how long the surrogate last layer should be trained. By default, a value of
        100 is chosen.
    """

    def __init__(
            self,
            loss_function: Union[Callable[[tf.Tensor, tf.Tensor], tf.Tensor], Loss],
            lambda_regularization: float,
            scaling_factor: float = 0.1,
            layer_index=-2,
            epochs: int = 100
    ):
        self.loss_function = loss_function
        self.lambda_regularization = lambda_regularization
        self.scaling_factor = scaling_factor
        self.epochs = epochs
        self.layer_index = layer_index

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              train_info: Any = None) -> RepresenterPointL2:
        """
        Builds an instance of the RepresenterPointL2 class following the provided model and training dataset.
        No additional information is required in this case.

        Parameters
        ----------
        training_dataset
            A TF dataset with the data on which the model was trained.
        model
            A TF model for which to compute the influence-related quantities.
        train_info
            None, as no additional information is required in this case.

        Returns
        -------
        The desired RepresenterPointL2 instance.
        """
        influence_calculator = RepresenterPointL2(model,
                                                  training_dataset,
                                                  self.loss_function,
                                                  self.lambda_regularization,
                                                  self.scaling_factor,
                                                  self.epochs,
                                                  self.layer_index)
        return influence_calculator


class WeightsBoundaryCalculatorFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of WeightsBoundaryCalculator objects.

    Attributes
    ----------
    step_nbr
        The number of iterations to search the boundary for. By default, a value of 100 is chosen.
    norm_type
        The norm type of the distance between the weights to measure the distance to the boundary.
        By default, the L2 distance is chosen.
    """
    def __init__(self, step_nbr: int = 100, norm_type: int = 2):
        self.step_nbr = step_nbr
        self.norm_type = norm_type

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              train_info: Any = None) -> WeightsBoundaryCalculator:
        """
        Builds an instance of the WeightsBoundaryCalculator class following the provided model and training dataset.
        No additional information is required in this case.

        Parameters
        ----------
        training_dataset
            A TF dataset with the data on which the model was trained.
        model
            A TF model for which to compute the influence-related quantities.
        train_info
            None, as no additional information is required in this case.

        Returns
        -------
        The desired WeightsBoundaryCalculator instance.
        """
        return WeightsBoundaryCalculator(model, self.step_nbr, self.norm_type)


class SampleBoundaryCalculatorFactory(InfluenceCalculatorFactory):
    """
    A factory for creating instances of SampleBoundaryCalculator objects.

    Attributes
    ----------
    step_nbr
        The number of iterations to search the boundary for. By default, a value of 100 is chosen.
    """

    def __init__(self, step_nbr: int = 100):
        self.step_nbr = step_nbr

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              train_info: Any = None) -> SampleBoundaryCalculator:
        """
        Builds an instance of the SampleBoundaryCalculator class following the provided model and training dataset.
        No additional information is required in this case.

        Parameters
        ----------
        training_dataset
            A TF dataset with the data on which the model was trained.
        model
            A TF model for which to compute the influence-related quantities.
        train_info
            None, as no additional information is required in this case.

        Returns
        -------
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
        A boolean indicating if we should force the projected matrix to be hermitian before the eigenvalue computation.
    k_largest_eig_vals
        An integer for the amount of top eigenvalues to keep for the influence estimations.
    start_layer
        An integer for the target layer on which to compute the influence. By default, the last layer of
        the model is chosen.
    dataset_hessian_size
        An integer for the amount of samples that should go into the computation of the hessian matrix. By
        default, the entire training dataset is used.
    dtype
        Numeric type for the Krylov basis (tf.float32 by default).
    """
    def __init__(
            self,
            subspace_dim: int,
            force_hermitian: bool,
            k_largest_eig_vals: int,
            start_layer: int = -1,
            dataset_hessian_size: int = -1,
            loss_function: Callable = CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE),
            dtype: tf.dtypes = tf.float32
    ):
        self.subspace_dim = subspace_dim
        self.force_hermitian = force_hermitian
        self.k_largest_eig_vals = k_largest_eig_vals
        self.start_layer = start_layer
        self.loss_function = loss_function
        self.dataset_hessian_size = dataset_hessian_size
        self.dtype = dtype

    def build(self, training_dataset: tf.data.Dataset, model: tf.keras.Model,
              train_info: Any = None) -> ArnoldiInfluenceCalculator:
        """
        Builds an instance of the ArnoldiInfluenceCalculator class following the provided model and training dataset.
        No additional information is required in this case.

        Parameters
        ----------
        training_dataset
            A TF dataset with the data on which the model was trained.
        model
            A TF model for which to compute the influence-related quantities.
        train_info
            None, as no additional information is required in this case.

        Returns
        -------
        The desired ArnoldiInfluenceCalculator instance.
        """
        influence_model = InfluenceModel(model, start_layer=self.start_layer, loss_function=self.loss_function)

        if self.dataset_hessian_size is None or self.dataset_hessian_size < 0:
            dataset_hessian = training_dataset
        else:
            batch_size = training_dataset._batch_size.numpy()  # pylint: disable=W0212
            take_size = int(np.ceil(float(self.dataset_hessian_size) / batch_size)) * batch_size
            dataset_hessian = training_dataset.take(take_size)

        return ArnoldiInfluenceCalculator(influence_model, dataset_hessian, self.subspace_dim, self.force_hermitian,
                                          self.k_largest_eig_vals, self.dtype)
