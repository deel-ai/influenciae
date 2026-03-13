# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module defining the interface and classes that implement factories for objects of
class InverseHessianVectorProduct. This will prove itself useful for creating
the objects necessary for computing the different (I)HVPs in second order influence
functions.
"""
from abc import abstractmethod

from .model_wrappers import InfluenceModel
from .inverse_hessian_vector_product import (
    InverseHessianVectorProduct,
    ExactIHVP,
    ConjugateGradientDescentIHVP,
    LissaIHVP,
    KfacIHVP,
    EkfacIHVP,
)

from ..types import Union, Optional, Any


class InverseHessianVectorProductFactory:
    """
    The base interface for InverseHessianVectorProduct factories.
    """
    @abstractmethod
    def build(self, model_influence: InfluenceModel, dataset: Any) -> InverseHessianVectorProduct:
        """
        Creates an instance of an InverseHessianVectorProduct class with the provided
        parameters.

        Parameters
        ----------
        model_influence
            A model implementing the InfluenceModel interface.
        dataset
            A batched dataset (tf.data.Dataset or PyTorch DataLoader) containing
            the model's (full or partial) training dataset.

        Returns
        -------
        ihvp
            An instance of an InverseHessianVectorProduct class.
        """
        raise NotImplementedError()


class ExactIHVPFactory(InverseHessianVectorProductFactory):
    """
    A factory for instantiating ExactIHVP objects.
    """
    def build(self, model_influence: InfluenceModel, dataset: Any) -> InverseHessianVectorProduct:
        """
        Creates an instance of the ExactIHVP class for a given model
        implementing the InfluenceModel interface and its (full or partial) training dataset.

        Parameters
        ----------
        model_influence
            A model implementing the InfluenceModel interface.
        dataset
            A batched dataset (tf.data.Dataset or PyTorch DataLoader) containing
            the model's (full or partial) training dataset.

        Returns
        -------
        exact_ihvp
            An instance of the ExactIHVP class.
        """
        return ExactIHVP(model_influence, dataset)


class CGDIHVPFactory(InverseHessianVectorProductFactory):
    """
    A factory for instantiating ConjugateGradientDescentIHVP objects.

    Attributes
    ----------
    feature_extractor
        Either a feature-extractor model (TF or PyTorch) or the index of the layer of a
        whole model which will be cut into two for computing the influence vectors and scores.
    n_cgd_iters
        An integer specifying the amount of iterations of the optimizer to run before
        (prematurely) considering the optimization completed.
    extractor_layer
        The cutoff layer for the feature extractor, if specified in model format.
    """
    def __init__(
        self,
        feature_extractor: Union[int, Any] = -1,
        n_cgd_iters: int = 100,
        extractor_layer: Optional[Union[str, int]] = None
    ):
        self.n_cgd_iters = n_cgd_iters
        if isinstance(feature_extractor, int):
            self.extractor_layer: Union[str, int] = feature_extractor
            self.feature_extractor: Optional[Any] = None
        else:
            assert extractor_layer is not None, "If you provide a model as a feature extractor, you should also" \
                                                "provide the id of the last extracted layer"
            self.extractor_layer = extractor_layer
            self.feature_extractor = feature_extractor

    def build(self, model_influence: InfluenceModel, dataset: Any) -> InverseHessianVectorProduct:
        """
        Creates an instance of the ConjugateGradientDescentIHVP class for the provided model and its
        corresponding (full or partial) training dataset.

        Parameters
        ----------
        model_influence
            A model implementing the InfluenceModel interface.
        dataset
            A batched dataset (tf.data.Dataset or PyTorch DataLoader) containing
            the model's (full or partial) training dataset.

        Returns
        -------
        cgd_ihvp
            An instance of the ConjugateGradientDescentIHVP class
        """
        return ConjugateGradientDescentIHVP(
            model_influence,
            self.extractor_layer,
            dataset,
            self.n_cgd_iters,
            self.feature_extractor,
        )


class LissaIHVPFactory(InverseHessianVectorProductFactory):
    """
    A factory for instantiating LissaIHVP objects.

    Attributes
    ----------
    feature_extractor
        Either a feature-extractor model (TF or PyTorch) or the index of the layer of a
        whole model which will be cut into two for computing the influence vectors and scores.
    n_cgd_iters
        An integer specifying the amount of iterations of the optimizer to run before
        (prematurely) considering the optimization completed.
    extractor_layer
        The cutoff layer for the feature extractor, if specified in model format.
    damping
        A damping parameter to regularize a nearly singular operator.
    scale
        A rescaling factor to verify the hypothesis of norm(operator / scale) < 1.
    """
    def __init__(
        self,
        feature_extractor: Union[int, Any] = -1,
        n_cgd_iters: int = 100,
        extractor_layer: Optional[Union[str, int]] = None,
        damping: float = 1e-4,
        scale: float = 10.
    ):
        self.n_cgd_iters = n_cgd_iters
        self.damping = damping
        self.scale = scale
        if isinstance(feature_extractor, int):
            self.extractor_layer: Union[str, int] = feature_extractor
            self.feature_extractor: Optional[Any] = None
        else:
            assert extractor_layer is not None, "If you provide a model as a feature extractor, you should also" \
                                                "provide the id of the last extracted layer"
            self.extractor_layer = extractor_layer
            self.feature_extractor = feature_extractor

    def build(self, model_influence: InfluenceModel, dataset: Any) -> InverseHessianVectorProduct:
        """
        Creates an instance of the LissaIHVP class for the provided model and its
        corresponding (full or partial) training dataset.

        Parameters
        ----------
        model_influence
            A model implementing the InfluenceModel interface.
        dataset
            A batched dataset (tf.data.Dataset or PyTorch DataLoader) containing
            the model's (full or partial) training dataset.

        Returns
        -------
        lissa_ihvp
            An instance of the LissaIHVP class.
        """
        return LissaIHVP(
            model_influence,
            self.extractor_layer,
            dataset,
            self.n_cgd_iters,
            self.feature_extractor,
            self.damping,
            self.scale
        )


class KfacIHVPFactory(InverseHessianVectorProductFactory):
    """
    A factory for instantiating KfacIHVP objects.

    Attributes
    ----------
    damping
        Tikhonov damping added to the Kronecker factors before inversion.
    target_layers
        Optional list of layer indices to restrict K-FAC to.
    fisher_type
        Fisher variant used for curvature estimation: ``"empirical"`` (default)
        or ``"true"``.
    module_partition_size
        Optional number of supported layers to process per pass while computing
        factors.
    offload_activations_to_cpu
        Whether to offload hook-captured activations/gradients to CPU.
    data_partition_size
        Optional number of batches per data partition during factor estimation.
    layer_collection
        Layer traversal mode for K-FAC mapping: ``"top_level"`` (default)
        or ``"recursive"``.
    """
    def __init__(
        self,
        damping: float = 1e-4,
        target_layers: Optional[list] = None,
        fisher_type: str = "empirical",
        module_partition_size: Optional[int] = None,
        offload_activations_to_cpu: bool = False,
        data_partition_size: Optional[int] = None,
        layer_collection: str = "top_level",
    ):
        self.damping = damping
        self.target_layers = target_layers
        self.fisher_type = fisher_type
        self.module_partition_size = module_partition_size
        self.offload_activations_to_cpu = offload_activations_to_cpu
        self.data_partition_size = data_partition_size
        self.layer_collection = layer_collection

    def build(self, model_influence: InfluenceModel, dataset: Any) -> InverseHessianVectorProduct:
        """
        Creates an instance of the KfacIHVP class for the provided model and its
        corresponding (full or partial) training dataset.

        Parameters
        ----------
        model_influence
            A model implementing the InfluenceModel interface.
        dataset
            A batched dataset containing the model's (full or partial) training dataset.

        Returns
        -------
        kfac_ihvp
            An instance of the KfacIHVP class.
        """
        return KfacIHVP(
            model_influence,
            dataset,
            damping=self.damping,
            target_layers=self.target_layers,
            layer_collection=self.layer_collection,
            fisher_type=self.fisher_type,
            module_partition_size=self.module_partition_size,
            offload_activations_to_cpu=self.offload_activations_to_cpu,
            data_partition_size=self.data_partition_size,
        )


class EkfacIHVPFactory(InverseHessianVectorProductFactory):
    """
    A factory for instantiating EkfacIHVP objects.

    Attributes
    ----------
    damping
        Tikhonov damping added to the corrected eigenvalues before inversion.
    target_layers
        Optional list of layer indices to restrict EK-FAC to.
    n_ekfac_samples
        Number of samples for corrected eigenvalue estimation.
    fisher_type
        Fisher variant used for curvature estimation: ``"empirical"`` (default)
        or ``"true"``.
    module_partition_size
        Optional number of supported layers to process per pass while computing
        factors.
    offload_activations_to_cpu
        Whether to offload hook-captured activations/gradients to CPU.
    data_partition_size
        Optional number of batches per data partition during factor estimation.
    layer_collection
        Layer traversal mode for EK-FAC mapping: ``"top_level"`` (default)
        or ``"recursive"``.
    """
    def __init__(
        self,
        damping: float = 1e-4,
        target_layers: Optional[list] = None,
        n_ekfac_samples: Optional[int] = None,
        fisher_type: str = "empirical",
        module_partition_size: Optional[int] = None,
        offload_activations_to_cpu: bool = False,
        data_partition_size: Optional[int] = None,
        layer_collection: str = "top_level",
    ):
        self.damping = damping
        self.target_layers = target_layers
        self.n_ekfac_samples = n_ekfac_samples
        self.fisher_type = fisher_type
        self.module_partition_size = module_partition_size
        self.offload_activations_to_cpu = offload_activations_to_cpu
        self.data_partition_size = data_partition_size
        self.layer_collection = layer_collection

    def build(self, model_influence: InfluenceModel, dataset: Any) -> InverseHessianVectorProduct:
        """
        Creates an instance of the EkfacIHVP class for the provided model and its
        corresponding (full or partial) training dataset.

        Parameters
        ----------
        model_influence
            A model implementing the InfluenceModel interface.
        dataset
            A batched dataset containing the model's (full or partial) training dataset.

        Returns
        -------
        ekfac_ihvp
            An instance of the EkfacIHVP class.
        """
        return EkfacIHVP(
            model_influence,
            dataset,
            damping=self.damping,
            target_layers=self.target_layers,
            layer_collection=self.layer_collection,
            n_ekfac_samples=self.n_ekfac_samples,
            fisher_type=self.fisher_type,
            module_partition_size=self.module_partition_size,
            offload_activations_to_cpu=self.offload_activations_to_cpu,
            data_partition_size=self.data_partition_size,
        )
