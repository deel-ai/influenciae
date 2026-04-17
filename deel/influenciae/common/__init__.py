# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Common classes and methods
"""

import importlib

from .backend import (
    Framework,
    BaseBackend,
    detect_framework,
    detect_tensor_framework,
    detect_dtype_framework,
    get_backend,
    get_backend_for_model,
    get_backend_for_tensor,
    get_available_frameworks,
)
from .model_wrappers import (
    BaseInfluenceModel,
    InfluenceModel,
    TensorFlowInfluenceModel,
    PyTorchInfluenceModel,
    default_process_batch,
)
from .evaluation import EvaluationRepresentationProvider, ObjectiveEvaluationRepresentationProvider
from .payloads import TrainingPayloadExtractor, default_training_payload_extractor

_LAZY_ATTRS = {
    "SelfInfluenceCalculator": ".base_influence",
    "BaseInfluenceCalculator": ".base_influence",
    "CACHE": ".base_influence",
    "InverseHessianVectorProduct": ".inverse_hessian_vector_product",
    "ExactIHVP": ".inverse_hessian_vector_product",
    "ConjugateGradientDescentIHVP": ".inverse_hessian_vector_product",
    "IHVPCalculator": ".inverse_hessian_vector_product",
    "LissaIHVP": ".inverse_hessian_vector_product",
    "ForwardOverBackwardHVP": ".inverse_hessian_vector_product",
    "KfacIHVP": ".inverse_hessian_vector_product",
    "EkfacIHVP": ".inverse_hessian_vector_product",
    "InverseHessianVectorProductFactory": ".ihvp_factory",
    "ExactIHVPFactory": ".ihvp_factory",
    "CGDIHVPFactory": ".ihvp_factory",
    "LissaIHVPFactory": ".ihvp_factory",
    "KfacIHVPFactory": ".ihvp_factory",
    "EkfacIHVPFactory": ".ihvp_factory",
    "LayerParameterMap": ".kfac_factors",
    "KroneckerFactors": ".kfac_factors",
    "EKFACFactors": ".kfac_factors",
}


def __getattr__(name):
    """Lazily import heavier common modules to avoid circular imports."""
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_LAZY_ATTRS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    """Expose lazily loaded attributes in interactive environments."""
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
