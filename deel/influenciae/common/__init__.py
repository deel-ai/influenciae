# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Common classes and methods shared across all influence calculator families.

Eagerly imported
~~~~~~~~~~~~~~~~
- :class:`BaseBackend`, :func:`get_backend`, :func:`detect_framework`, and
  other backend utilities from :mod:`.backend`.
- :class:`BaseInfluenceModel`, :class:`InfluenceModel`,
  :class:`TensorFlowInfluenceModel`, :class:`PyTorchInfluenceModel`, and
  :func:`default_process_batch` from :mod:`.model_wrappers`.
- :class:`EvaluationRepresentationProvider` and
  :class:`ObjectiveEvaluationRepresentationProvider` from
  :mod:`.evaluation` -- the hook for injecting custom evaluation
  representations (e.g. for object detection).
- :class:`TrainingPayloadExtractor` and
  :func:`default_training_payload_extractor` from :mod:`.payloads` -- the hook
  for customizing the training-side payload returned with influence scores.

Lazily imported
~~~~~~~~~~~~~~~
- :class:`BaseInfluenceCalculator`, :class:`SelfInfluenceCalculator` from
  :mod:`.base_influence`.
- IHVP implementations and factories (``ExactIHVP``, ``KfacIHVP``,
  ``EkfacIHVP``, ``ConjugateGradientDescentIHVP``, ``LissaIHVP``,
  ``ForwardOverBackwardHVP``, ``ExactIHVPFactory``, ``KfacIHVPFactory``,
  ``EkfacIHVPFactory``, etc.) from :mod:`.inverse_hessian_vector_product`
  and :mod:`.ihvp_factory`.
- K-FAC / EK-FAC data structures (``LayerParameterMap``,
  ``KroneckerFactors``, ``EKFACFactors``) from :mod:`.kfac_factors`.
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
from .parameter_layout import ParameterLayout, ParameterLayoutEntry, build_parameter_layout
from .representation_store import (
    DirectoryRepresentationStore,
    MemoryRepresentationStore,
    RepresentationShard,
    RepresentationStore,
)

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
    "AstraConfig": ".inverse_hessian_vector_product",
    "AstraIHVP": ".inverse_hessian_vector_product",
    "InverseHessianVectorProductFactory": ".ihvp_factory",
    "ExactIHVPFactory": ".ihvp_factory",
    "CGDIHVPFactory": ".ihvp_factory",
    "LissaIHVPFactory": ".ihvp_factory",
    "KfacIHVPFactory": ".ihvp_factory",
    "EkfacIHVPFactory": ".ihvp_factory",
    "AstraIHVPFactory": ".ihvp_factory",
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
