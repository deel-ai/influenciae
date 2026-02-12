# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Benchmark module
"""

from typing import TYPE_CHECKING

from .._optional_imports import import_optional_attr
from .base_benchmark import BaseTrainingProcedure, MislabelingDetectorEvaluator
from .influence_factory import (
    InfluenceCalculatorFactory,
    FirstOrderFactory,
    RPSLJEFactory,
    TracInFactory,
    WeightsBoundaryCalculatorFactory,
    SampleBoundaryCalculatorFactory,
    ArnoldiCalculatorFactory,
)

__all__ = [
    "BaseTrainingProcedure",
    "MislabelingDetectorEvaluator",
    "InfluenceCalculatorFactory",
    "FirstOrderFactory",
    "RPSLJEFactory",
    "TracInFactory",
    "WeightsBoundaryCalculatorFactory",
    "SampleBoundaryCalculatorFactory",
    "ArnoldiCalculatorFactory",
    "Cifar10TrainingProcedure",
    "Cifar10MislabelingDetectorEvaluator",
]


def __getattr__(name):
    """Lazy import TensorFlow-specific benchmark helpers."""
    if name in {"Cifar10TrainingProcedure", "Cifar10MislabelingDetectorEvaluator"}:
        attr = import_optional_attr(
            ".cifar10_benchmark",
            name,
            package=__package__,
            extra="tensorflow",
        )
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .cifar10_benchmark import Cifar10TrainingProcedure, Cifar10MislabelingDetectorEvaluator
