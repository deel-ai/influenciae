# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Benchmark module
"""

from typing import TYPE_CHECKING

from .base_benchmark import BaseTrainingProcedure, MislabelingDetectorEvaluator
from .influence_factory import (
    InfluenceCalculatorFactory,
    FirstOrderFactory,
    RPSLJEFactory,
    TracInFactory,
    WeightsBoundaryCalculatorFactory,
    SampleBoundaryCalculatorFactory,
    ArnoldiCalculatorFactory
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
        try:
            from .cifar10_benchmark import Cifar10TrainingProcedure, Cifar10MislabelingDetectorEvaluator
        except ImportError as exc:
            raise ImportError(
                "CIFAR-10 benchmark helpers require TensorFlow. "
                "Install with: pip install influenciae[tensorflow]"
            ) from exc

        exports = {
            "Cifar10TrainingProcedure": Cifar10TrainingProcedure,
            "Cifar10MislabelingDetectorEvaluator": Cifar10MislabelingDetectorEvaluator,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .cifar10_benchmark import Cifar10TrainingProcedure, Cifar10MislabelingDetectorEvaluator
