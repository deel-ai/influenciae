# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Benchmark module
"""

from .base_benchmark import BaseTrainingProcedure, MislabelingDetectorEvaluator
from .influence_factory import InfluenceCalculatorFactory, FirstOrderFactory, RPSLJEFactory, TracInFactory
from .cifar10_benchmark import Cifar10TrainingProcedure, Cifar10MislabelingDetectorEvaluator
