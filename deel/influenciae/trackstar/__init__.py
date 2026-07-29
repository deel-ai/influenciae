# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""TrackStar attribution with structured projection and projected curvature."""

from .index import ExactStreamingVectorIndex, VectorSearchResult
from .optimizer_state import OptimizerSecondMoments, extract_optimizer_second_moments
from .projected_curvature import (
    AutoMix,
    FixedMix,
    GramSnapshot,
    MixRule,
    ProjectedCurvature,
    ProjectedGramAccumulator,
    build_projected_curvature,
    inverse_sqrt_psd,
)
from .projection import ProjectionBlock, ProjectionTerm, TrackStarProjectionPlan
from .trackstar_influence_calculator import (
    TrackStarBuilder,
    TrackStarInfluenceCalculator,
    TrackStarTopKBatch,
)

__all__ = [
    "AutoMix",
    "ExactStreamingVectorIndex",
    "FixedMix",
    "GramSnapshot",
    "MixRule",
    "OptimizerSecondMoments",
    "ProjectedCurvature",
    "ProjectedGramAccumulator",
    "ProjectionBlock",
    "ProjectionTerm",
    "TrackStarBuilder",
    "TrackStarInfluenceCalculator",
    "TrackStarProjectionPlan",
    "TrackStarTopKBatch",
    "VectorSearchResult",
    "build_projected_curvature",
    "extract_optimizer_second_moments",
    "inverse_sqrt_psd",
]
