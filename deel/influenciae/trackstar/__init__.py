# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""TrackStar attribution with structured projection and projected curvature."""

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

__all__ = [
    "AutoMix",
    "FixedMix",
    "GramSnapshot",
    "MixRule",
    "OptimizerSecondMoments",
    "ProjectedCurvature",
    "ProjectedGramAccumulator",
    "ProjectionBlock",
    "ProjectionTerm",
    "TrackStarProjectionPlan",
    "build_projected_curvature",
    "extract_optimizer_second_moments",
    "inverse_sqrt_psd",
]
