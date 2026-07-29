# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Strict optimizer-state snapshots used by TrackStar correction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ..common.model_wrappers import BaseInfluenceModel


@dataclass(frozen=True)
class OptimizerSecondMoments:
    """Immutable raw Adam second moments in canonical parameter order."""

    parameter_second_moments: Tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        snapshots = []
        for index, moment in enumerate(self.parameter_second_moments):
            array = np.array(moment, copy=True)
            if array.dtype.kind not in "iuf":
                raise TypeError(f"Second moment {index} must have a real numeric dtype.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Second moment {index} contains non-finite values.")
            if np.any(array < 0):
                raise ValueError(f"Second moment {index} contains negative values.")
            array.setflags(write=False)
            snapshots.append(array)
        object.__setattr__(self, "parameter_second_moments", tuple(snapshots))


def extract_optimizer_second_moments(
    model: BaseInfluenceModel,
    optimizer: Any,
) -> OptimizerSecondMoments:
    """Snapshot raw Adam moments aligned exactly with ``model.parameter_layout``."""
    if not model.weights:
        raise ValueError("Cannot extract optimizer state for an empty parameter layout.")
    if len({id(weight) for weight in model.weights}) != len(model.weights):
        raise ValueError("The canonical parameter list contains duplicate parameter objects.")

    moment_tensors = model.backend.get_optimizer_second_moment_tensors(optimizer, model.weights)
    if len(moment_tensors) != len(model.weights):
        raise ValueError("Backend optimizer state does not match the canonical parameter list.")

    moments = []
    for index, (moment_tensor, expected_shape) in enumerate(
        zip(moment_tensors, model.parameter_layout.parameter_shapes)
    ):
        moment = model.backend.to_numpy(moment_tensor)
        if moment.shape != expected_shape:
            raise ValueError(
                f"Second moment {index} must have shape {expected_shape}; got {moment.shape}."
            )
        moments.append(moment)
    return OptimizerSecondMoments(tuple(moments))
