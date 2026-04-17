# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Helpers for customizing payloads returned by influence queries.
"""
from typing import Any, Protocol, Tuple

from ..types import Tensor


class TrainingPayloadExtractor(Protocol):
    """Protocol for extracting a batched payload from a training batch."""

    def __call__(self, batch: Tuple[Any, ...]) -> Tensor:
        """Return a tensor payload with a leading batch dimension."""


def default_training_payload_extractor(batch: Tuple[Any, ...]) -> Tensor:
    """Return the first batch element, preserving the legacy behavior."""
    if not isinstance(batch, (list, tuple)):
        raise ValueError("Training payload extraction expects a batched tuple/list.")
    return batch[0]
