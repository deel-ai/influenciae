# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Helpers for customizing the payload carried through influence outputs.

When influence methods return scores for training samples, they can also return
an associated payload describing which training entries those scores correspond
to. For simple classification datasets the payload is usually the model input
itself, but richer tasks may prefer stable sample ids or another batched tensor
that is easier to inspect downstream.

This module defines the public hook used to extract that payload from each
training batch while keeping the influence core agnostic to the dataset's exact
batch structure.

Payload extraction is the training-side counterpart to evaluation representation
providers: evaluation providers decide how query batches are differentiated,
whereas payload extractors decide what training-side value is returned next to
the resulting scores.  Payloads are output metadata only; they do not affect the
influence computation.
"""
from typing import Any, Protocol, Tuple

from ..types import Tensor


class TrainingPayloadExtractor(Protocol):
    """Callable extracting the per-sample payload associated with a train batch.

    The extracted payload is propagated alongside influence scores, especially
    in top-k queries. It should therefore be a batched tensor-like object whose
    leading dimension matches the training batch size.  Public scoring methods
    pass this callable through
    :meth:`~deel.influenciae.common.base_influence.BaseInfluenceCalculator._extract_training_payload`.
    """

    def __call__(self, batch: Tuple[Any, ...]) -> Tensor:
        """Return the payload to keep alongside the batch's influence scores.

        Parameters
        ----------
        batch
            Raw batch from the training dataset. For standard supervised
            datasets this is usually ``(inputs, targets, ...)``. Structured
            datasets and adapters may pass a one-element tuple wrapping a richer
            batch object.

        Returns
        -------
        Tensor
            Batched payload with the same leading batch dimension as ``batch``.
            For ``top_k`` outputs, the payload should have a stable shape and
            dtype across training batches.
        """


def default_training_payload_extractor(batch: Tuple[Any, ...]) -> Tensor:
    """Return ``batch[0]``, preserving the library's historical default.

    This works well for standard supervised datasets where the first batch
    element is the model input tensor. More structured tasks can provide a
    custom extractor to return stable sample identifiers or any other batched
    tensor they want exposed in influence outputs.

    Parameters
    ----------
    batch
        Raw tuple/list batch from the training dataset.

    Returns
    -------
    Tensor
        The first element of ``batch``.

    Raises
    ------
    ValueError
        If ``batch`` is not a tuple or list.
    """
    if not isinstance(batch, (list, tuple)):
        raise ValueError("Training payload extraction expects a batched tuple/list.")
    return batch[0]
