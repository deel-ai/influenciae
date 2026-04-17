# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for layer selection utilities with the PyTorch backend.
"""
import pytest
import torch.nn as nn

from deel.influenciae.common import get_backend_for_model
from deel.influenciae.utils.layer_selection import resolve_layer_selection


pytestmark = pytest.mark.pytorch


def test_resolve_layer_selection_keeps_original_recursive_indices():
    """Resolved layer indices should match recursive KFAC indexing semantics."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.Sequential(nn.ReLU(), nn.Linear(4, 2)),
    )

    backend = get_backend_for_model(model)
    resolved = resolve_layer_selection(
        model,
        backend,
        selector=lambda layer_idx, layer_name, layer: "1.1" in layer_name or layer_name == "1.1",
        layer_collection="recursive",
        supported_only=True,
    )

    assert resolved.layer_names == ["1.1"]
    assert resolved.layer_indices == [3]
    assert len(resolved.layers) == 1


def test_resolve_layer_selection_warns_when_no_layers_match():
    """Empty layer selections should warn instead of silently succeeding."""
    model = nn.Sequential(nn.ReLU())

    backend = get_backend_for_model(model)
    with pytest.warns(RuntimeWarning, match="No layers matched selector"):
        resolved = resolve_layer_selection(
            model,
            backend,
            selector="missing",
            layer_collection="recursive",
            supported_only=True,
        )

    assert resolved.layer_indices == []
    assert resolved.layer_names == []
    assert resolved.layers == []
