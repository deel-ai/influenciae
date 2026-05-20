# tests/common/test_kfac_shared_layers_pytorch.py
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Regression tests for K-FAC on reused PyTorch layers."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.common import InfluenceModel, KfacIHVP, EkfacIHVP


pytestmark = pytest.mark.pytorch


class _SharedConvModel(nn.Module):
    """Use one Conv2d module twice with different spatial sizes."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, bias=True, dtype=torch.float64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2, bias=False, dtype=torch.float64)

    def forward(self, x):
        big = F.relu(self.conv(x))
        small_input = F.interpolate(x, size=(6, 6), mode="nearest")
        small = F.relu(self.conv(small_input))

        features = torch.cat([
            self.pool(big).flatten(1),
            self.pool(small).flatten(1),
        ], dim=1)
        return self.fc(features)


def _make_shared_conv_case(seed=42, batch_size=3):
    torch.manual_seed(seed)
    model = _SharedConvModel()
    gen = torch.Generator().manual_seed(seed + 1)
    inputs = torch.randn(batch_size, 3, 10, 10, generator=gen, dtype=torch.float64)
    targets = torch.randn(batch_size, 2, generator=gen, dtype=torch.float64)
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)
    return model, train_loader, batch_size


def _make_influence_model(model):
    return InfluenceModel(
        model,
        start_layer=0,
        last_layer=-1,
        loss_function=nn.MSELoss(reduction="none"),
    )


def _shared_conv_info(ihvp, model):
    for info in ihvp.layer_map.layers_info:
        if info.layer is model.conv:
            return info
    raise AssertionError("Shared convolution was not collected by K-FAC.")


def _expected_shared_conv_rows(batch_size):
    return batch_size * (8 * 8 + 4 * 4)


def test_kfac_shared_layer_no_mismatch_error():
    model, train_loader, _ = _make_shared_conv_case()
    influence_model = _make_influence_model(model)

    kfac = KfacIHVP(influence_model, train_loader, damping=1e-3)
    info = _shared_conv_info(kfac, model)

    assert info.layer_idx in kfac.factors.A
    assert info.layer_idx in kfac.factors.G


def test_kfac_shared_layer_factors_accumulate_all_calls():
    model, train_loader, batch_size = _make_shared_conv_case()
    influence_model = _make_influence_model(model)

    kfac = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        data_partition_size=1,
        accumulator_offload_mode="memory",
    )
    info = _shared_conv_info(kfac, model)

    checkpoint = kfac.factors._factor_checkpoint
    assert checkpoint is not None
    assert checkpoint["n_rows_per_layer"][info.layer_idx] == _expected_shared_conv_rows(batch_size)


def test_ekfac_shared_layer_no_mismatch_error():
    model, train_loader, batch_size = _make_shared_conv_case()
    influence_model = _make_influence_model(model)

    ekfac = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        n_ekfac_samples=None,
        data_partition_size=1,
        accumulator_offload_mode="memory",
    )
    info = _shared_conv_info(ekfac, model)

    assert info.layer_idx in ekfac.factors.Lambda_corrected
    corrected_checkpoint = ekfac.factors._corrected_checkpoint
    assert corrected_checkpoint is not None
    assert corrected_checkpoint["n_rows_per_layer"][info.layer_idx] == _expected_shared_conv_rows(batch_size)
