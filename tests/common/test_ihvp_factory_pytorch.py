# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the IHVP factory with PyTorch backend.
These tests verify the PyTorch-specific factory functionality works correctly.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils_test import max_abs_almost_equal as almost_equal

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP, KfacIHVP, EkfacIHVP
from deel.influenciae.common import (
    InverseHessianVectorProductFactory,
    ExactIHVPFactory,
    CGDIHVPFactory,
    LissaIHVPFactory,
    KfacIHVPFactory,
    EkfacIHVPFactory,
)


pytestmark = pytest.mark.pytorch


def test_exact_factory():
    """Test that ExactIHVPFactory produces equivalent ExactIHVP instances."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.Linear(2, 1, bias=False)
    )

    loss_fn = nn.MSELoss(reduction="none")

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    inputs = torch.randn((25, 1, 3))
    targets = torch.randn((25, 1, 1))
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=False)

    ihvp = ExactIHVP(influence_model, train_loader)
    exact_factory = ExactIHVPFactory()
    assert isinstance(exact_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = exact_factory.build(influence_model, train_loader)
    assert isinstance(ihvp_from_factory, ExactIHVP)

    assert almost_equal(ihvp.inv_hessian, ihvp_from_factory.inv_hessian, epsilon=1e-3)


def test_cgd_factory():
    """Test that CGDIHVPFactory produces equivalent ConjugateGradientDescentIHVP instances."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.Linear(2, 1, bias=False)
    )

    loss_fn = nn.MSELoss(reduction="none")

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    inputs = torch.randn((25, 1, 3))
    targets = torch.randn((25, 1, 1))
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=False)

    n_cgd_iters = 100

    # case 1: model feature extractor
    feature_extractor = nn.Sequential(model[0])
    ihvp = ConjugateGradientDescentIHVP(influence_model, 1, train_loader, n_cgd_iters, feature_extractor)
    cgd_factory = CGDIHVPFactory(feature_extractor, n_cgd_iters, 1)
    assert isinstance(cgd_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = cgd_factory.build(influence_model, train_loader)
    assert ihvp.extractor_layer == ihvp_from_factory.extractor_layer
    assert len(list(ihvp.feature_extractor.children())) == len(list(ihvp_from_factory.feature_extractor.children()))
    assert ihvp.weights == ihvp_from_factory.weights
    for ihvp_batch, factory_batch in zip(ihvp.train_set, ihvp_from_factory.train_set):
        assert almost_equal(ihvp_batch[0], factory_batch[0])
        assert almost_equal(ihvp_batch[1], factory_batch[1])

    # case 2: layer position feature extractor
    feature_extractor = 1
    ihvp = ConjugateGradientDescentIHVP(influence_model, 1, train_loader, n_cgd_iters, None)
    cgd_factory = CGDIHVPFactory(feature_extractor, n_cgd_iters)
    assert isinstance(cgd_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = cgd_factory.build(influence_model, train_loader)
    assert ihvp.extractor_layer == ihvp_from_factory.extractor_layer
    assert len(list(ihvp.feature_extractor.children())) == len(list(ihvp_from_factory.feature_extractor.children()))
    assert ihvp.weights == ihvp_from_factory.weights
    for ihvp_batch, factory_batch in zip(ihvp.train_set, ihvp_from_factory.train_set):
        assert almost_equal(ihvp_batch[0], factory_batch[0])
        assert almost_equal(ihvp_batch[1], factory_batch[1])

    # case 3: model feature extractor without layer position
    feature_extractor = nn.Sequential(model[0])
    with pytest.raises(AssertionError):
        cgd_factory = CGDIHVPFactory(feature_extractor, n_cgd_iters)


def test_lissa_factory():
    """Test that LissaIHVPFactory produces equivalent LissaIHVP instances."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.Linear(2, 1, bias=False)
    )

    loss_fn = nn.MSELoss(reduction="none")

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    inputs = torch.randn((25, 1, 3))
    targets = torch.randn((25, 1, 1))
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=False)

    n_lissa_iters = 100

    # case 1: model feature extractor
    feature_extractor = nn.Sequential(model[0])
    ihvp = LissaIHVP(influence_model, 1, train_loader, n_lissa_iters, feature_extractor)
    lissa_factory = LissaIHVPFactory(feature_extractor, n_lissa_iters, 1)
    assert isinstance(lissa_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = lissa_factory.build(influence_model, train_loader)
    assert ihvp.extractor_layer == ihvp_from_factory.extractor_layer
    assert len(list(ihvp.feature_extractor.children())) == len(list(ihvp_from_factory.feature_extractor.children()))
    assert ihvp.weights == ihvp_from_factory.weights
    for ihvp_batch, factory_batch in zip(ihvp.train_set, ihvp_from_factory.train_set):
        assert almost_equal(ihvp_batch[0], factory_batch[0])
        assert almost_equal(ihvp_batch[1], factory_batch[1])

    # case 2: layer position feature extractor
    feature_extractor = 1
    ihvp = LissaIHVP(influence_model, 1, train_loader, n_lissa_iters, None)
    lissa_factory = LissaIHVPFactory(feature_extractor, n_lissa_iters)
    assert isinstance(lissa_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = lissa_factory.build(influence_model, train_loader)
    assert ihvp.extractor_layer == ihvp_from_factory.extractor_layer
    assert len(list(ihvp.feature_extractor.children())) == len(list(ihvp_from_factory.feature_extractor.children()))
    assert ihvp.weights == ihvp_from_factory.weights
    for ihvp_batch, factory_batch in zip(ihvp.train_set, ihvp_from_factory.train_set):
        assert almost_equal(ihvp_batch[0], factory_batch[0])
        assert almost_equal(ihvp_batch[1], factory_batch[1])

    # case 3: model feature extractor without layer position
    feature_extractor = nn.Sequential(model[0])
    with pytest.raises(AssertionError):
        lissa_factory = LissaIHVPFactory(feature_extractor, n_lissa_iters)


def test_kfac_factory(tmp_path):
    """Test that KfacIHVPFactory produces configured KfacIHVP instances."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(4, 3, bias=False, dtype=torch.float64),
        nn.Linear(3, 2, bias=False, dtype=torch.float64),
    )
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)

    inputs = torch.randn((10, 4), dtype=torch.float64)
    targets = torch.randn((10, 2), dtype=torch.float64)
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=False)

    factors_path = str(tmp_path / "kfac_factors")
    kfac_factory = KfacIHVPFactory(
        damping=1e-3,
        layer_collection="recursive",
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="memory",
        factors_path=factors_path,
        overwrite_factors=True,
    )
    assert isinstance(kfac_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = kfac_factory.build(influence_model, train_loader)
    assert isinstance(ihvp_from_factory, KfacIHVP)
    assert ihvp_from_factory.layer_map.layer_collection == "recursive"
    assert ihvp_from_factory.factors.module_partition_size == 1
    assert ihvp_from_factory.factors.offload_activations_to_cpu
    assert ihvp_from_factory.factors.data_partition_size == 2
    assert ihvp_from_factory.factors.accumulator_offload_mode == "memory"
    assert ihvp_from_factory.factors_path == factors_path
    assert ihvp_from_factory.overwrite_factors


def test_ekfac_factory(tmp_path):
    """Test that EkfacIHVPFactory produces configured EkfacIHVP instances."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(4, 3, bias=False, dtype=torch.float64),
        nn.Linear(3, 2, bias=False, dtype=torch.float64),
    )
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)

    inputs = torch.randn((10, 4), dtype=torch.float64)
    targets = torch.randn((10, 2), dtype=torch.float64)
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=False)

    factors_path = str(tmp_path / "ekfac_factors")
    ekfac_factory = EkfacIHVPFactory(
        damping=1e-3,
        layer_collection="recursive",
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="memory",
        factors_path=factors_path,
        overwrite_factors=True,
    )
    assert isinstance(ekfac_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = ekfac_factory.build(influence_model, train_loader)
    assert isinstance(ihvp_from_factory, EkfacIHVP)
    assert ihvp_from_factory.layer_map.layer_collection == "recursive"
    assert ihvp_from_factory.factors.module_partition_size == 1
    assert ihvp_from_factory.factors.offload_activations_to_cpu
    assert ihvp_from_factory.factors.data_partition_size == 2
    assert ihvp_from_factory.factors.accumulator_offload_mode == "memory"
    assert ihvp_from_factory.factors_path == factors_path
    assert ihvp_from_factory.overwrite_factors
