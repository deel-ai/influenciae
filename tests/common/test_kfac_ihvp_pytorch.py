# tests/common/test_kfac_ihvp_pytorch.py
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for K-FAC and EK-FAC IHVP implementations (PyTorch backend).

Test strategy
-------------
1. **LayerParameterMap**: verify correct enumeration and flat-index mapping.
2. **K-FAC/EK-FAC implementation checks**: validate exact agreement with
   ExactIHVP on a linear setting where Hessian and empirical Fisher coincide.
3. **K-FAC/EK-FAC robustness checks**: smoke and determinism tests on a
   nonlinear model where exact Hessian and Fisher differ.
4. **Integration & factories**: verify enum/factory wiring and unsupported HVP behavior.
5. **Checkpointing & partitioning**: validate factor checkpoint and partition options.
"""
import os
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.common import (
    InfluenceModel,
    ExactIHVP,
    KfacIHVP,
    EkfacIHVP,
    IHVPCalculator,
    KfacIHVPFactory,
)
from deel.influenciae.common.kfac_factors import LayerParameterMap, KroneckerFactors, EKFACFactors


pytestmark = pytest.mark.pytorch


LINEAR_EXACT_TARGET = 1.0 / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seed():
    torch.manual_seed(42)
    return 42


def _make_simple_mlp(seed=42):
    """Two-layer linear model: Linear(4,3) -> Linear(3,2), no bias, float64."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(4, 3, bias=False, dtype=torch.float64),
        nn.Linear(3, 2, bias=False, dtype=torch.float64),
    )
    return model


def _make_simple_mlp_with_bias(seed=42):
    """Two-layer linear model with bias."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(4, 3, bias=True, dtype=torch.float64),
        nn.Linear(3, 2, bias=True, dtype=torch.float64),
    )
    return model


def _make_nested_mlp(seed=42):
    """Nested model where one supported linear layer is inside a submodule."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(4, 3, bias=False, dtype=torch.float64),
            nn.ReLU(),
        ),
        nn.Linear(3, 2, bias=False, dtype=torch.float64),
    )
    return model


def _make_nonlinear_mlp(seed=42):
    """Two-layer nonlinear model used for smoke/determinism tests."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(4, 8, bias=True, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(8, 2, bias=True, dtype=torch.float64),
    )
    return model


def _make_single_layer_linear_model(seed=42):
    """Single-layer linear model with zero weights for exact K-FAC checks."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(4, 1, bias=False, dtype=torch.float64),
    )
    with torch.no_grad():
        model[0].weight.zero_()
    return model


def _make_dataset(n_samples=20, n_features=4, n_outputs=2, seed=123, batch_size=5):
    """Create a simple regression dataset."""
    gen = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_samples, n_features, generator=gen, dtype=torch.float64)
    targets = torch.randn(n_samples, n_outputs, generator=gen, dtype=torch.float64)
    ds = TensorDataset(inputs, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_linear_exact_dataset(n_samples=64, n_features=4, seed=123, batch_size=16):
    """Dataset where single-layer linear MSE satisfies Hessian == empirical Fisher."""
    gen = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_samples, n_features, generator=gen, dtype=torch.float64)
    targets = torch.full((n_samples, 1), fill_value=LINEAR_EXACT_TARGET, dtype=torch.float64)
    ds = TensorDataset(inputs, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _cosine_similarity(a, b):
    """Compute cosine similarity between two flat vectors (numpy)."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _relative_error(reference, approx):
    """Relative L2 error between two vectors (numpy)."""
    reference = reference.flatten().astype(np.float64)
    approx = approx.flatten().astype(np.float64)
    denom = np.linalg.norm(reference)
    if denom < 1e-12:
        return np.linalg.norm(approx)
    return np.linalg.norm(approx - reference) / denom


def _compute_alignment_metrics(reference, approx):
    """Compute per-sample cosine similarity and relative error metrics."""
    reference_np = reference.detach().cpu().numpy().T
    approx_np = approx.detach().cpu().numpy().T

    cosine_scores = []
    relative_errors = []
    for ref_sample, approx_sample in zip(reference_np, approx_np):
        cosine_scores.append(_cosine_similarity(ref_sample, approx_sample))
        relative_errors.append(_relative_error(ref_sample, approx_sample))

    return {
        "median_cosine": float(np.median(cosine_scores)),
        "median_relative_error": float(np.median(relative_errors)),
    }


# ---------------------------------------------------------------------------
# 1. LayerParameterMap
# ---------------------------------------------------------------------------


def test_maps_all_linear_layers(seed):
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    layer_map = LayerParameterMap(influence_model, backend)

    # Two linear layers, both should be supported
    assert layer_map.n_supported_layers == 2

    # First layer: Linear(4,3) -> 12 params
    info0 = layer_map.layers_info[0]
    assert info0.flat_start == 0
    assert info0.flat_end == 12  # 4*3
    assert not info0.has_bias

    # Second layer: Linear(3,2) -> 6 params
    info1 = layer_map.layers_info[1]
    assert info1.flat_start == 12
    assert info1.flat_end == 18  # 3*2
    assert not info1.has_bias

def test_target_layers_filter(seed):
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    # Only include layer 0
    layer_map = LayerParameterMap(influence_model, backend, target_layers=[0])
    assert layer_map.n_supported_layers == 1
    assert layer_map.layers_info[0].layer_idx == 0

def test_warns_on_unsupported_target_layer(seed):
    """If target_layers includes a non-linear layer, warn."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(4, 3, bias=False, dtype=torch.float64),
        nn.BatchNorm1d(3, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(3, 2, bias=False, dtype=torch.float64),
    )
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    all_layers = backend.get_layers(model)
    all_indices = list(range(len(all_layers)))
    unsupported_indices = [
        i for i, layer in enumerate(all_layers)
        if not backend.is_linear_layer(layer) and not backend.is_conv2d_layer(layer)
    ]
    supported_indices = [
        i for i, layer in enumerate(all_layers)
        if backend.is_linear_layer(layer) or backend.is_conv2d_layer(layer)
    ]

    assert unsupported_indices

    with pytest.warns(UserWarning, match="not a Linear/Dense"):
        layer_map = LayerParameterMap(influence_model, backend, target_layers=all_indices)

    assert layer_map.n_supported_layers == len(supported_indices)
    assert {info.layer_idx for info in layer_map.layers_info} == set(supported_indices)

def test_recursive_collection_tracks_nested_supported_layers(seed):
    """Recursive collection should include supported nested layers."""
    model = _make_nested_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    top_level_map = LayerParameterMap(influence_model, backend, layer_collection="top_level")
    recursive_map = LayerParameterMap(influence_model, backend, layer_collection="recursive")

    assert top_level_map.n_supported_layers == 1
    assert recursive_map.n_supported_layers == 2

    recursive_names = {info.layer_name for info in recursive_map.layers_info}
    assert "0.0" in recursive_names
    assert "1" in recursive_names

def test_recursive_target_layers_filter(seed):
    """target_layers should apply to recursive traversal indices when recursive mode is used."""
    model = _make_nested_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    recursive_named_layers = backend.get_named_layers(model, recursive=True)
    nested_linear_idx = next(
        idx for idx, (name, layer) in enumerate(recursive_named_layers)
        if name == "0.0" and backend.is_linear_layer(layer)
    )

    recursive_map = LayerParameterMap(
        influence_model,
        backend,
        target_layers=[nested_linear_idx],
        layer_collection="recursive",
    )
    assert recursive_map.n_supported_layers == 1
    assert recursive_map.layers_info[0].layer_name == "0.0"

def test_invalid_layer_collection_raises(seed):
    """An invalid layer_collection value should raise ValueError."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    with pytest.raises(ValueError, match="layer_collection"):
        LayerParameterMap(influence_model, backend, layer_collection="invalid")


# ---------------------------------------------------------------------------
# 2. K-FAC IHVP validation
# ---------------------------------------------------------------------------


def test_kfac_matches_exact_on_linear_model(seed):
    """On a crafted linear setting, K-FAC should match ExactIHVP closely."""
    model = _make_single_layer_linear_model(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_linear_exact_dataset(n_samples=96, n_features=4, seed=123, batch_size=16)
    batch = next(iter(train_loader))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    exact_ihvp = ExactIHVP(influence_model, train_loader)
    kfac_ihvp = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-8,
        fisher_type="empirical",
    )

    exact_result = exact_ihvp._compute_ihvp_single_batch(batch)
    kfac_result = kfac_ihvp._compute_ihvp_single_batch(batch)
    metrics = _compute_alignment_metrics(exact_result, kfac_result)

    assert metrics["median_cosine"] > 0.995, metrics
    assert metrics["median_relative_error"] < 8e-2, metrics

@pytest.mark.slow
@pytest.mark.parametrize("fisher_type", ["empirical", "true"])
def test_kfac_ihvp_smoke_nonlinear(fisher_type, seed):
    """K-FAC should return finite, non-degenerate IHVPs on a nonlinear model."""
    model = _make_nonlinear_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=128, n_features=4, n_outputs=2, seed=321, batch_size=16)
    batch = next(iter(_make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=654, batch_size=8)))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    kfac_ihvp = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        fisher_type=fisher_type,
    )

    result = kfac_ihvp._compute_ihvp_single_batch(batch)
    result_np = result.detach().cpu().numpy()

    assert result.shape == (influence_model.nb_params, batch[0].shape[0])
    assert np.all(np.isfinite(result_np))
    assert np.median(np.linalg.norm(result_np, axis=0)) > 1e-10

def test_kfac_ihvp_deterministic(seed):
    """Empirical K-FAC should be deterministic for fixed data and seeds."""
    model = _make_nonlinear_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=96, n_features=4, n_outputs=2, seed=2024, batch_size=16)
    batch = next(iter(_make_dataset(n_samples=18, n_features=4, n_outputs=2, seed=2025, batch_size=6)))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    first = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        fisher_type="empirical",
    )
    second = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        fisher_type="empirical",
    )

    first_result = first._compute_ihvp_single_batch(batch).detach().cpu().numpy()
    second_result = second._compute_ihvp_single_batch(batch).detach().cpu().numpy()
    assert np.allclose(first_result, second_result, rtol=1e-10, atol=1e-12)

def test_kfac_enum_from_string():
    """IHVPCalculator.from_string('kfac') should return Kfac enum."""
    calc = IHVPCalculator.from_string('kfac')
    assert calc is IHVPCalculator.Kfac
    assert calc.value is KfacIHVP


# ---------------------------------------------------------------------------
# 3. EK-FAC IHVP validation
# ---------------------------------------------------------------------------


def test_ekfac_matches_exact_on_linear_model(seed):
    """On a crafted linear setting, EK-FAC should match ExactIHVP closely."""
    model = _make_single_layer_linear_model(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_linear_exact_dataset(n_samples=96, n_features=4, seed=123, batch_size=16)
    batch = next(iter(train_loader))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    exact_ihvp = ExactIHVP(influence_model, train_loader)
    ekfac_ihvp = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-8,
        fisher_type="empirical",
        n_ekfac_samples=96,
    )

    exact_result = exact_ihvp._compute_ihvp_single_batch(batch)
    ekfac_result = ekfac_ihvp._compute_ihvp_single_batch(batch)
    metrics = _compute_alignment_metrics(exact_result, ekfac_result)

    assert metrics["median_cosine"] > 0.995, metrics
    assert metrics["median_relative_error"] < 1.2e-1, metrics

@pytest.mark.slow
@pytest.mark.parametrize("fisher_type", ["empirical", "true"])
def test_ekfac_ihvp_smoke_nonlinear(fisher_type, seed):
    """EK-FAC should return finite, non-degenerate IHVPs on a nonlinear model."""
    model = _make_nonlinear_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=128, n_features=4, n_outputs=2, seed=321, batch_size=16)
    batch = next(iter(_make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=654, batch_size=8)))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    ekfac_ihvp = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        fisher_type=fisher_type,
        n_ekfac_samples=128,
    )

    result = ekfac_ihvp._compute_ihvp_single_batch(batch)
    result_np = result.detach().cpu().numpy()

    assert result.shape == (influence_model.nb_params, batch[0].shape[0])
    assert np.all(np.isfinite(result_np))
    assert np.median(np.linalg.norm(result_np, axis=0)) > 1e-10

def test_ekfac_ihvp_deterministic(seed):
    """Empirical EK-FAC should be deterministic for fixed data and seeds."""
    model = _make_nonlinear_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=96, n_features=4, n_outputs=2, seed=2024, batch_size=16)
    batch = next(iter(_make_dataset(n_samples=18, n_features=4, n_outputs=2, seed=2025, batch_size=6)))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    first = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        fisher_type="empirical",
        n_ekfac_samples=96,
    )
    second = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        fisher_type="empirical",
        n_ekfac_samples=96,
    )

    first_result = first._compute_ihvp_single_batch(batch).detach().cpu().numpy()
    second_result = second._compute_ihvp_single_batch(batch).detach().cpu().numpy()
    assert np.allclose(first_result, second_result, rtol=1e-10, atol=1e-12)

def test_ekfac_enum_from_string():
    """IHVPCalculator.from_string('ekfac') should return Ekfac enum."""
    calc = IHVPCalculator.from_string('ekfac')
    assert calc is IHVPCalculator.Ekfac
    assert calc.value is EkfacIHVP


# ---------------------------------------------------------------------------
# 4. Factor checkpointing
# ---------------------------------------------------------------------------


def test_kfac_factor_checkpoint_roundtrip(seed, tmp_path):
    """K-FAC checkpoint load should reproduce the same IHVP output."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=16, n_features=4, n_outputs=2, seed=123, batch_size=4)
    batch = next(iter(train_loader))
    checkpoint_dir = str(tmp_path / "kfac_factors")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    computed = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        factors_path=checkpoint_dir,
    )
    computed_result = computed._compute_ihvp_single_batch(batch)

    loaded = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        factors_path=checkpoint_dir,
    )
    loaded_result = loaded._compute_ihvp_single_batch(batch)

    assert np.allclose(
        computed_result.detach().cpu().numpy(),
        loaded_result.detach().cpu().numpy(),
        atol=1e-10,
        rtol=1e-8,
    )

def test_kfac_checkpoint_load_skips_recompute(seed, tmp_path, monkeypatch):
    """Existing K-FAC checkpoint should avoid factor recomputation."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=123, batch_size=4)
    checkpoint_dir = str(tmp_path / "kfac_skip_recompute")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    _ = KfacIHVP(influence_model, train_loader, damping=1e-3, factors_path=checkpoint_dir)

    def _raise_if_recomputed(*_args, **_kwargs):
        raise AssertionError("K-FAC factors were recomputed instead of loaded from checkpoint")

    monkeypatch.setattr(KroneckerFactors, "_compute_factors", _raise_if_recomputed)

    loaded = KfacIHVP(influence_model, train_loader, damping=1e-3, factors_path=checkpoint_dir)
    assert isinstance(loaded, KfacIHVP)

def test_kfac_checkpoint_layer_mismatch_raises(seed, tmp_path):
    """Layer signature mismatch should raise a clear checkpoint error."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=321, batch_size=4)
    checkpoint_dir = str(tmp_path / "kfac_mismatch")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    _ = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        target_layers=[0],
        factors_path=checkpoint_dir,
    )

    with pytest.raises(ValueError, match="layer signature mismatch"):
        KfacIHVP(
            influence_model,
            train_loader,
            damping=1e-3,
            target_layers=[1],
            factors_path=checkpoint_dir,
        )

def test_ekfac_factor_checkpoint_roundtrip(seed, tmp_path):
    """EK-FAC checkpoint load should reproduce the same IHVP output."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=16, n_features=4, n_outputs=2, seed=456, batch_size=4)
    batch = next(iter(train_loader))
    checkpoint_dir = str(tmp_path / "ekfac_factors")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    computed = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        factors_path=checkpoint_dir,
        n_ekfac_samples=None,
    )
    computed_result = computed._compute_ihvp_single_batch(batch)

    loaded = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        factors_path=checkpoint_dir,
        n_ekfac_samples=None,
    )
    loaded_result = loaded._compute_ihvp_single_batch(batch)

    assert np.allclose(
        computed_result.detach().cpu().numpy(),
        loaded_result.detach().cpu().numpy(),
        atol=1e-10,
        rtol=1e-8,
    )

def test_ekfac_checkpoint_load_skips_recompute(seed, tmp_path, monkeypatch):
    """Existing EK-FAC checkpoint should avoid corrected-eigenvalue recomputation."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=654, batch_size=4)
    checkpoint_dir = str(tmp_path / "ekfac_skip_recompute")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    _ = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        factors_path=checkpoint_dir,
        n_ekfac_samples=None,
    )

    def _raise_if_recomputed(*_args, **_kwargs):
        raise AssertionError("EK-FAC factors were recomputed instead of loaded from checkpoint")

    monkeypatch.setattr(EKFACFactors, "_estimate_corrected_eigenvalues", _raise_if_recomputed)

    loaded = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        factors_path=checkpoint_dir,
        n_ekfac_samples=None,
    )
    assert isinstance(loaded, EkfacIHVP)

def test_kfac_factory_uses_checkpoint(seed, tmp_path, monkeypatch):
    """Factory should pass checkpoint options through to K-FAC IHVP."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=777, batch_size=4)
    checkpoint_dir = str(tmp_path / "kfac_factory_checkpoint")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    factory = KfacIHVPFactory(damping=1e-3, factors_path=checkpoint_dir)
    _ = factory.build(influence_model, train_loader)
    assert (tmp_path / "kfac_factory_checkpoint" / "metadata.json").exists()

    def _raise_if_recomputed(*_args, **_kwargs):
        raise AssertionError("Factory build recomputed factors instead of loading checkpoint")

    monkeypatch.setattr(KroneckerFactors, "_compute_factors", _raise_if_recomputed)
    loaded = factory.build(influence_model, train_loader)
    assert isinstance(loaded, KfacIHVP)


# ---------------------------------------------------------------------------
# 5. Integration: HVP not supported
# ---------------------------------------------------------------------------


def test_kfac_hvp_raises(seed):
    """K-FAC compute_hvp should raise NotImplementedError."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=10, n_features=4, n_outputs=2, seed=123, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    kfac = KfacIHVP(influence_model, train_loader, damping=1e-3)

    batch = next(iter(train_loader))
    with pytest.raises(NotImplementedError):
        kfac._compute_hvp_single_batch(batch)

def test_ekfac_hvp_raises(seed):
    """EK-FAC compute_hvp should raise NotImplementedError."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=10, n_features=4, n_outputs=2, seed=123, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    ekfac = EkfacIHVP(influence_model, train_loader, damping=1e-3)

    batch = next(iter(train_loader))
    with pytest.raises(NotImplementedError):
        ekfac._compute_hvp_single_batch(batch)


# ---------------------------------------------------------------------------
# 6. Model with bias
# ---------------------------------------------------------------------------


def test_kfac_with_bias_runs(seed):
    """K-FAC should work with biased layers without errors."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=10, n_features=4, n_outputs=2, seed=123, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    kfac = KfacIHVP(influence_model, train_loader, damping=1e-3)

    batch = next(iter(train_loader))
    result = kfac._compute_ihvp_single_batch(batch)
    # nb_params = 4*3 + 3 + 3*2 + 2 = 23
    assert result.shape[0] == influence_model.nb_params


# ---------------------------------------------------------------------------
# 7. Memory partitioning options
# ---------------------------------------------------------------------------


def test_invalid_partition_sizes_raise(seed):
    """Invalid partition sizes should raise explicit errors."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=123, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    with pytest.raises(ValueError, match="module_partition_size"):
        KfacIHVP(
            influence_model,
            train_loader,
            module_partition_size=0,
        )

    with pytest.raises(ValueError, match="data_partition_size"):
        KfacIHVP(
            influence_model,
            train_loader,
            data_partition_size=0,
        )

    with pytest.raises(ValueError, match="accumulator_offload_mode"):
        KfacIHVP(
            influence_model,
            train_loader,
            accumulator_offload_mode="invalid",
        )

def test_kfac_memory_options_match_baseline(seed):
    """K-FAC memory partitioning options should preserve results."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=777, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    baseline = KfacIHVP(influence_model, train_loader, damping=1e-3)
    partitioned = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="memory",
    )
    assert partitioned.factors._factor_checkpoint is not None

    for info in baseline.layer_map.layers_info:
        idx = info.layer_idx
        assert torch.allclose(
            baseline.factors.A[idx],
            partitioned.factors.A[idx],
            rtol=1e-6,
            atol=1e-8,
        )
        assert torch.allclose(
            baseline.factors.G[idx],
            partitioned.factors.G[idx],
            rtol=1e-6,
            atol=1e-8,
        )

    batch = next(iter(train_loader))
    baseline_ihvp = baseline._compute_ihvp_single_batch(batch)
    partitioned_ihvp = partitioned._compute_ihvp_single_batch(batch)
    assert torch.allclose(
        baseline_ihvp,
        partitioned_ihvp,
        rtol=1e-5,
        atol=1e-7,
    )

def test_ekfac_memory_options_match_baseline(seed):
    """EK-FAC memory partitioning options should preserve results."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=999, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    baseline = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        n_ekfac_samples=24,
    )
    partitioned = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        n_ekfac_samples=24,
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="memory",
    )
    assert partitioned.factors._factor_checkpoint is not None
    assert partitioned.factors._corrected_checkpoint is not None

    for info in baseline.layer_map.layers_info:
        idx = info.layer_idx
        assert torch.allclose(
            baseline.factors.Lambda_corrected[idx],
            partitioned.factors.Lambda_corrected[idx],
            rtol=1e-6,
            atol=1e-8,
        )

    batch = next(iter(train_loader))
    baseline_ihvp = baseline._compute_ihvp_single_batch(batch)
    partitioned_ihvp = partitioned._compute_ihvp_single_batch(batch)
    assert torch.allclose(
        baseline_ihvp,
        partitioned_ihvp,
        rtol=1e-5,
        atol=1e-7,
    )

def test_kfac_disk_offload_matches_baseline(seed, tmp_path):
    """K-FAC disk accumulator offload should preserve results."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=432, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    baseline = KfacIHVP(influence_model, train_loader, damping=1e-3)
    partitioned = KfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="disk",
        accumulator_offload_dir=str(tmp_path / "kfac_disk_offload"),
        keep_accumulator_offload_artifacts=True,
    )

    assert partitioned.factors._factor_checkpoint is not None
    assert partitioned.factors._factor_checkpoint["mode"] == "disk"
    partition_files = partitioned.factors._factor_checkpoint["partition_files"]
    assert partition_files
    for partition_file in partition_files:
        assert os.path.isfile(partition_file)

    for info in baseline.layer_map.layers_info:
        idx = info.layer_idx
        assert torch.allclose(
            baseline.factors.A[idx],
            partitioned.factors.A[idx],
            rtol=1e-6,
            atol=1e-8,
        )
        assert torch.allclose(
            baseline.factors.G[idx],
            partitioned.factors.G[idx],
            rtol=1e-6,
            atol=1e-8,
        )

def test_ekfac_disk_offload_matches_baseline(seed, tmp_path):
    """EK-FAC disk accumulator offload should preserve results."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = nn.MSELoss(reduction="none")
    train_loader = _make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=876, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    baseline = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        n_ekfac_samples=24,
    )
    partitioned = EkfacIHVP(
        influence_model,
        train_loader,
        damping=1e-3,
        n_ekfac_samples=24,
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="disk",
        accumulator_offload_dir=str(tmp_path / "ekfac_disk_offload"),
        keep_accumulator_offload_artifacts=True,
    )

    assert partitioned.factors._factor_checkpoint is not None
    assert partitioned.factors._factor_checkpoint["mode"] == "disk"
    assert partitioned.factors._corrected_checkpoint is not None
    assert partitioned.factors._corrected_checkpoint["mode"] == "disk"
    for partition_file in partitioned.factors._factor_checkpoint["partition_files"]:
        assert os.path.isfile(partition_file)
    for partition_file in partitioned.factors._corrected_checkpoint["partition_files"]:
        assert os.path.isfile(partition_file)

    for info in baseline.layer_map.layers_info:
        idx = info.layer_idx
        assert torch.allclose(
            baseline.factors.Lambda_corrected[idx],
            partitioned.factors.Lambda_corrected[idx],
            rtol=1e-6,
            atol=1e-8,
        )
