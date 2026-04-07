# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Integration tests for query-batched influence computation (PyTorch backend).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.common import InfluenceModel, ExactIHVP, KfacIHVP, EkfacIHVP, CACHE
from deel.influenciae.common.query_batching import QueryBatchingConfig, PreconditioningMode
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils.sorted_dict import ORDER
from ..utils_test import set_seed_torch


pytestmark = pytest.mark.pytorch

set_seed = set_seed_torch
_ATOL = 1e-4


def _make_calc(normalize=False):
    """Build a tiny linear regression setup and return the calculator + data loaders."""
    set_seed(0)
    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    torch.manual_seed(1)
    x_train = torch.randn(20, 4)
    y_train = torch.randn(20, 2)
    x_test = torch.randn(6, 4)
    y_test = torch.randn(6, 2)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=5)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=3)

    ihvp = ExactIHVP(influence_model, train_loader)
    calc = FirstOrderInfluenceCalculator(
        influence_model, train_loader, ihvp, normalize=normalize
    )
    return calc, train_loader, test_loader


def _make_factorized_calc(ihvp_cls, normalize=False):
    """Build a small two-layer setup for K-FAC/EK-FAC query-side tests."""
    set_seed(0)
    model = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Linear(3, 2, bias=False))
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=0, loss_function=loss_fn)

    torch.manual_seed(4)
    x_train = torch.randn(24, 4)
    y_train = torch.randn(24, 2)
    x_test = torch.randn(6, 4)
    y_test = torch.randn(6, 2)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=6)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=3)

    ihvp = ihvp_cls(influence_model, train_loader, damping=1e-3)
    calc = FirstOrderInfluenceCalculator(influence_model, train_loader, ihvp, normalize=normalize)
    return calc, train_loader, test_loader


# ---------------------------------------------------------------------------
# supports_query_preconditioning
# ---------------------------------------------------------------------------

def test_exact_ihvp_supports_query_preconditioning():
    calc, _, _ = _make_calc()
    assert calc.ihvp_calculator.supports_query_preconditioning is True


def test_cgd_does_not_support_query_preconditioning():
    from deel.influenciae.common import ConjugateGradientDescentIHVP
    set_seed(0)
    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    torch.manual_seed(2)
    x_train = torch.randn(10, 4)
    y_train = torch.randn(10, 2)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=5)
    cgd = ConjugateGradientDescentIHVP(influence_model, -1, train_loader)
    assert cgd.supports_query_preconditioning is False


# ---------------------------------------------------------------------------
# Helper: collect scores from the two paths
# ---------------------------------------------------------------------------

def _collect_standard_scores(calc, test_loader, train_loader):
    """Collect full score matrix using the standard (train-side) path."""
    inf_vect_ds = calc.compute_influence_vector(train_loader)
    all_scores = []
    for test_batch in test_loader:
        test_tuple = tuple(test_batch)
        preproc = calc._preprocess_samples(test_tuple)
        row_scores = []
        for item in inf_vect_ds:
            inf_vect = item[-1]
            scores = calc._estimate_influence_value_from_influence_vector(preproc, inf_vect)
            row_scores.append(scores.detach().numpy())
        all_scores.append(np.concatenate(row_scores, axis=1))
    return np.concatenate(all_scores, axis=0)


def _collect_query_batched_scores(calc, test_loader, train_loader, config=None):
    """Collect full score matrix using the query-batched path."""
    all_scores = []
    for _query_batch, scores_ds in calc.estimate_influence_values_query_batched(
        test_loader, train_loader, config
    ):
        batch_scores = []
        for _train_batch, scores in scores_ds:
            batch_scores.append(scores.detach().numpy())
        all_scores.append(np.concatenate(batch_scores, axis=1))
    return np.concatenate(all_scores, axis=0)


def _collect_public_query_mode_scores(calc, test_loader, train_loader, config=None):
    """Collect full scores through the public QUERY-mode API."""
    all_scores = []
    eval_inf_ds = calc.estimate_influence_values_in_batches(
        test_loader,
        train_loader,
        preconditioning_mode=PreconditioningMode.QUERY,
        query_batching_config=config,
    )
    for _query_batch, scores_ds in eval_inf_ds:
        batch_scores = []
        for _train_batch, scores in scores_ds:
            batch_scores.append(scores.detach().numpy())
        all_scores.append(np.concatenate(batch_scores, axis=1))
    return np.concatenate(all_scores, axis=0)


def _collect_train_inputs(train_loader):
    """Collect the training inputs tensor from a data loader."""
    return torch.cat([batch_x for batch_x, _ in train_loader], dim=0).numpy()


def _collect_top_k_query_mode(calc, test_loader, train_loader, k, config=None, order=ORDER.DESCENDING):
    """Collect top-k values and samples through the public QUERY-mode API."""
    values = []
    samples = []
    top_k_ds = calc.top_k(
        test_loader,
        train_loader,
        k=k,
        order=order,
        preconditioning_mode=PreconditioningMode.QUERY,
        query_batching_config=config,
    )
    for _query_batch, influence_values, training_samples in top_k_ds:
        values.append(influence_values.detach().numpy())
        samples.append(training_samples.detach().numpy())
    return np.concatenate(values, axis=0), np.concatenate(samples, axis=0)


# ---------------------------------------------------------------------------
# Query-batched scores match standard scores
# ---------------------------------------------------------------------------

def test_query_batched_matches_standard_full_rank():
    calc, train_loader, test_loader = _make_calc(normalize=False)
    standard = _collect_standard_scores(calc, test_loader, train_loader)
    query_batched = _collect_query_batched_scores(calc, test_loader, train_loader)
    np.testing.assert_allclose(query_batched, standard, atol=_ATOL)


def test_query_batched_matches_standard_normalized():
    calc, train_loader, test_loader = _make_calc(normalize=True)
    standard = _collect_standard_scores(calc, test_loader, train_loader)
    query_batched = _collect_query_batched_scores(calc, test_loader, train_loader)
    np.testing.assert_allclose(query_batched, standard, atol=_ATOL)


def test_accumulation_steps_match_no_accumulation():
    calc, train_loader, test_loader = _make_calc(normalize=False)
    config_1 = QueryBatchingConfig(query_gradient_accumulation_steps=1)
    config_3 = QueryBatchingConfig(query_gradient_accumulation_steps=3)
    scores_1 = _collect_query_batched_scores(calc, test_loader, train_loader, config_1)
    scores_3 = _collect_query_batched_scores(calc, test_loader, train_loader, config_3)
    np.testing.assert_allclose(scores_1, scores_3, atol=_ATOL)


def test_data_partitions_match_unpartitioned_scores():
    """Partitioning train-batch scoring should preserve ExactIHVP scores."""
    calc, train_loader, test_loader = _make_calc(normalize=False)
    baseline = _collect_query_batched_scores(calc, test_loader, train_loader)
    partitioned = _collect_query_batched_scores(
        calc,
        test_loader,
        train_loader,
        QueryBatchingConfig(score_data_partitions=2),
    )
    np.testing.assert_allclose(partitioned, baseline, atol=_ATOL)


def test_public_query_mode_matches_standard_full_rank():
    calc, train_loader, test_loader = _make_calc(normalize=False)
    standard = _collect_standard_scores(calc, test_loader, train_loader)
    query_mode = _collect_public_query_mode_scores(calc, test_loader, train_loader)
    np.testing.assert_allclose(query_mode, standard, atol=_ATOL)


def test_public_query_mode_matches_standard_normalized():
    calc, train_loader, test_loader = _make_calc(normalize=True)
    standard = _collect_standard_scores(calc, test_loader, train_loader)
    query_mode = _collect_public_query_mode_scores(calc, test_loader, train_loader)
    np.testing.assert_allclose(query_mode, standard, atol=_ATOL)


def test_public_top_k_query_mode_matches_full_matrix():
    calc, train_loader, test_loader = _make_calc(normalize=False)
    full_scores = _collect_standard_scores(calc, test_loader, train_loader)
    train_inputs = _collect_train_inputs(train_loader)

    indices = np.argsort(-full_scores, axis=1)[:, :3]
    expected_values = np.take_along_axis(full_scores, indices, axis=1)
    expected_samples = train_inputs[indices]

    values, samples = _collect_top_k_query_mode(calc, test_loader, train_loader, k=3)
    np.testing.assert_allclose(values, expected_values, atol=_ATOL)
    np.testing.assert_allclose(samples, expected_samples, atol=_ATOL)


def test_public_query_mode_rejects_train_side_vector_args():
    calc, train_loader, test_loader = _make_calc(normalize=False)

    with pytest.raises(ValueError, match="load_influence_vector_path"):
        calc.estimate_influence_values_in_batches(
            test_loader,
            train_loader,
            preconditioning_mode=PreconditioningMode.QUERY,
            load_influence_vector_path="ignored",
        )

    with pytest.raises(ValueError, match="save_influence_vector_path"):
        calc.estimate_influence_values_in_batches(
            test_loader,
            train_loader,
            preconditioning_mode=PreconditioningMode.QUERY,
            save_influence_vector_path="ignored",
        )

    with pytest.raises(ValueError, match="influence_vector_in_cache"):
        calc.estimate_influence_values_in_batches(
            test_loader,
            train_loader,
            preconditioning_mode=PreconditioningMode.QUERY,
            influence_vector_in_cache=CACHE.DISK,
        )

    with pytest.raises(ValueError, match="load_influence_vector_ds_path"):
        calc.top_k(
            test_loader,
            train_loader,
            preconditioning_mode=PreconditioningMode.QUERY,
            load_influence_vector_ds_path="ignored",
        )

    with pytest.raises(ValueError, match="save_influence_vector_ds_path"):
        calc.top_k(
            test_loader,
            train_loader,
            preconditioning_mode=PreconditioningMode.QUERY,
            save_influence_vector_ds_path="ignored",
        )

    with pytest.raises(ValueError, match="influence_vector_in_cache"):
        calc.top_k(
            test_loader,
            train_loader,
            preconditioning_mode=PreconditioningMode.QUERY,
            influence_vector_in_cache=CACHE.NO_CACHE,
        )


# ---------------------------------------------------------------------------
# Low-rank compression
# ---------------------------------------------------------------------------

def test_low_rank_compression_runs():
    calc, train_loader, test_loader = _make_calc(normalize=False)
    config = QueryBatchingConfig(query_gradient_low_rank=2)
    scores = _collect_query_batched_scores(calc, test_loader, train_loader, config)
    assert np.all(np.isfinite(scores)), "Scores contain inf/nan"


def test_low_rank_compression_shape():
    calc, train_loader, test_loader = _make_calc(normalize=False)
    config = QueryBatchingConfig(query_gradient_low_rank=1)
    scores = _collect_query_batched_scores(calc, test_loader, train_loader, config)
    assert scores.shape == (6, 20), f"Unexpected shape {scores.shape}"


@pytest.mark.parametrize("ihvp_cls", [KfacIHVP, EkfacIHVP])
def test_factorized_query_batching_matches_standard_full_rank(ihvp_cls):
    calc, train_loader, test_loader = _make_factorized_calc(ihvp_cls, normalize=False)
    standard = _collect_standard_scores(calc, test_loader, train_loader)
    query_batched = _collect_query_batched_scores(calc, test_loader, train_loader)
    np.testing.assert_allclose(query_batched, standard, atol=_ATOL)


@pytest.mark.parametrize("ihvp_cls", [KfacIHVP, EkfacIHVP])
def test_factorized_low_rank_and_partitioning_match_unpartitioned(ihvp_cls):
    calc, train_loader, test_loader = _make_factorized_calc(ihvp_cls, normalize=False)
    baseline = _collect_query_batched_scores(calc, test_loader, train_loader)
    config = QueryBatchingConfig(
        query_gradient_low_rank=3,
        score_data_partitions=2,
        score_module_partitions=2,
    )
    partitioned = _collect_query_batched_scores(calc, test_loader, train_loader, config)
    np.testing.assert_allclose(partitioned, baseline, atol=_ATOL)


# ---------------------------------------------------------------------------
# Error when IHVP does not support preconditioning
# ---------------------------------------------------------------------------

def test_raises_when_ihvp_not_supported():
    from deel.influenciae.common import ConjugateGradientDescentIHVP
    set_seed(0)
    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    loss_fn = nn.MSELoss(reduction="none")
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    torch.manual_seed(3)
    x_train = torch.randn(10, 4)
    y_train = torch.randn(10, 2)
    x_test = torch.randn(4, 4)
    y_test = torch.randn(4, 2)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=5)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=4)
    cgd = ConjugateGradientDescentIHVP(influence_model, -1, train_loader)
    calc = FirstOrderInfluenceCalculator(influence_model, train_loader, cgd)
    with pytest.raises(ValueError, match="does not support query-side preconditioning"):
        list(calc.estimate_influence_values_query_batched(test_loader, train_loader))
