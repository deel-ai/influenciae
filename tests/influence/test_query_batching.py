# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Integration tests for query-batched influence computation (TensorFlow backend).

The key invariant under test:
  For ExactIHVP, the scores produced by ``estimate_influence_values_query_batched``
  must match those produced by the standard ``estimate_influence_values_in_batches``
  path up to numerical precision.
"""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError, Reduction

from deel.influenciae.common import InfluenceModel, ExactIHVP, CACHE
from deel.influenciae.common import ConjugateGradientDescentIHVP
from deel.influenciae.common.query_batching import QueryBatchingConfig, PreconditioningMode
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils.sorted_dict import ORDER
from ..utils_test import set_seed_tf


pytestmark = pytest.mark.tensorflow

set_seed = set_seed_tf
_ATOL = 1e-4


def _make_calc(normalize=False):
    """Build a tiny linear regression setup and return the calculator + datasets."""
    set_seed()
    model = Sequential([Input(shape=(3,)), Dense(2, use_bias=False)])
    loss = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss)

    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((20, 3)).astype(np.float32)
    y_train = rng.standard_normal((20, 2)).astype(np.float32)
    x_test = rng.standard_normal((6, 3)).astype(np.float32)
    y_test = rng.standard_normal((6, 2)).astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(5)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(3)

    ihvp = ExactIHVP(influence_model, train_ds)
    calc = FirstOrderInfluenceCalculator(
        influence_model, train_ds, ihvp, normalize=normalize
    )
    return calc, train_ds, test_ds


# ---------------------------------------------------------------------------
# supports_query_preconditioning
# ---------------------------------------------------------------------------

def test_exact_ihvp_supports_query_preconditioning():
    calc, _, _ = _make_calc()
    assert calc.ihvp_calculator.supports_query_preconditioning is True


def test_iterative_ihvp_does_not_support_query_preconditioning():
    set_seed()
    model = Sequential([Input(shape=(3,)), Dense(2, use_bias=False)])
    loss = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss)
    rng = np.random.default_rng(1)
    x_train = rng.standard_normal((10, 3)).astype(np.float32)
    y_train = rng.standard_normal((10, 2)).astype(np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(5)
    cgd = ConjugateGradientDescentIHVP(influence_model, -1, train_ds)
    assert cgd.supports_query_preconditioning is False


# ---------------------------------------------------------------------------
# query-batched scores match standard scores (full-rank, no compression)
# ---------------------------------------------------------------------------

def _collect_standard_scores(calc, test_ds, train_ds):
    """Collect all influence scores using the standard train-side path."""
    all_scores = []
    train_batches = [tuple(batch) if isinstance(batch, (list, tuple)) else (batch,) for batch in train_ds]
    for test_batch in test_ds:
        test_tuple = tuple(test_batch) if isinstance(test_batch, (list, tuple)) else (test_batch,)
        preproc = calc._preprocess_samples(test_tuple)
        row_scores = []
        for train_batch in train_batches:
            inf_vect = calc._compute_influence_vector(train_batch)
            scores = calc._estimate_influence_value_from_influence_vector(preproc, inf_vect)
            row_scores.append(scores.numpy())
        all_scores.append(np.concatenate(row_scores, axis=1))
    return np.concatenate(all_scores, axis=0)  # (n_test, n_train)


def _collect_query_batched_scores(calc, test_ds, train_ds, config=None):
    """Collect all influence scores using the query-batched path."""
    all_scores = []
    for query_batch, scores_ds in calc.estimate_influence_values_query_batched(
        test_ds, train_ds, config
    ):
        batch_scores = []
        for _train_batch, scores in scores_ds:
            batch_scores.append(scores.numpy())
        all_scores.append(np.concatenate(batch_scores, axis=1))
    return np.concatenate(all_scores, axis=0)  # (n_test, n_train)


def _collect_public_query_mode_scores(calc, test_ds, train_ds, config=None):
    """Collect full scores through the public QUERY-mode API."""
    all_scores = []
    eval_inf_ds = calc.estimate_influence_values_in_batches(
        test_ds,
        train_ds,
        preconditioning_mode=PreconditioningMode.QUERY,
        query_batching_config=config,
    )
    for _query_batch, scores_ds in eval_inf_ds:
        batch_scores = []
        for _train_batch, scores in scores_ds:
            batch_scores.append(scores.numpy())
        all_scores.append(np.concatenate(batch_scores, axis=1))
    return np.concatenate(all_scores, axis=0)


def _collect_train_inputs(train_ds):
    """Collect the training inputs tensor from a batched dataset."""
    return np.concatenate([batch_x.numpy() for batch_x, _ in train_ds], axis=0)


def _collect_top_k_query_mode(calc, test_ds, train_ds, k, config=None, order=ORDER.DESCENDING):
    """Collect top-k values and samples through the public QUERY-mode API."""
    values = []
    samples = []
    top_k_ds = calc.top_k(
        test_ds,
        train_ds,
        k=k,
        order=order,
        preconditioning_mode=PreconditioningMode.QUERY,
        query_batching_config=config,
    )
    for _query_batch, influence_values, training_samples in top_k_ds:
        values.append(influence_values.numpy())
        samples.append(training_samples.numpy())
    return np.concatenate(values, axis=0), np.concatenate(samples, axis=0)


def test_query_batched_matches_standard_full_rank():
    calc, train_ds, test_ds = _make_calc(normalize=False)
    standard = _collect_standard_scores(calc, test_ds, train_ds)
    query_batched = _collect_query_batched_scores(calc, test_ds, train_ds)
    np.testing.assert_allclose(query_batched, standard, atol=_ATOL)


def test_query_batched_matches_standard_normalized():
    calc, train_ds, test_ds = _make_calc(normalize=True)
    standard = _collect_standard_scores(calc, test_ds, train_ds)
    query_batched = _collect_query_batched_scores(calc, test_ds, train_ds)
    np.testing.assert_allclose(query_batched, standard, atol=_ATOL)


# ---------------------------------------------------------------------------
# Query gradient accumulation steps
# ---------------------------------------------------------------------------

def test_accumulation_steps_match_no_accumulation():
    """Scores should be identical regardless of query_gradient_accumulation_steps."""
    calc, train_ds, test_ds = _make_calc(normalize=False)
    config_1 = QueryBatchingConfig(query_gradient_accumulation_steps=1)
    config_3 = QueryBatchingConfig(query_gradient_accumulation_steps=3)
    scores_1 = _collect_query_batched_scores(calc, test_ds, train_ds, config_1)
    scores_3 = _collect_query_batched_scores(calc, test_ds, train_ds, config_3)
    np.testing.assert_allclose(scores_1, scores_3, atol=_ATOL)


def test_data_partitions_match_unpartitioned_scores():
    """TensorFlow partitioned scoring should match the unpartitioned path."""
    calc, train_ds, test_ds = _make_calc(normalize=False)
    baseline = _collect_query_batched_scores(calc, test_ds, train_ds)
    partitioned = _collect_query_batched_scores(
        calc,
        test_ds,
        train_ds,
        QueryBatchingConfig(score_data_partitions=2),
    )
    np.testing.assert_allclose(partitioned, baseline, atol=_ATOL)


def test_public_query_mode_matches_standard_full_rank():
    calc, train_ds, test_ds = _make_calc(normalize=False)
    standard = _collect_standard_scores(calc, test_ds, train_ds)
    query_mode = _collect_public_query_mode_scores(calc, test_ds, train_ds)
    np.testing.assert_allclose(query_mode, standard, atol=_ATOL)


def test_public_query_mode_matches_standard_normalized():
    calc, train_ds, test_ds = _make_calc(normalize=True)
    standard = _collect_standard_scores(calc, test_ds, train_ds)
    query_mode = _collect_public_query_mode_scores(calc, test_ds, train_ds)
    np.testing.assert_allclose(query_mode, standard, atol=_ATOL)


def test_public_top_k_query_mode_matches_full_matrix():
    calc, train_ds, test_ds = _make_calc(normalize=False)
    full_scores = _collect_standard_scores(calc, test_ds, train_ds)
    train_inputs = _collect_train_inputs(train_ds)

    indices = np.argsort(-full_scores, axis=1)[:, :3]
    expected_values = np.take_along_axis(full_scores, indices, axis=1)
    expected_samples = train_inputs[indices]

    values, samples = _collect_top_k_query_mode(calc, test_ds, train_ds, k=3)
    np.testing.assert_allclose(values, expected_values, atol=_ATOL)
    np.testing.assert_allclose(samples, expected_samples, atol=_ATOL)


def test_public_query_mode_rejects_train_side_vector_args():
    calc, train_ds, test_ds = _make_calc(normalize=False)

    with pytest.raises(ValueError, match="load_influence_vector_path"):
        calc.estimate_influence_values_in_batches(
            test_ds,
            train_ds,
            preconditioning_mode=PreconditioningMode.QUERY,
            load_influence_vector_path="ignored",
        )

    with pytest.raises(ValueError, match="save_influence_vector_path"):
        calc.estimate_influence_values_in_batches(
            test_ds,
            train_ds,
            preconditioning_mode=PreconditioningMode.QUERY,
            save_influence_vector_path="ignored",
        )

    with pytest.raises(ValueError, match="influence_vector_in_cache"):
        calc.estimate_influence_values_in_batches(
            test_ds,
            train_ds,
            preconditioning_mode=PreconditioningMode.QUERY,
            influence_vector_in_cache=CACHE.DISK,
        )

    with pytest.raises(ValueError, match="load_influence_vector_ds_path"):
        calc.top_k(
            test_ds,
            train_ds,
            preconditioning_mode=PreconditioningMode.QUERY,
            load_influence_vector_ds_path="ignored",
        )

    with pytest.raises(ValueError, match="save_influence_vector_ds_path"):
        calc.top_k(
            test_ds,
            train_ds,
            preconditioning_mode=PreconditioningMode.QUERY,
            save_influence_vector_ds_path="ignored",
        )

    with pytest.raises(ValueError, match="influence_vector_in_cache"):
        calc.top_k(
            test_ds,
            train_ds,
            preconditioning_mode=PreconditioningMode.QUERY,
            influence_vector_in_cache=CACHE.NO_CACHE,
        )


# ---------------------------------------------------------------------------
# Low-rank compression does not crash and produces reasonable approximation
# ---------------------------------------------------------------------------

def test_low_rank_compression_runs():
    """Low-rank compression must run without error and return finite scores."""
    calc, train_ds, test_ds = _make_calc(normalize=False)
    config = QueryBatchingConfig(query_gradient_low_rank=2)
    scores = _collect_query_batched_scores(calc, test_ds, train_ds, config)
    assert np.all(np.isfinite(scores)), "Scores contain inf/nan"


def test_low_rank_compression_shape():
    """Output shape must be (n_test, n_train) regardless of rank."""
    calc, train_ds, test_ds = _make_calc(normalize=False)
    config = QueryBatchingConfig(query_gradient_low_rank=1)
    scores = _collect_query_batched_scores(calc, test_ds, train_ds, config)
    assert scores.shape == (6, 20), f"Unexpected shape {scores.shape}"


# ---------------------------------------------------------------------------
# Error raised when IHVP does not support preconditioning
# ---------------------------------------------------------------------------

def test_raises_when_ihvp_not_supported():
    set_seed()
    model = Sequential([Input(shape=(3,)), Dense(2, use_bias=False)])
    loss = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss)
    rng = np.random.default_rng(2)
    x_train = rng.standard_normal((10, 3)).astype(np.float32)
    y_train = rng.standard_normal((10, 2)).astype(np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(5)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (rng.standard_normal((4, 3)).astype(np.float32),
         rng.standard_normal((4, 2)).astype(np.float32))
    ).batch(4)
    cgd = ConjugateGradientDescentIHVP(influence_model, -1, train_ds)
    calc = FirstOrderInfluenceCalculator(influence_model, train_ds, cgd)
    with pytest.raises(ValueError, match="does not support query-side preconditioning"):
        list(calc.estimate_influence_values_query_batched(test_ds, train_ds))


# ---------------------------------------------------------------------------
# QueryBatchingConfig validation
# ---------------------------------------------------------------------------

def test_query_batching_config_validation_accumulation():
    with pytest.raises(ValueError, match="accumulation_steps"):
        QueryBatchingConfig(query_gradient_accumulation_steps=0).validate()


def test_query_batching_config_validation_rank():
    with pytest.raises(ValueError, match="low_rank"):
        QueryBatchingConfig(query_gradient_low_rank=0).validate()


def test_query_batching_config_valid_defaults():
    QueryBatchingConfig().validate()  # should not raise
