# tests/common/test_kfac_ihvp.py
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for K-FAC and EK-FAC IHVP implementations (TensorFlow backend).

Test strategy
-------------
1. **LayerParameterMap**: verify correct enumeration and flat-index mapping.
2. **Loss scaling diagnostics**: validate expected A/G scaling behavior.
3. **K-FAC/EK-FAC implementation checks**: validate exact agreement with
   ExactIHVP on a linear setting where Hessian and empirical Fisher coincide.
4. **K-FAC/EK-FAC robustness checks**: smoke and determinism tests on a
   nonlinear model where exact Hessian and Fisher differ.
5. **Integration & factories**: verify enum/factory wiring and unsupported HVP behavior.
6. **Checkpointing & partitioning**: validate factor checkpoint and partition options.
"""
import os
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Reduction, MeanSquaredError

from deel.influenciae.common import (
    InfluenceModel,
    ExactIHVP,
    KfacIHVP,
    EkfacIHVP,
    IHVPCalculator,
    KfacIHVPFactory,
)
from deel.influenciae.common.kfac_factors import LayerParameterMap, KroneckerFactors, EKFACFactors


pytestmark = pytest.mark.tensorflow


LINEAR_EXACT_TARGET = 1.0 / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seed():
    tf.random.set_seed(42)
    return 42


def _make_simple_mlp(seed=42):
    """Two-layer Dense model: Dense(3, no bias) -> Dense(2, no bias), float64."""
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(4,), dtype=tf.float64),
        Dense(3, use_bias=False, dtype=tf.float64),
        Dense(2, use_bias=False, dtype=tf.float64),
    ])
    return model


def _make_simple_mlp_with_bias(seed=42):
    """Two-layer Dense model with bias."""
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(4,), dtype=tf.float64),
        Dense(3, use_bias=True, dtype=tf.float64),
        Dense(2, use_bias=True, dtype=tf.float64),
    ])
    return model


def _make_nested_mlp(seed=42):
    """Nested Dense model where one supported layer is inside a submodel."""
    tf.random.set_seed(seed)
    inner_block = Sequential(
        [
            Dense(3, use_bias=False, dtype=tf.float64),
            LayerNormalization(dtype=tf.float64),
        ],
        name="inner_block",
    )
    model = Sequential([
        Input(shape=(4,), dtype=tf.float64),
        inner_block,
        Dense(2, use_bias=False, dtype=tf.float64),
    ])
    return model


def _make_nonlinear_mlp(seed=42):
    """Two-layer nonlinear Dense model used for smoke/determinism tests."""
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(4,), dtype=tf.float64),
        Dense(8, activation="tanh", use_bias=True, dtype=tf.float64),
        Dense(2, use_bias=True, dtype=tf.float64),
    ])
    return model


def _make_single_layer_linear_model(seed=42):
    """Single-layer linear model with zero weights for exact K-FAC checks."""
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(4,), dtype=tf.float64),
        Dense(1, use_bias=False, dtype=tf.float64, kernel_initializer="zeros"),
    ])
    model.layers[0].set_weights([np.zeros((4, 1), dtype=np.float64)])
    return model


def _make_dataset(n_samples=20, n_features=4, n_outputs=2, seed=123, batch_size=5):
    """Create a simple regression dataset as a batched tf.data.Dataset."""
    tf.random.set_seed(seed)
    inputs = tf.random.normal((n_samples, n_features), dtype=tf.float64)
    targets = tf.random.normal((n_samples, n_outputs), dtype=tf.float64)
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    return ds.batch(batch_size)


def _make_linear_exact_dataset(n_samples=64, n_features=4, seed=123, batch_size=16):
    """Dataset where single-layer linear MSE satisfies Hessian == empirical Fisher."""
    inputs = tf.random.stateless_normal(
        (n_samples, n_features),
        seed=(seed, 0),
        dtype=tf.float64,
    )
    targets = tf.fill((n_samples, 1), tf.cast(LINEAR_EXACT_TARGET, tf.float64))
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    return ds.batch(batch_size)


def _assert_query_module_shapes(ihvp, influence_model, batch):
    """Assert that module reshape helpers preserve flat parameter counts."""
    grads = influence_model.batch_jacobian_tensor(tuple(batch))
    grads = tf.reshape(grads, (grads.shape[0], -1))

    reshaped = ihvp.reshape_gradient_per_module(grads)
    preconditioned = ihvp.precondition_gradient_per_module(grads)

    assert ihvp.supports_query_preconditioning is True
    assert reshaped.keys() == preconditioned.keys()
    assert reshaped, "Expected at least one supported module."

    for info in ihvp.layer_map.layers_info:
        layer_idx = info.layer_idx
        if layer_idx not in reshaped:
            continue
        expected_flat = info.flat_end - info.flat_start
        assert int(np.prod(reshaped[layer_idx].shape[1:])) == expected_flat
        assert int(np.prod(preconditioned[layer_idx].shape[1:])) == expected_flat
        assert int(reshaped[layer_idx].shape[0]) == grads.shape[0]
        assert int(preconditioned[layer_idx].shape[0]) == grads.shape[0]


def _sum_squared_error_per_sample(y_true, y_pred):
    """Return per-sample SSE, aligned with PyTorch MSE(reduction='none').sum(-1)."""
    sq_err = tf.math.squared_difference(y_true, y_pred)
    batch_size = tf.shape(sq_err)[0]
    sq_err = tf.reshape(sq_err, (batch_size, -1))
    return tf.reduce_sum(sq_err, axis=1)


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
    reference_np = reference.numpy().T
    approx_np = approx.numpy().T

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


def test_maps_all_dense_layers(seed):
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    layer_map = LayerParameterMap(influence_model, backend)

    # Two Dense layers, both should be supported
    assert layer_map.n_supported_layers == 2

    # First layer: Dense(4 -> 3, no bias)
    # TF Dense kernel shape is (n_in, n_out) = (4, 3) -> 12 params
    info0 = layer_map.layers_info[0]
    assert info0.flat_start == 0
    assert info0.flat_end == 12
    assert not info0.has_bias

    # Second layer: Dense(3 -> 2, no bias) -> 6 params
    info1 = layer_map.layers_info[1]
    assert info1.flat_start == 12
    assert info1.flat_end == 18
    assert not info1.has_bias

def test_target_layers_filter(seed):
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    all_layers = backend.get_layers(model)
    # Find the first Dense layer's index
    dense_indices = [
        i for i, l in enumerate(all_layers) if backend.is_linear_layer(l)
    ]
    assert len(dense_indices) >= 1

    layer_map = LayerParameterMap(influence_model, backend, target_layers=[dense_indices[0]])
    assert layer_map.n_supported_layers == 1

def test_warns_on_unsupported_target_layer(seed):
    """If target_layers includes a non-Dense layer, warn."""
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(4,), dtype=tf.float64),
        Dense(3, use_bias=False, activation="relu", dtype=tf.float64),
        BatchNormalization(dtype=tf.float64),
        Dense(2, use_bias=False, dtype=tf.float64),
    ])
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    # Include all layer indices and verify that unsupported layers are warned/skipped.
    all_layers = backend.get_layers(model)
    all_indices = list(range(len(all_layers)))
    unsupported_indices = [
        i for i, layer in enumerate(all_layers)
        if not backend.is_kfac_supported_layer(layer)
    ]
    supported_indices = [
        i for i, layer in enumerate(all_layers)
        if backend.is_kfac_supported_layer(layer)
    ]

    assert unsupported_indices

    with pytest.warns(UserWarning, match="not a K-FAC-supported"):
        layer_map = LayerParameterMap(influence_model, backend, target_layers=all_indices)

    assert layer_map.n_supported_layers == len(supported_indices)
    assert {info.layer_idx for info in layer_map.layers_info} == set(supported_indices)

def test_recursive_collection_tracks_nested_supported_layers(seed):
    """Recursive collection should include supported nested Dense layers."""
    model = _make_nested_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    top_level_map = LayerParameterMap(influence_model, backend, layer_collection="top_level")
    recursive_map = LayerParameterMap(influence_model, backend, layer_collection="recursive")

    assert top_level_map.n_supported_layers == 1
    assert recursive_map.n_supported_layers >= 2

    recursive_flat_sizes = sorted(info.flat_end - info.flat_start for info in recursive_map.layers_info)
    assert 6 in recursive_flat_sizes
    assert 12 in recursive_flat_sizes

def test_recursive_target_layers_filter(seed):
    """target_layers should apply to recursive traversal indices in recursive mode."""
    model = _make_nested_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    recursive_named_layers = backend.get_named_layers(model, recursive=True)
    nested_dense_idx = None
    for idx, (_, layer) in enumerate(recursive_named_layers):
        if not backend.is_linear_layer(layer):
            continue
        weight, _ = backend.get_layer_weight_and_bias(layer)
        weight_shape = backend.tensor_shape(weight)
        if int(weight_shape[0]) == 4 and int(weight_shape[1]) == 3:
            nested_dense_idx = idx
            break

    assert nested_dense_idx is not None

    recursive_map = LayerParameterMap(
        influence_model,
        backend,
        target_layers=[nested_dense_idx],
        layer_collection="recursive",
    )
    assert recursive_map.n_supported_layers == 1
    assert recursive_map.layers_info[0].flat_end - recursive_map.layers_info[0].flat_start == 12

def test_invalid_layer_collection_raises(seed):
    """An invalid layer_collection value should raise ValueError."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    backend = influence_model.backend

    with pytest.raises(ValueError, match="layer_collection"):
        LayerParameterMap(influence_model, backend, layer_collection="invalid")


# ---------------------------------------------------------------------------
# 2. Loss scaling diagnostics
# ---------------------------------------------------------------------------


def test_sum_squared_error_alignment_with_mse_none():
    """SSE should be n_outputs times MeanSquaredError(reduction=NONE)."""
    y_true = tf.constant([[0.0, 1.0], [2.0, 3.0]], dtype=tf.float64)
    y_pred = tf.constant([[1.0, -1.0], [1.5, 2.5]], dtype=tf.float64)

    mse_none = MeanSquaredError(reduction=Reduction.NONE)(y_true, y_pred)
    sse = _sum_squared_error_per_sample(y_true, y_pred)

    n_outputs = y_true.shape[-1]
    assert n_outputs is not None
    assert np.allclose(
        sse.numpy(),
        float(n_outputs) * mse_none.numpy(),
        atol=1e-12,
    )

def test_kfac_factor_scaling_between_mean_and_sum_losses(seed):
    """A stays unchanged while G scales by n_outputs^2 between mean and SSE losses."""
    mean_loss = MeanSquaredError(reduction=Reduction.NONE)
    sum_loss = _sum_squared_error_per_sample

    # Use exactly the same initialized weights for both models so any
    # factor difference is only due to loss scaling.
    mean_keras_model = _make_simple_mlp(seed=seed)
    sum_keras_model = tf.keras.models.clone_model(mean_keras_model)
    sum_keras_model.set_weights(mean_keras_model.get_weights())

    for mean_w, sum_w in zip(
        mean_keras_model.get_weights(),
        sum_keras_model.get_weights(),
    ):
        assert np.array_equal(mean_w, sum_w)

    # Use the exact same tensors for both datasets.
    train_inputs = tf.random.stateless_normal(
        (12, 4),
        seed=(123, 0),
        dtype=tf.float64,
    )
    train_targets = tf.random.stateless_normal(
        (12, 2),
        seed=(456, 0),
        dtype=tf.float64,
    )
    mean_train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets)).batch(4)
    sum_train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets)).batch(4)

    mean_model = InfluenceModel(
        mean_keras_model,
        start_layer=0,
        last_layer=-1,
        loss_function=mean_loss,
    )
    sum_model = InfluenceModel(
        sum_keras_model,
        start_layer=0,
        last_layer=-1,
        loss_function=sum_loss,
    )

    mean_layer_map = LayerParameterMap(mean_model, mean_model.backend)
    sum_layer_map = LayerParameterMap(sum_model, sum_model.backend)

    mean_factors = KroneckerFactors(
        mean_model,
        mean_train_ds,
        mean_model.backend,
        mean_layer_map,
    )
    sum_factors = KroneckerFactors(
        sum_model,
        sum_train_ds,
        sum_model.backend,
        sum_layer_map,
    )

    assert len(mean_layer_map.layers_info) == len(sum_layer_map.layers_info)
    n_outputs = 2.0

    for mean_info, sum_info in zip(mean_layer_map.layers_info, sum_layer_map.layers_info):
        mean_idx = mean_info.layer_idx
        sum_idx = sum_info.layer_idx

        assert np.allclose(
            sum_factors.A[sum_idx].numpy(),
            mean_factors.A[mean_idx].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
        assert np.allclose(
            sum_factors.G[sum_idx].numpy(),
            (n_outputs ** 2) * mean_factors.G[mean_idx].numpy(),
            rtol=1e-5,
            atol=1e-7,
        )


# ---------------------------------------------------------------------------
# 3. K-FAC IHVP validation
# ---------------------------------------------------------------------------


def test_kfac_query_preconditioning_module_shapes(seed):
    """K-FAC should expose per-module reshape helpers for query batching."""
    model = _make_simple_mlp(seed)
    train_ds = _make_dataset(n_samples=12, seed=seed, batch_size=4)
    influence_model = InfluenceModel(
        model,
        start_layer=-1,
        loss_function=MeanSquaredError(reduction=Reduction.NONE),
    )

    kfac = KfacIHVP(influence_model, train_ds, damping=1e-3)
    batch = next(iter(train_ds))
    _assert_query_module_shapes(kfac, influence_model, batch)


def test_ekfac_query_preconditioning_module_shapes(seed):
    """EK-FAC should expose per-module reshape helpers for query batching."""
    model = _make_simple_mlp(seed)
    train_ds = _make_dataset(n_samples=12, seed=seed, batch_size=4)
    influence_model = InfluenceModel(
        model,
        start_layer=-1,
        loss_function=MeanSquaredError(reduction=Reduction.NONE),
    )

    ekfac = EkfacIHVP(influence_model, train_ds, damping=1e-3, n_ekfac_samples=None)
    batch = next(iter(train_ds))
    _assert_query_module_shapes(ekfac, influence_model, batch)


def test_kfac_matches_exact_on_linear_model(seed):
    """On a crafted linear setting, K-FAC should match ExactIHVP closely."""
    model = _make_single_layer_linear_model(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_linear_exact_dataset(n_samples=96, n_features=4, seed=123, batch_size=16)
    batch = next(iter(train_ds))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    exact_ihvp = ExactIHVP(influence_model, train_ds)
    kfac_ihvp = KfacIHVP(
        influence_model,
        train_ds,
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
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=128, n_features=4, n_outputs=2, seed=321, batch_size=16)
    batch = next(iter(_make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=654, batch_size=8)))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    kfac_ihvp = KfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        fisher_type=fisher_type,
    )

    result = kfac_ihvp._compute_ihvp_single_batch(batch)
    result_np = result.numpy()

    assert result.shape == (influence_model.nb_params, batch[0].shape[0])
    assert np.all(np.isfinite(result_np))
    assert np.median(np.linalg.norm(result_np, axis=0)) > 1e-10

def test_kfac_ihvp_deterministic(seed):
    """Empirical K-FAC should be deterministic for fixed data and seeds."""
    model = _make_nonlinear_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=96, n_features=4, n_outputs=2, seed=2024, batch_size=16)
    batch = next(iter(_make_dataset(n_samples=18, n_features=4, n_outputs=2, seed=2025, batch_size=6)))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    first = KfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        fisher_type="empirical",
    )
    second = KfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        fisher_type="empirical",
    )

    first_result = first._compute_ihvp_single_batch(batch).numpy()
    second_result = second._compute_ihvp_single_batch(batch).numpy()
    assert np.allclose(first_result, second_result, rtol=1e-10, atol=1e-12)

def test_kfac_enum_from_string():
    """IHVPCalculator.from_string('kfac') should return Kfac enum."""
    calc = IHVPCalculator.from_string('kfac')
    assert calc is IHVPCalculator.Kfac
    assert calc.value is KfacIHVP


@pytest.mark.parametrize("ihvp_cls,kwargs", [
    (KfacIHVP, {"damping": 1e-8, "fisher_type": "empirical"}),
    (EkfacIHVP, {"damping": 1e-8, "fisher_type": "empirical", "n_ekfac_samples": 96}),
])
def test_precondition_gradient_per_module_correctness(ihvp_cls, kwargs, seed):
    """precondition_gradient_per_module must agree with ExactIHVP on a linear model.

    On the single-layer linear + constant-target dataset the empirical Fisher
    equals the Hessian, so K-FAC / EK-FAC should match ExactIHVP closely.
    The per-module preconditioned gradients are concatenated and compared
    against the flat ExactIHVP result using cosine similarity and relative error.
    """
    model = _make_single_layer_linear_model(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_linear_exact_dataset(n_samples=96, n_features=4, seed=123, batch_size=16)
    batch = next(iter(train_ds))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    exact_ihvp = ExactIHVP(influence_model, train_ds)
    approx_ihvp = ihvp_cls(influence_model, train_ds, **kwargs)

    # Reference: flat IHVP from ExactIHVP — shape (n_params, batch_size)
    exact_result = exact_ihvp._compute_ihvp_single_batch(batch)

    # Compute flat gradients, then apply per-module preconditioning
    grads = influence_model.batch_jacobian_tensor(tuple(batch))
    grads_flat = tf.reshape(grads, (grads.shape[0], -1))
    per_module = approx_ihvp.precondition_gradient_per_module(grads_flat)

    # Re-assemble the per-module tensors into a single flat vector per sample.
    # layer_map.layers_info is ordered by flat_start; concatenate slices in order.
    ordered_infos = sorted(
        [info for info in approx_ihvp.layer_map.layers_info if info.layer_idx in per_module],
        key=lambda info: info.flat_start,
    )
    pieces = [
        tf.reshape(per_module[info.layer_idx], (grads_flat.shape[0], -1))
        for info in ordered_infos
    ]
    approx_flat = tf.concat(pieces, axis=1)  # (batch_size, n_params)
    # Transpose to (n_params, batch_size) to match exact_result convention
    approx_result = tf.transpose(approx_flat)

    metrics = _compute_alignment_metrics(exact_result, approx_result)
    assert metrics["median_cosine"] > 0.99, metrics
    assert metrics["median_relative_error"] < 1.5e-1, metrics


# ---------------------------------------------------------------------------
# 4. EK-FAC IHVP validation
# ---------------------------------------------------------------------------


def test_ekfac_matches_exact_on_linear_model(seed):
    """On a crafted linear setting, EK-FAC should match ExactIHVP closely."""
    model = _make_single_layer_linear_model(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_linear_exact_dataset(n_samples=96, n_features=4, seed=123, batch_size=16)
    batch = next(iter(train_ds))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    exact_ihvp = ExactIHVP(influence_model, train_ds)
    ekfac_ihvp = EkfacIHVP(
        influence_model,
        train_ds,
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
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=128, n_features=4, n_outputs=2, seed=321, batch_size=16)
    batch = next(iter(_make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=654, batch_size=8)))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    ekfac_ihvp = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        fisher_type=fisher_type,
        n_ekfac_samples=128,
    )

    result = ekfac_ihvp._compute_ihvp_single_batch(batch)
    result_np = result.numpy()

    assert result.shape == (influence_model.nb_params, batch[0].shape[0])
    assert np.all(np.isfinite(result_np))
    assert np.median(np.linalg.norm(result_np, axis=0)) > 1e-10

def test_ekfac_ihvp_deterministic(seed):
    """Empirical EK-FAC should be deterministic for fixed data and seeds."""
    model = _make_nonlinear_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=96, n_features=4, n_outputs=2, seed=2024, batch_size=16)
    batch = next(iter(_make_dataset(n_samples=18, n_features=4, n_outputs=2, seed=2025, batch_size=6)))

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    first = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        fisher_type="empirical",
        n_ekfac_samples=96,
    )
    second = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        fisher_type="empirical",
        n_ekfac_samples=96,
    )

    first_result = first._compute_ihvp_single_batch(batch).numpy()
    second_result = second._compute_ihvp_single_batch(batch).numpy()
    assert np.allclose(first_result, second_result, rtol=1e-10, atol=1e-12)

def test_ekfac_enum_from_string():
    """IHVPCalculator.from_string('ekfac') should return Ekfac enum."""
    calc = IHVPCalculator.from_string('ekfac')
    assert calc is IHVPCalculator.Ekfac
    assert calc.value is EkfacIHVP


# ---------------------------------------------------------------------------
# 5. Factor checkpointing
# ---------------------------------------------------------------------------


def test_kfac_factor_checkpoint_roundtrip(seed, tmp_path):
    """K-FAC checkpoint load should reproduce the same IHVP output."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=16, n_features=4, n_outputs=2, seed=123, batch_size=4)
    batch = next(iter(train_ds))
    checkpoint_dir = str(tmp_path / "kfac_factors")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    computed = KfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        factors_path=checkpoint_dir,
    )
    computed_result = computed._compute_ihvp_single_batch(batch)

    loaded = KfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        factors_path=checkpoint_dir,
    )
    loaded_result = loaded._compute_ihvp_single_batch(batch)

    assert np.allclose(computed_result.numpy(), loaded_result.numpy(), atol=1e-10, rtol=1e-8)

def test_kfac_checkpoint_load_skips_recompute(seed, tmp_path, monkeypatch):
    """Existing K-FAC checkpoint should avoid factor recomputation."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=123, batch_size=4)
    checkpoint_dir = str(tmp_path / "kfac_skip_recompute")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    _ = KfacIHVP(influence_model, train_ds, damping=1e-3, factors_path=checkpoint_dir)

    def _raise_if_recomputed(*_args, **_kwargs):
        raise AssertionError("K-FAC factors were recomputed instead of loaded from checkpoint")

    monkeypatch.setattr(KroneckerFactors, "_compute_factors", _raise_if_recomputed)

    loaded = KfacIHVP(influence_model, train_ds, damping=1e-3, factors_path=checkpoint_dir)
    assert isinstance(loaded, KfacIHVP)

def test_kfac_checkpoint_layer_mismatch_raises(seed, tmp_path):
    """Layer signature mismatch should raise a clear checkpoint error."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=321, batch_size=4)
    checkpoint_dir = str(tmp_path / "kfac_mismatch")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    _ = KfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        target_layers=[0],
        factors_path=checkpoint_dir,
    )

    with pytest.raises(ValueError, match="layer signature mismatch"):
        KfacIHVP(
            influence_model,
            train_ds,
            damping=1e-3,
            target_layers=[1],
            factors_path=checkpoint_dir,
        )

def test_ekfac_factor_checkpoint_roundtrip(seed, tmp_path):
    """EK-FAC checkpoint load should reproduce the same IHVP output."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=16, n_features=4, n_outputs=2, seed=456, batch_size=4)
    batch = next(iter(train_ds))
    checkpoint_dir = str(tmp_path / "ekfac_factors")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    computed = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        factors_path=checkpoint_dir,
        n_ekfac_samples=None,
    )
    computed_result = computed._compute_ihvp_single_batch(batch)

    loaded = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        factors_path=checkpoint_dir,
        n_ekfac_samples=None,
    )
    loaded_result = loaded._compute_ihvp_single_batch(batch)

    assert np.allclose(computed_result.numpy(), loaded_result.numpy(), atol=1e-10, rtol=1e-8)

def test_ekfac_checkpoint_load_skips_recompute(seed, tmp_path, monkeypatch):
    """Existing EK-FAC checkpoint should avoid corrected-eigenvalue recomputation."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=654, batch_size=4)
    checkpoint_dir = str(tmp_path / "ekfac_skip_recompute")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    _ = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        factors_path=checkpoint_dir,
        n_ekfac_samples=None,
    )

    def _raise_if_recomputed(*_args, **_kwargs):
        raise AssertionError("EK-FAC factors were recomputed instead of loaded from checkpoint")

    monkeypatch.setattr(EKFACFactors, "_estimate_corrected_eigenvalues", _raise_if_recomputed)

    loaded = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        factors_path=checkpoint_dir,
        n_ekfac_samples=None,
    )
    assert isinstance(loaded, EkfacIHVP)

def test_kfac_factory_uses_checkpoint(seed, tmp_path, monkeypatch):
    """Factory should pass checkpoint options through to K-FAC IHVP."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=777, batch_size=4)
    checkpoint_dir = str(tmp_path / "kfac_factory_checkpoint")

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    factory = KfacIHVPFactory(damping=1e-3, factors_path=checkpoint_dir)
    _ = factory.build(influence_model, train_ds)
    assert (tmp_path / "kfac_factory_checkpoint" / "metadata.json").exists()

    def _raise_if_recomputed(*_args, **_kwargs):
        raise AssertionError("Factory build recomputed factors instead of loading checkpoint")

    monkeypatch.setattr(KroneckerFactors, "_compute_factors", _raise_if_recomputed)
    loaded = factory.build(influence_model, train_ds)
    assert isinstance(loaded, KfacIHVP)


# ---------------------------------------------------------------------------
# 6. Integration: HVP not supported
# ---------------------------------------------------------------------------


def test_kfac_hvp_raises(seed):
    """K-FAC compute_hvp should raise NotImplementedError."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=10, n_features=4, n_outputs=2, seed=123, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    kfac = KfacIHVP(influence_model, train_ds, damping=1e-3)

    batch = next(iter(train_ds))
    with pytest.raises(NotImplementedError):
        kfac._compute_hvp_single_batch(batch)

def test_ekfac_hvp_raises(seed):
    """EK-FAC compute_hvp should raise NotImplementedError."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=10, n_features=4, n_outputs=2, seed=123, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    ekfac = EkfacIHVP(influence_model, train_ds, damping=1e-3)

    batch = next(iter(train_ds))
    with pytest.raises(NotImplementedError):
        ekfac._compute_hvp_single_batch(batch)


# ---------------------------------------------------------------------------
# 7. Model with bias
# ---------------------------------------------------------------------------


def test_kfac_with_bias_runs(seed):
    """K-FAC should work with biased layers without errors."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=10, n_features=4, n_outputs=2, seed=123, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    kfac = KfacIHVP(influence_model, train_ds, damping=1e-3)

    batch = next(iter(train_ds))
    result = kfac._compute_ihvp_single_batch(batch)
    # nb_params = 4*3 + 3 + 3*2 + 2 = 23
    assert result.shape[0] == influence_model.nb_params


# ---------------------------------------------------------------------------
# 8. Memory partitioning options
# ---------------------------------------------------------------------------


def test_invalid_partition_sizes_raise(seed):
    """Invalid partition sizes should raise explicit errors."""
    model = _make_simple_mlp(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=12, n_features=4, n_outputs=2, seed=123, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    with pytest.raises(ValueError, match="module_partition_size"):
        KfacIHVP(
            influence_model,
            train_ds,
            module_partition_size=0,
        )

    with pytest.raises(ValueError, match="data_partition_size"):
        KfacIHVP(
            influence_model,
            train_ds,
            data_partition_size=0,
        )

    with pytest.raises(ValueError, match="accumulator_offload_mode"):
        KfacIHVP(
            influence_model,
            train_ds,
            accumulator_offload_mode="invalid",
        )

def test_kfac_memory_options_match_baseline(seed):
    """K-FAC memory partitioning options should preserve results."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=777, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    baseline = KfacIHVP(influence_model, train_ds, damping=1e-3)
    partitioned = KfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="memory",
    )
    assert partitioned.factors._factor_checkpoint is not None

    for info in baseline.layer_map.layers_info:
        idx = info.layer_idx
        assert np.allclose(
            baseline.factors.A[idx].numpy(),
            partitioned.factors.A[idx].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
        assert np.allclose(
            baseline.factors.G[idx].numpy(),
            partitioned.factors.G[idx].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

    batch = next(iter(train_ds))
    baseline_ihvp = baseline._compute_ihvp_single_batch(batch)
    partitioned_ihvp = partitioned._compute_ihvp_single_batch(batch)
    assert np.allclose(
        baseline_ihvp.numpy(),
        partitioned_ihvp.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_ekfac_memory_options_match_baseline(seed):
    """EK-FAC memory partitioning options should preserve results."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=999, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    baseline = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        n_ekfac_samples=24,
    )
    partitioned = EkfacIHVP(
        influence_model,
        train_ds,
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
        assert np.allclose(
            baseline.factors.Lambda_corrected[idx].numpy(),
            partitioned.factors.Lambda_corrected[idx].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

    batch = next(iter(train_ds))
    baseline_ihvp = baseline._compute_ihvp_single_batch(batch)
    partitioned_ihvp = partitioned._compute_ihvp_single_batch(batch)
    assert np.allclose(
        baseline_ihvp.numpy(),
        partitioned_ihvp.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_kfac_disk_offload_matches_baseline(seed, tmp_path):
    """K-FAC disk accumulator offload should preserve results."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=432, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    baseline = KfacIHVP(influence_model, train_ds, damping=1e-3)
    partitioned = KfacIHVP(
        influence_model,
        train_ds,
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
        assert np.allclose(
            baseline.factors.A[idx].numpy(),
            partitioned.factors.A[idx].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
        assert np.allclose(
            baseline.factors.G[idx].numpy(),
            partitioned.factors.G[idx].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

def test_ekfac_disk_offload_matches_baseline(seed, tmp_path):
    """EK-FAC disk accumulator offload should preserve results."""
    model = _make_simple_mlp_with_bias(seed=seed)
    loss_fn = MeanSquaredError(reduction=Reduction.NONE)
    train_ds = _make_dataset(n_samples=24, n_features=4, n_outputs=2, seed=876, batch_size=4)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1, loss_function=loss_fn)
    baseline = EkfacIHVP(
        influence_model,
        train_ds,
        damping=1e-3,
        n_ekfac_samples=24,
    )
    partitioned = EkfacIHVP(
        influence_model,
        train_ds,
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
        assert np.allclose(
            baseline.factors.Lambda_corrected[idx].numpy(),
            partitioned.factors.Lambda_corrected[idx].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
