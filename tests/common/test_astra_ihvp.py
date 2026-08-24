# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Focused ASTRA IHVP tests for TensorFlow."""
import numpy as np
import pytest
import tensorflow as tf

from deel.influenciae.common import AstraConfig, AstraIHVP, EkfacIHVP, InfluenceModel


pytestmark = pytest.mark.tensorflow


@pytest.fixture(scope="module")
def astra_setup():
    tf.random.set_seed(7)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((2,), dtype=tf.float64),
        tf.keras.layers.Dense(1, use_bias=False, dtype=tf.float64),
    ])
    inputs = tf.constant([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]], tf.float64)
    targets = tf.constant([[1.], [-1.], [0.5], [0.]], tf.float64)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)
    influence_model = InfluenceModel(
        model, start_layer=0, last_layer=-1,
        loss_function=tf.keras.losses.MeanSquaredError(reduction="none"),
    )
    ekfac = EkfacIHVP(influence_model, dataset, damping=0.2)
    return influence_model, dataset, ekfac, (inputs[:2], targets[:2])


def _astra(setup, **config_kwargs):
    model, dataset, ekfac, _ = setup
    config = AstraConfig(preconditioner_damping=0.2, **config_kwargs)
    return AstraIHVP(model, dataset, config=config, ekfac_factors=ekfac.factors)


def test_zero_iteration_and_one_step_initializations(astra_setup):
    _, _, ekfac, batch = astra_setup
    rhs = tf.constant([[1., -2.], [0.5, 3.]], tf.float64)
    expected = ekfac.precondition_gradient(rhs)
    initialized = _astra(astra_setup, n_iterations=0).precondition_gradient(rhs)
    one_step = _astra(
        astra_setup, n_iterations=1, learning_rate=1.0,
        damping=0.2, initialize_from_ekfac=False,
    ).precondition_gradient(rhs)
    np.testing.assert_allclose(initialized.numpy(), expected.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(one_step.numpy(), expected.numpy(), rtol=1e-10, atol=1e-10)

    from_gradient = _astra(astra_setup, n_iterations=0)._compute_ihvp_single_batch(batch)
    grads = astra_setup[0].batch_jacobian_tensor(batch)
    from_vectors = _astra(astra_setup, n_iterations=0).precondition_gradient(grads)
    np.testing.assert_allclose(from_gradient.numpy(), from_vectors.numpy(), rtol=1e-10, atol=1e-10)


def test_matrix_rhs_chunk_parity_and_shared_sampler(astra_setup):
    model, dataset, ekfac, batch = astra_setup
    rhs = tf.constant([[1., 2.], [-2., 1.], [0.5, -1.]], tf.float64)
    calls = []

    def sampler(step):
        calls.append(step)
        return batch

    common = dict(
        n_iterations=3, learning_rate=0.05, damping=0.2,
        preconditioner_damping=0.2, initialize_from_ekfac=False,
    )
    chunked = AstraIHVP(
        model, dataset, config=AstraConfig(rhs_chunk_size=1, **common),
        ekfac_factors=ekfac.factors, curvature_batch_sampler=sampler,
    ).precondition_gradient(rhs)
    unchunked = AstraIHVP(
        model, dataset, config=AstraConfig(**common),
        ekfac_factors=ekfac.factors, curvature_batch_sampler=lambda _: batch,
    ).precondition_gradient(rhs)
    assert calls == [0, 1, 2]
    assert tuple(chunked.shape) == (2, 3)
    np.testing.assert_allclose(chunked.numpy(), unchunked.numpy(), rtol=1e-10, atol=1e-10)


def test_strict_coverage_rejects_an_empty_layer_selection(astra_setup):
    model, dataset, _, _ = astra_setup
    with pytest.warns(UserWarning, match="No supported layers"):
        with pytest.raises(ValueError, match="Strict EK-FAC coverage"):
            AstraIHVP(model, dataset, target_layers=[])


def test_compute_ihvp_dataset_api_traces_with_dynamic_batch_size(astra_setup):
    """The public TensorFlow dataset API should remain graph-traceable."""
    model, dataset, ekfac, batch = astra_setup
    astra = AstraIHVP(
        model,
        dataset,
        config=AstraConfig(
            damping=0.2,
            preconditioner_damping=0.2,
            n_iterations=1,
            learning_rate=0.05,
        ),
        ekfac_factors=ekfac.factors,
        curvature_batch_sampler=lambda _step: batch,
    )
    rhs = tf.constant([[1., -2.], [0.5, 3.], [-1., 0.25]], tf.float64)
    rhs_dataset = tf.data.Dataset.from_tensor_slices(rhs).batch(2)

    actual = list(astra.compute_ihvp(rhs_dataset, use_gradient=False))
    expected = [
        astra.precondition_gradient(rhs[:2]),
        astra.precondition_gradient(rhs[2:]),
    ]

    assert len(actual) == 2
    for actual_batch, expected_batch in zip(actual, expected):
        np.testing.assert_allclose(
            actual_batch.numpy(), expected_batch.numpy(), rtol=1e-10, atol=1e-10
        )


def test_compute_ihvp_dataset_api_uses_default_curvature_stream(astra_setup):
    """The default seeded curvature stream should also execute under tf.data."""
    # Five iterations force the two-batch curvature dataset to restart twice.
    astra = _astra(astra_setup, n_iterations=5, learning_rate=0.05)
    rhs = tf.constant([[1., -2.], [0.5, 3.], [-1., 0.25]], tf.float64)
    results = list(
        astra.compute_ihvp(
            tf.data.Dataset.from_tensor_slices(rhs).batch(2),
            use_gradient=False,
        )
    )

    assert [tuple(result.shape) for result in results] == [(2, 2), (2, 1)]
    assert all(bool(tf.reduce_all(tf.math.is_finite(result)).numpy()) for result in results)
