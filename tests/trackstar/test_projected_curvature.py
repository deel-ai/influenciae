# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import pytest

from deel.influenciae.trackstar.projected_curvature import (
    AutoMix,
    FixedMix,
    ProjectedGramAccumulator,
    build_projected_curvature,
    inverse_sqrt_psd,
)


pytestmark = pytest.mark.backend_agnostic


def _snapshot(rows):
    accumulator = ProjectedGramAccumulator({"a": slice(0, 2), "b": slice(2, 3)})
    for batch in rows:
        accumulator.update(np.asarray(batch, dtype=np.float64))
    return accumulator.snapshot()


def test_streaming_gram_sums_and_direct_fixed_mix_formula():
    train_rows = np.array([[1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    eval_rows = np.array([[2.0, 1.0, 1.0]])
    train = _snapshot((train_rows[:1], train_rows[1:]))
    evaluation = _snapshot((eval_rows,))
    curvature = build_projected_curvature(train, evaluation, FixedMix(0.25), rcond=0.0)

    direct = 0.75 * (train_rows[:, :2].T @ train_rows[:, :2] / 2) + 0.25 * (
        eval_rows[:, :2].T @ eval_rows[:, :2]
    )
    np.testing.assert_allclose(
        curvature.transforms[0] @ direct @ curvature.transforms[0], np.eye(2), atol=1e-14
    )
    assert train.count == 2
    assert curvature.alphas == (0.25, 0.25)


def test_auto_mix_equalizes_weighted_spectral_radii():
    train = _snapshot((np.array([[2.0, 0.0, 0.0]]),))  # block-a spectral radius 4
    evaluation = _snapshot((np.array([[0.0, 1.0, 0.0]]),))  # block-a radius 1
    curvature = build_projected_curvature(train, evaluation, AutoMix())
    assert curvature.alphas[0] == pytest.approx(4.0 / 5.0)
    assert (1.0 - curvature.alphas[0]) * 4.0 == pytest.approx(curvature.alphas[0] * 1.0)
    assert curvature.alphas[1] == 0.5


def test_moore_penrose_inverse_sqrt_handles_rank_deficiency_without_ridge():
    matrix = np.array([[4.0, 0.0], [0.0, 0.0]])
    transform = inverse_sqrt_psd(matrix)
    np.testing.assert_allclose(transform, np.diag([0.5, 0.0]))
    np.testing.assert_allclose(transform @ matrix @ transform, np.diag([1.0, 0.0]))


def test_psd_roundoff_is_clipped_but_material_negative_eigenvalue_is_rejected():
    np.testing.assert_allclose(inverse_sqrt_psd(np.diag([1.0, -1e-12])), np.diag([1.0, 0.0]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        inverse_sqrt_psd(np.diag([1.0, -1e-3]))


def test_transform_is_blockwise_then_rows_are_normalized_globally_and_safely():
    train = _snapshot((np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]]),))
    evaluation = _snapshot((np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]]),))
    curvature = build_projected_curvature(train, evaluation, FixedMix(0.5))
    rows = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    raw = curvature.apply(rows, normalize=False)
    normalized = curvature.apply(rows)

    np.testing.assert_allclose(np.linalg.norm(normalized[0]), 1.0)
    np.testing.assert_allclose(normalized[0], raw[0] / np.linalg.norm(raw[0]))
    np.testing.assert_array_equal(normalized[1], np.zeros(3))
    assert np.all(np.isfinite(normalized))
