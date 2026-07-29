# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import pytest

from deel.influenciae.trackstar.projection import (
    ProjectionBlock,
    ProjectionTerm,
    TrackStarProjectionPlan,
)


pytestmark = pytest.mark.backend_agnostic


def _term(name, index, shape, projected_shape, left, right, output_axis=0):
    return ProjectionTerm(name, (index,), (shape,), projected_shape, output_axis, left, right)


def test_two_sided_projection_matches_vec_kronecker_identity():
    weight = np.arange(6.0).reshape(1, 2, 3)
    left = np.array([[1.0, 2.0], [-1.0, 0.5]])
    right = np.array([[2.0], [1.0], [-1.0]])
    term = _term("weight", 0, (2, 3), (2, 1), left, right)
    plan = TrackStarProjectionPlan(((2, 3),), (ProjectionBlock("layer", (term,)),))

    actual = plan.apply((weight,))[0]
    explicit_projection = np.kron(left, right.T)
    expected = explicit_projection @ weight[0].reshape(-1)
    np.testing.assert_allclose(actual, expected)


def test_optimizer_correction_happens_before_projection():
    gradient = np.array([[[2.0, 8.0], [18.0, 32.0]]])
    moment = np.array([[1.0, 4.0], [9.0, 16.0]])
    left = np.array([[1.0, 1.0]])
    right = np.array([[1.0], [2.0]])
    term = _term("weight", 0, (2, 2), (1, 1), left, right)
    plan = TrackStarProjectionPlan(((2, 2),), (ProjectionBlock("layer", (term,)),))

    actual = plan.apply((gradient,), (moment,))
    expected = left @ (gradient[0] / np.sqrt(moment)) @ right
    np.testing.assert_allclose(actual, expected.reshape(1, 1))

    class OptimizerStateAdapter:
        parameter_second_moments = (moment,)

    np.testing.assert_allclose(plan.apply((gradient,), OptimizerStateAdapter()), actual)


def test_weight_bias_packing_and_term_sum_with_canonical_conv_view():
    # TF-style convolution kernel: (height, width, in_channels, out_channels).
    weight = np.arange(12.0).reshape(1, 2, 1, 2, 3)
    bias = np.array([[10.0, 20.0, 30.0]])
    packed_left = np.array([[1.0, 0.0, -1.0]])
    packed_right = np.ones((5, 1))
    packed = ProjectionTerm(
        "conv+bias", (0, 1), ((2, 1, 2, 3), (3,)), (1, 1), 3,
        packed_left, packed_right,
    )
    scalar = _term("scalar", 2, (), (1, 1), np.ones((1, 1)), np.ones((1, 1)))
    plan = TrackStarProjectionPlan(
        ((2, 1, 2, 3), (3,), ()),
        (ProjectionBlock("combined", (packed, scalar)),),
    )

    actual = plan.apply((weight, bias, np.array([7.0])))
    canonical_weight = np.moveaxis(weight, 4, 1).reshape(1, 3, 4)
    canonical = np.concatenate((canonical_weight, bias[..., None]), axis=2)
    expected = packed_left @ canonical @ packed_right + 7.0
    np.testing.assert_allclose(actual, expected.reshape(1, 1))


def test_plan_requires_exactly_once_coverage_and_returns_block_slices():
    term0 = _term("a", 0, (2,), (1, 2), np.ones((1, 2)), np.ones((1, 2)))
    term1 = _term("b", 1, (), (1, 1), np.ones((1, 1)), np.ones((1, 1)))
    plan = TrackStarProjectionPlan(
        ((2,), ()), (ProjectionBlock("first", (term0,)), ProjectionBlock("second", (term1,)))
    )
    assert plan.block_slices == {"first": slice(0, 2), "second": slice(2, 3)}
    assert plan.apply((np.ones((3, 2)), np.ones(3))).shape == (3, 3)

    with pytest.raises(ValueError, match="exactly once"):
        TrackStarProjectionPlan(((2,), ()), (ProjectionBlock("bad", (term0,)),))


def test_hash_derived_gaussians_are_reproducible_and_name_separated():
    first = ProjectionTerm("a", (0,), ((3, 2),), (2, 2))
    same = ProjectionTerm("a", (0,), ((3, 2),), (2, 2))
    other = ProjectionTerm("b", (0,), ((3, 2),), (2, 2))
    first_matrices = first.matrices(17, "block")
    same_matrices = same.matrices(17, "block")
    other_matrices = other.matrices(17, "block")
    np.testing.assert_array_equal(first_matrices[0], same_matrices[0])
    np.testing.assert_array_equal(first_matrices[1], same_matrices[1])
    assert not np.array_equal(first_matrices[0], other_matrices[0])


def test_optimizer_correction_rejects_non_finite_and_zero_denominators():
    term = _term("weight", 0, (1,), (1, 1), np.ones((1, 1)), np.ones((1, 1)))
    plan = TrackStarProjectionPlan(((1,),), (ProjectionBlock("layer", (term,)),))
    gradient = np.ones((1, 1))

    with pytest.raises(ValueError, match="finite and non-negative"):
        plan.apply((gradient,), (np.ones(1),), optimizer_epsilon=np.inf)
    with pytest.raises(ValueError, match="finite values"):
        plan.apply((gradient,), (np.array([np.nan]),))
    with pytest.raises(ValueError, match="strictly positive"):
        plan.apply((gradient,), (np.zeros(1),))

    np.testing.assert_allclose(
        plan.apply((gradient,), (np.zeros(1),), optimizer_epsilon=1.0),
        np.ones((1, 1)),
    )
