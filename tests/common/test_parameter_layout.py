# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import pytest

from deel.influenciae.common.parameter_layout import ParameterLayout
from deel.influenciae.trackstar.projection import (
    ProjectionBlock,
    ProjectionTerm,
    TrackStarProjectionPlan,
)


pytestmark = pytest.mark.backend_agnostic


class _NumpyBackend:
    @staticmethod
    def tensor_shape(tensor):
        return tensor.shape

    @staticmethod
    def reshape(tensor, shape):
        return tensor.reshape(shape)

    @staticmethod
    def concat(tensors, axis=0):
        return np.concatenate(tensors, axis=axis)


def test_layout_metadata_and_unique_names():
    layout = ParameterLayout.from_shapes(((2, 3), (3,), ()), ("weight", "bias", "bias"))

    assert layout.parameter_names == ("weight", "bias", "bias#2")
    assert layout.parameter_shapes == ((2, 3), (3,), ())
    assert layout.parameter_sizes == (6, 3, 1)
    assert layout.parameter_slices == (slice(0, 6), slice(6, 9), slice(9, 10))
    assert layout.num_parameters == 3
    assert layout.total_size == 10


def test_split_and_flatten_round_trip_with_leading_dimensions():
    layout = ParameterLayout.from_shapes(((2, 2), (2,), ()))
    flat = np.arange(2 * 3 * 7).reshape(2, 3, 7)

    components = layout.split_flat(flat, _NumpyBackend())

    assert tuple(component.shape for component in components) == ((2, 3, 2, 2), (2, 3, 2), (2, 3))
    np.testing.assert_array_equal(layout.flatten_components(components, _NumpyBackend()), flat)


def test_layout_rejects_mismatched_shapes_and_widths():
    layout = ParameterLayout.from_shapes(((2, 2), (2,)))

    with pytest.raises(ValueError, match="final dimension 6"):
        layout.split_flat(np.ones((3, 5)), _NumpyBackend())
    with pytest.raises(ValueError, match="must end with shape"):
        layout.flatten_components((np.ones((3, 4)), np.ones((3, 2))), _NumpyBackend())
    with pytest.raises(ValueError, match="number of parameter shapes"):
        ParameterLayout.from_shapes(((2,),), ("first", "extra"))


def test_layout_satisfies_trackstar_projection_protocol():
    layout = ParameterLayout.from_shapes(((2, 2),), ("weight",))
    term = ProjectionTerm(
        "weight",
        (0,),
        ((2, 2),),
        (1, 1),
        left_matrix=np.ones((1, 2)),
        right_matrix=np.ones((2, 1)),
    )

    plan = TrackStarProjectionPlan.from_layout(layout, (ProjectionBlock("layer", (term,)),))

    assert plan.parameter_shapes == layout.parameter_shapes
