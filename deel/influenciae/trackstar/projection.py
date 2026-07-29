# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Backend-neutral, structured random projection for TrackStar gradients.

The core accepts NumPy arrays deliberately. Framework adapters are responsible
for extracting per-parameter gradients and optimizer second moments.
"""
# Validation stays centralized so cross-parameter constraints remain visible.
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np


Shape = Tuple[int, ...]


class ParameterLayoutLike(Protocol):
    """Minimal layout interface needed by :meth:`TrackStarProjectionPlan.from_layout`."""

    @property
    def parameter_shapes(self) -> Sequence[Shape]:
        """Shapes in gradient-list order."""


class OptimizerSecondMomentsLike(Protocol):
    """Minimal optimizer-state interface accepted by a projection plan."""

    @property
    def parameter_second_moments(self) -> Sequence[np.ndarray]:
        """Elementwise second moments in gradient-list order."""


SecondMomentSource = Union[Sequence[np.ndarray], OptimizerSecondMomentsLike]


def _shape_size(shape: Shape) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(int(seed)), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _gaussian(shape: Shape, seed: int, scale: float) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(shape) * scale


@dataclass(frozen=True)
class ProjectionTerm:
    """One two-sided projection contribution within a projected block.

    A term owns either one parameter or a weight and its bias. The first
    parameter is viewed as ``(output, remaining dimensions)``. For rank two
    and above, ``output_axis`` explicitly identifies the output dimension;
    this makes TensorFlow- and PyTorch-shaped kernels unambiguous. Vectors and
    scalars have canonical shapes ``(n, 1)`` and ``(1, 1)`` respectively.

    When a bias is present it is appended as the final matrix column, yielding
    ``[weight | bias]`` before either projection matrix is applied.
    """

    name: str
    parameter_indices: Tuple[int, ...]
    parameter_shapes: Tuple[Shape, ...]
    projected_shape: Shape
    output_axis: int = 0
    left_matrix: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    right_matrix: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.parameter_indices)
        shapes = tuple(tuple(int(dim) for dim in shape) for shape in self.parameter_shapes)
        projected_shape = tuple(int(dim) for dim in self.projected_shape)
        object.__setattr__(self, "parameter_indices", indices)
        object.__setattr__(self, "parameter_shapes", shapes)
        object.__setattr__(self, "projected_shape", projected_shape)

        if not self.name:
            raise ValueError("ProjectionTerm name must not be empty.")
        if len(indices) not in (1, 2) or len(indices) != len(shapes):
            raise ValueError("A projection term must own one parameter, or one weight and one bias.")
        if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError("ProjectionTerm parameter indices must be distinct and non-negative.")
        if len(projected_shape) != 2 or any(dim <= 0 for dim in projected_shape):
            raise ValueError("projected_shape must contain two positive dimensions.")
        if any(dim <= 0 for shape in shapes for dim in shape):
            raise ValueError("Parameter shape dimensions must be positive.")

        rows, columns = self.canonical_shape
        if len(shapes) == 2 and shapes[1] != (rows,):
            raise ValueError(
                f"Bias shape must be ({rows},) after weight canonicalization; got {shapes[1]}."
            )
        expected_left = (projected_shape[0], rows)
        expected_right = (columns, projected_shape[1])
        for matrix, expected, side in (
            (self.left_matrix, expected_left, "left"),
            (self.right_matrix, expected_right, "right"),
        ):
            if matrix is not None:
                array = np.asarray(matrix)
                if array.shape != expected:
                    raise ValueError(f"{side}_matrix must have shape {expected}; got {array.shape}.")
                object.__setattr__(self, f"{side}_matrix", array.copy())

    @property
    def canonical_shape(self) -> Shape:
        """Shape of the packed matrix before projection."""
        shape = self.parameter_shapes[0]
        if len(shape) == 0:
            rows, columns = 1, 1
        elif len(shape) == 1:
            rows, columns = shape[0], 1
        else:
            axis = self.output_axis % len(shape)
            rows = shape[axis]
            columns = _shape_size(shape) // rows
        if len(self.parameter_shapes) == 2:
            columns += 1
        return rows, columns

    @property
    def output_size(self) -> int:
        """Flattened output width of this term."""
        return self.projected_shape[0] * self.projected_shape[1]

    def matrices(self, seed: int, block_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return injected or deterministic Gaussian left/right matrices."""
        rows, columns = self.canonical_shape
        out_rows, out_columns = self.projected_shape
        left = self.left_matrix
        right = self.right_matrix
        if left is None:
            left = _gaussian(
                (out_rows, rows),
                _stable_seed(seed, block_name, self.name, "left"),
                1.0 / np.sqrt(out_rows),
            )
        if right is None:
            right = _gaussian(
                (columns, out_columns),
                _stable_seed(seed, block_name, self.name, "right"),
                1.0 / np.sqrt(out_columns),
            )
        return left, right

    def canonical_view(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Return a batched canonical matrix, including explicit bias packing."""
        if len(arrays) != len(self.parameter_shapes):
            raise ValueError(f"Term {self.name!r} received the wrong number of arrays.")
        checked = []
        batch_size = None
        for array, shape in zip(arrays, self.parameter_shapes):
            value = np.asarray(array)
            expected_ndim = len(shape) + 1
            if value.ndim != expected_ndim or tuple(value.shape[1:]) != shape:
                raise ValueError(
                    f"Parameter batch must have shape (batch, {shape}); got {value.shape}."
                )
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("All parameter gradients must have the same batch size.")
            checked.append(value)

        weight = checked[0]
        if len(self.parameter_shapes[0]) == 0:
            matrix = weight.reshape((-1, 1, 1))
        elif len(self.parameter_shapes[0]) == 1:
            matrix = weight.reshape((-1, self.parameter_shapes[0][0], 1))
        else:
            axis = self.output_axis % len(self.parameter_shapes[0])
            matrix = np.moveaxis(weight, axis + 1, 1)
            matrix = matrix.reshape((matrix.shape[0], matrix.shape[1], -1))
        if len(checked) == 2:
            matrix = np.concatenate((matrix, checked[1][..., None]), axis=2)
        return matrix


@dataclass(frozen=True)
class ProjectionBlock:
    """Terms whose projected contributions are summed into one block."""

    name: str
    terms: Tuple[ProjectionTerm, ...]

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        object.__setattr__(self, "terms", terms)
        if not self.name or not terms:
            raise ValueError("A ProjectionBlock needs a name and at least one term.")
        shape = terms[0].projected_shape
        if any(term.projected_shape != shape for term in terms):
            raise ValueError("All terms in a block must have the same projected_shape.")

    @property
    def output_size(self) -> int:
        """Flattened width of this block."""
        return self.terms[0].output_size


@dataclass(frozen=True)
class TrackStarProjectionPlan:
    """Complete structured projection plan over an ordered parameter list."""

    parameter_shapes: Tuple[Shape, ...]
    blocks: Tuple[ProjectionBlock, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        shapes = tuple(tuple(int(dim) for dim in shape) for shape in self.parameter_shapes)
        blocks = tuple(self.blocks)
        object.__setattr__(self, "parameter_shapes", shapes)
        object.__setattr__(self, "blocks", blocks)
        if not shapes or not blocks:
            raise ValueError("A projection plan needs parameters and blocks.")
        if len({block.name for block in blocks}) != len(blocks):
            raise ValueError("Projection block names must be unique.")

        covered: list[int] = []
        for block in blocks:
            for term in block.terms:
                covered.extend(term.parameter_indices)
                for index, shape in zip(term.parameter_indices, term.parameter_shapes):
                    if index >= len(shapes):
                        raise ValueError(f"Parameter index {index} is outside the layout.")
                    if shape != shapes[index]:
                        raise ValueError(
                            f"Shape mismatch for parameter {index}: layout {shapes[index]}, term {shape}."
                        )
        expected = list(range(len(shapes)))
        if sorted(covered) != expected:
            missing = sorted(set(expected) - set(covered))
            repeated = sorted(index for index in set(covered) if covered.count(index) > 1)
            raise ValueError(
                "Projection terms must cover every parameter index exactly once; "
                f"missing={missing}, repeated={repeated}."
            )

    @classmethod
    def from_layout(
        cls, layout: ParameterLayoutLike, blocks: Sequence[ProjectionBlock], seed: int = 0
    ) -> "TrackStarProjectionPlan":
        """Build against the minimal ``parameter_shapes`` layout protocol."""
        return cls(tuple(tuple(shape) for shape in layout.parameter_shapes), tuple(blocks), seed)

    @property
    def output_size(self) -> int:
        """Width of the concatenated projected representation."""
        return sum(block.output_size for block in self.blocks)

    @property
    def block_slices(self) -> Mapping[str, slice]:
        """Ordered slices locating blocks in the concatenated representation."""
        start = 0
        slices = {}
        for block in self.blocks:
            slices[block.name] = slice(start, start + block.output_size)
            start += block.output_size
        return slices

    def apply(
        self,
        gradients: Sequence[np.ndarray],
        second_moments: Optional[SecondMomentSource] = None,
        *,
        optimizer_epsilon: float = 0.0,
    ) -> np.ndarray:
        """Correct, project, sum terms by block, and concatenate the blocks.

        Correction is exactly ``gradient / sqrt(second_moment + epsilon)`` and
        is intentionally performed before canonicalization and projection.
        """
        if len(gradients) != len(self.parameter_shapes):
            raise ValueError("Gradient list does not match the parameter layout.")
        if not np.isfinite(optimizer_epsilon) or optimizer_epsilon < 0:
            raise ValueError("optimizer_epsilon must be finite and non-negative.")
        corrected = []
        batch_size = None
        for index, gradient in enumerate(gradients):
            gradient_array = np.asarray(gradient)
            expected_shape = self.parameter_shapes[index]
            if gradient_array.ndim != len(expected_shape) + 1 or gradient_array.shape[1:] != expected_shape:
                raise ValueError(
                    f"Gradient {index} must have shape (batch, {expected_shape}); "
                    f"got {gradient_array.shape}."
                )
            if not np.issubdtype(gradient_array.dtype, np.number) or np.iscomplexobj(gradient_array):
                raise TypeError("TrackStar gradients must have a real numeric dtype.")
            if not np.all(np.isfinite(gradient_array)):
                raise ValueError("TrackStar gradients must contain only finite values.")
            if batch_size is None:
                batch_size = gradient_array.shape[0]
            elif gradient_array.shape[0] != batch_size:
                raise ValueError("All parameter gradients must have the same batch size.")
            corrected.append(gradient_array)
        if second_moments is not None:
            if hasattr(second_moments, "parameter_second_moments"):
                moments = cast(OptimizerSecondMomentsLike, second_moments).parameter_second_moments
            else:
                moments = cast(Sequence[np.ndarray], second_moments)
            if len(moments) != len(corrected):
                raise ValueError("Second-moment list does not match the parameter layout.")
            corrected = []
            for index, (gradient, moment) in enumerate(zip(gradients, moments)):
                gradient_array = np.asarray(gradient)
                moment_array = np.asarray(moment)
                if moment_array.shape != self.parameter_shapes[index]:
                    raise ValueError(
                        f"Second moment {index} must have shape {self.parameter_shapes[index]}; "
                        f"got {moment_array.shape}."
                    )
                if not np.issubdtype(moment_array.dtype, np.number) or np.iscomplexobj(moment_array):
                    raise TypeError("Optimizer second moments must have a real numeric dtype.")
                if not np.all(np.isfinite(moment_array)):
                    raise ValueError("Optimizer second moments must contain only finite values.")
                if np.any(moment_array < 0):
                    raise ValueError("Optimizer second moments must be non-negative.")
                denominator_squared = moment_array + optimizer_epsilon
                if np.any(denominator_squared == 0):
                    raise ValueError(
                        "Second moments plus optimizer_epsilon must be strictly positive."
                    )
                corrected.append(gradient_array / np.sqrt(denominator_squared))

        outputs: list[np.ndarray] = []
        for block in self.blocks:
            block_output: Optional[np.ndarray] = None
            for term in block.terms:
                values = [corrected[index] for index in term.parameter_indices]
                matrix = term.canonical_view(values)
                left, right = term.matrices(self.seed, block.name)
                contribution = np.matmul(np.matmul(left, matrix), right)
                contribution = contribution.reshape((contribution.shape[0], -1))
                block_output = contribution if block_output is None else block_output + contribution
            assert block_output is not None  # ProjectionBlock rejects empty term lists.
            outputs.append(block_output)
        return np.concatenate(outputs, axis=1)
