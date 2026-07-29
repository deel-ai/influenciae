# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Canonical metadata for flattened model parameters and Jacobians."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..types import Tensor, WeightVariable
from .backend import BaseBackend


Shape = Tuple[int, ...]


@dataclass(frozen=True)
class ParameterLayoutEntry:
    """Metadata for one parameter in the watched-parameter order."""

    index: int
    name: str
    shape: Shape
    size: int
    flat_start: int
    flat_end: int

    @property
    def flat_slice(self) -> slice:
        """Return this parameter's columns in a flat representation."""
        return slice(self.flat_start, self.flat_end)


@dataclass(frozen=True)
class ParameterLayout:
    """Describe the exact native order used to flatten watched parameters."""

    entries: Tuple[ParameterLayoutEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        expected_start = 0
        names = set()
        for index, entry in enumerate(entries):
            if entry.index != index:
                raise ValueError("Parameter layout indices must be contiguous and ordered.")
            if not entry.name or entry.name in names:
                raise ValueError("Parameter layout names must be non-empty and unique.")
            if any(dim <= 0 for dim in entry.shape):
                raise ValueError("Parameter shape dimensions must be positive.")
            expected_size = int(np.prod(entry.shape, dtype=np.int64)) if entry.shape else 1
            if entry.size != expected_size:
                raise ValueError(f"Parameter {entry.name!r} has an inconsistent size.")
            if entry.flat_start != expected_start or entry.flat_end != expected_start + entry.size:
                raise ValueError("Parameter layout slices must be contiguous and ordered.")
            expected_start = entry.flat_end
            names.add(entry.name)

    @classmethod
    def from_shapes(
        cls,
        shapes: Sequence[Shape],
        names: Optional[Sequence[Optional[str]]] = None,
    ) -> "ParameterLayout":
        """Build a layout from ordered native parameter shapes and optional names."""
        normalized_shapes = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
        if names is not None and len(names) != len(normalized_shapes):
            raise ValueError("Parameter names must match the number of parameter shapes.")
        supplied_names = names if names is not None else (None,) * len(normalized_shapes)

        entries = []
        used_names = set()
        flat_start = 0
        for index, (shape, supplied_name) in enumerate(zip(normalized_shapes, supplied_names)):
            base_name = str(supplied_name) if supplied_name else f"parameter_{index}"
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{base_name}#{suffix}"
                suffix += 1
            size = int(np.prod(shape, dtype=np.int64)) if shape else 1
            entries.append(
                ParameterLayoutEntry(index, name, shape, size, flat_start, flat_start + size)
            )
            flat_start += size
            used_names.add(name)
        return cls(tuple(entries))

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """Names in watched-parameter order."""
        return tuple(entry.name for entry in self.entries)

    @property
    def parameter_shapes(self) -> Tuple[Shape, ...]:
        """Native shapes in watched-parameter order."""
        return tuple(entry.shape for entry in self.entries)

    @property
    def parameter_sizes(self) -> Tuple[int, ...]:
        """Flattened sizes in watched-parameter order."""
        return tuple(entry.size for entry in self.entries)

    @property
    def parameter_slices(self) -> Tuple[slice, ...]:
        """Slices into the final axis of a flat representation."""
        return tuple(entry.flat_slice for entry in self.entries)

    @property
    def num_parameters(self) -> int:
        """Number of watched parameter tensors."""
        return len(self.entries)

    @property
    def total_size(self) -> int:
        """Width of the flattened parameter representation."""
        return self.entries[-1].flat_end if self.entries else 0

    def split_flat(self, tensor: Tensor, backend: BaseBackend) -> Tuple[Tensor, ...]:
        """Split the final flat axis and restore every native parameter shape."""
        tensor_shape = backend.tensor_shape(tensor)
        if not tensor_shape or tensor_shape[-1] != self.total_size:
            actual = tensor_shape[-1] if tensor_shape else None
            raise ValueError(
                f"Flat tensor must have final dimension {self.total_size}; got {actual}."
            )
        leading_shape = tensor_shape[:-1]
        return tuple(
            backend.reshape(tensor[..., entry.flat_slice], leading_shape + entry.shape)
            for entry in self.entries
        )

    def flatten_components(
        self, components: Sequence[Tensor], backend: BaseBackend
    ) -> Tensor:
        """Flatten native parameter components and concatenate their final axes."""
        if len(components) != len(self.entries):
            raise ValueError("Component list does not match the parameter layout.")
        flattened = []
        leading_shape = None
        for component, entry in zip(components, self.entries):
            component_shape = backend.tensor_shape(component)
            parameter_rank = len(entry.shape)
            trailing_shape = component_shape[-parameter_rank:] if parameter_rank else ()
            current_leading = component_shape[:-parameter_rank] if parameter_rank else component_shape
            if trailing_shape != entry.shape:
                raise ValueError(
                    f"Component {entry.name!r} must end with shape {entry.shape}; "
                    f"got {component_shape}."
                )
            if leading_shape is None:
                leading_shape = current_leading
            elif current_leading != leading_shape:
                raise ValueError("All parameter components must have matching leading dimensions.")
            flattened.append(backend.reshape(component, current_leading + (entry.size,)))
        if not flattened:
            raise ValueError("Cannot flatten an empty parameter layout.")
        return backend.concat(flattened, axis=-1)


def build_parameter_layout(
    weights: Sequence[WeightVariable],
    backend: BaseBackend,
    names: Optional[Sequence[Optional[str]]] = None,
) -> ParameterLayout:
    """Build layout metadata for an exact ordered list of watched weights."""
    shapes = tuple(backend.tensor_shape(weight) for weight in weights)
    return ParameterLayout.from_shapes(shapes, names)
