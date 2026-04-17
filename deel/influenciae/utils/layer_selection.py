# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Helpers for resolving layer selections against backend naming conventions.
"""
from dataclasses import dataclass
from warnings import warn
from typing import Callable, List, Sequence, TypeAlias, Union

from ..common.backend import BaseBackend
from ..types import Layer, Model

LayerSelectionTypeAlias: TypeAlias = Union[
    str,
    Sequence[int],
    Callable[[int, str, Layer], bool],
]


@dataclass(frozen=True)
class ResolvedLayerSelection:
    """Concrete layer selection resolved against a model."""

    layer_indices: List[int]
    layer_names: List[str]
    layers: List[Layer]


def resolve_layer_selection(
    model: Model,
    backend: BaseBackend,
    selector: LayerSelectionTypeAlias,
    *,
    layer_collection: str = "recursive",
    supported_only: bool = False,
) -> ResolvedLayerSelection:
    """
    Resolve *selector* against the backend's named layers.

    ``layer_indices`` always refer to the indices used internally by
    :class:`~deel.influenciae.common.kfac_factors.LayerParameterMap`, i.e. the
    enumeration of all named layers prior to filtering unsupported types.
    """
    if layer_collection not in ("top_level", "recursive"):
        raise ValueError("layer_collection must be either 'top_level' or 'recursive'.")

    named_layers = backend.get_named_layers(model, recursive=layer_collection == "recursive")
    unique_named_layers = []
    seen_layer_ids = set()
    for layer_name, layer in named_layers:
        layer_id = id(layer)
        if layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(layer_id)
        unique_named_layers.append((str(layer_name), layer))

    if isinstance(selector, str):
        def _matches(layer_idx: int, layer_name: str, layer: Layer) -> bool:
            _ = layer_idx, layer
            if layer_collection == "recursive":
                return layer_name == selector or layer_name.startswith(f"{selector}.")
            return layer_name == selector
    elif callable(selector):
        _matches = selector
    else:
        target_indices = set(int(index) for index in selector)

        def _matches(layer_idx: int, layer_name: str, layer: Layer) -> bool:
            _ = layer_name, layer
            return layer_idx in target_indices

    resolved_indices: List[int] = []
    resolved_names: List[str] = []
    resolved_layers: List[Layer] = []
    for layer_idx, (layer_name, layer) in enumerate(unique_named_layers):
        if not _matches(layer_idx, layer_name, layer):
            continue
        if supported_only and not (backend.is_linear_layer(layer) or backend.is_conv2d_layer(layer)):
            continue
        resolved_indices.append(layer_idx)
        resolved_names.append(layer_name)
        resolved_layers.append(layer)

    if not resolved_indices:
        warn(
            f"No layers matched selector {selector!r} with layer_collection={layer_collection!r}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return ResolvedLayerSelection(
        layer_indices=resolved_indices,
        layer_names=resolved_names,
        layers=resolved_layers,
    )
