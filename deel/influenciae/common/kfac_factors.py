# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
K-FAC and EK-FAC Kronecker factor computation for Inverse Hessian Vector Products.

This module implements the Kronecker-Factored Approximate Curvature (K-FAC) and its
eigenvalue-corrected variant (EK-FAC) as described in:

- Martens & Grosse (2015), "Optimizing Neural Networks with Kronecker-Factored
  Approximate Curvature"
- Grosse et al. (2023), "Studying Large Language Model Generalization with Influence
  Functions" (arXiv:2308.03296)

K-FAC approximates the Fisher information matrix block-diagonally per layer, factoring
each block as a Kronecker product of input activation covariance (A) and output gradient
covariance (G).  EK-FAC refines this by eigendecomposing the factors and estimating
corrected diagonal eigenvalues.
"""
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..types import DatasetLike, LossFunction, Tensor
from .backend import BaseBackend
from .model_wrappers import BaseInfluenceModel


def _chunk_layer_infos(layer_infos: List["LayerInfo"], chunk_size: int) -> List[List["LayerInfo"]]:
    """Split layer infos into chunks of at most *chunk_size* entries."""
    return [layer_infos[i:i + chunk_size] for i in range(0, len(layer_infos), chunk_size)]


_FACTOR_CHECKPOINT_SCHEMA_VERSION = 1
_FACTOR_CHECKPOINT_METADATA_FILE = "metadata.json"
HEURISTIC_DAMPING_SCALE = 0.1


def _shape_to_list(shape: Tuple[int, ...]) -> List[Optional[int]]:
    """Convert tensor shape tuples to JSON-serializable lists."""
    serialized_shape: List[Optional[int]] = []
    for dim in shape:
        if dim is None:
            serialized_shape.append(None)
        else:
            serialized_shape.append(int(dim))
    return serialized_shape


def _serialize_layer_infos(layer_infos: List["LayerInfo"]) -> List[Dict[str, Any]]:
    """Serialize layer signatures for checkpoint metadata."""
    serialized = []
    for info in layer_infos:
        serialized.append(
            {
                "layer_name": str(info.layer_name),
                "layer_idx": int(info.layer_idx),
                "flat_start": int(info.flat_start),
                "flat_end": int(info.flat_end),
                "has_bias": bool(info.has_bias),
                "weight_shape": _shape_to_list(info.weight_shape),
            }
        )
    return serialized


def _remove_path(path: str) -> None:
    """Remove a file or directory if it exists."""
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def _atomic_write_directory(target_dir: str, writer: Callable[[str], None]) -> None:
    """Atomically write a checkpoint directory by replacing it at the end."""
    parent_dir = os.path.dirname(target_dir) or "."
    os.makedirs(parent_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=".tmp_kfac_", dir=parent_dir)
    try:
        writer(tmp_dir)
        _remove_path(target_dir)
        os.replace(tmp_dir, target_dir)
    except Exception:
        _remove_path(tmp_dir)
        raise


def _read_checkpoint_metadata(path: str) -> Dict[str, Any]:
    """Load and validate the checkpoint metadata JSON file."""
    metadata_path = os.path.join(path, _FACTOR_CHECKPOINT_METADATA_FILE)
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"No factor checkpoint metadata found at '{metadata_path}'."
        )
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("Factor checkpoint metadata must be a JSON object.")
    return metadata


def _write_checkpoint_metadata(path: str, metadata: Dict[str, Any]) -> None:
    """Write metadata.json for factor checkpoints."""
    metadata_path = os.path.join(path, _FACTOR_CHECKPOINT_METADATA_FILE)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def _make_tensor_metadata(file_name: str, array: np.ndarray) -> Dict[str, Any]:
    """Create metadata entry for a saved tensor."""
    return {
        "file": file_name,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
    }


def _validate_checkpoint_compatibility(
    metadata: Dict[str, Any],
    expected_method: str,
    expected_fisher_type: str,
    model: BaseInfluenceModel,
    layer_map: "LayerParameterMap",
    expected_n_ekfac_samples: Optional[int] = None,
) -> None:
    """Validate that checkpoint metadata matches the current model/configuration."""
    schema_version = metadata.get("schema_version")
    if schema_version != _FACTOR_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported factor checkpoint schema version "
            f"{schema_version!r}; expected {_FACTOR_CHECKPOINT_SCHEMA_VERSION}."
        )

    method = metadata.get("method")
    if method != expected_method:
        raise ValueError(
            f"Factor checkpoint method mismatch: expected '{expected_method}', got '{method}'."
        )

    fisher_type = metadata.get("fisher_type")
    if fisher_type != expected_fisher_type:
        raise ValueError(
            "Factor checkpoint fisher_type mismatch: "
            f"expected '{expected_fisher_type}', got '{fisher_type}'."
        )

    checkpoint_nb_params = metadata.get("nb_params")
    if checkpoint_nb_params != int(model.nb_params):
        raise ValueError(
            "Factor checkpoint nb_params mismatch: "
            f"expected {int(model.nb_params)}, got {checkpoint_nb_params}."
        )

    if expected_method == "ekfac":
        checkpoint_n_ekfac_samples = metadata.get("n_ekfac_samples")
        if checkpoint_n_ekfac_samples != expected_n_ekfac_samples:
            raise ValueError(
                "EK-FAC checkpoint n_ekfac_samples mismatch: "
                f"expected {expected_n_ekfac_samples}, got {checkpoint_n_ekfac_samples}."
            )

    checkpoint_layers = metadata.get("layers")
    if not isinstance(checkpoint_layers, list):
        raise ValueError("Factor checkpoint metadata must contain a 'layers' list.")

    current_layers = _serialize_layer_infos(layer_map.layers_info)
    if len(checkpoint_layers) != len(current_layers):
        raise ValueError(
            "Factor checkpoint layer count mismatch: "
            f"expected {len(current_layers)}, got {len(checkpoint_layers)}."
        )

    layer_collection = metadata.get("layer_collection")
    if layer_collection != layer_map.layer_collection:
        raise ValueError(
            "Factor checkpoint layer_collection mismatch: "
            f"expected '{layer_map.layer_collection}', got '{layer_collection}'."
        )

    compared_fields = ("layer_name", "layer_idx", "flat_start", "flat_end", "has_bias", "weight_shape")
    for layer_position, (saved_layer, current_layer) in enumerate(zip(checkpoint_layers, current_layers)):
        if not isinstance(saved_layer, dict):
            raise ValueError(
                f"Invalid layer metadata at position {layer_position}: expected object."
            )
        for field in compared_fields:
            if saved_layer.get(field) != current_layer[field]:
                raise ValueError(
                    "Factor checkpoint layer signature mismatch at position "
                    f"{layer_position} for field '{field}': "
                    f"expected {current_layer[field]!r}, got {saved_layer.get(field)!r}."
                )


def _load_tensor_from_checkpoint(
    backend: BaseBackend,
    checkpoint_dir: str,
    layer_idx: int,
    tensor_name: str,
    tensor_metadata: Dict[str, Any],
    reference_tensor: Optional[Tensor],
) -> Tensor:
    """Load a tensor from checkpoint metadata and move it to the target device."""
    file_name = tensor_metadata.get("file")
    if not isinstance(file_name, str):
        raise ValueError(
            f"Invalid checkpoint metadata for layer {layer_idx} tensor '{tensor_name}': missing file name."
        )

    tensor_path = os.path.join(checkpoint_dir, file_name)
    if not os.path.isfile(tensor_path):
        raise FileNotFoundError(
            f"Missing checkpoint tensor file for layer {layer_idx} tensor '{tensor_name}': '{tensor_path}'."
        )

    tensor_array = np.load(tensor_path, allow_pickle=False)

    expected_shape = tensor_metadata.get("shape")
    if expected_shape is not None:
        expected_shape_tuple = tuple(int(dim) for dim in expected_shape)
        if tuple(tensor_array.shape) != expected_shape_tuple:
            raise ValueError(
                f"Checkpoint tensor shape mismatch for layer {layer_idx} tensor '{tensor_name}': "
                f"expected {expected_shape_tuple}, got {tuple(tensor_array.shape)}."
            )

    tensor = backend.convert_to_tensor(tensor_array)
    return backend.to_device(tensor, reference=reference_tensor)


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerInfo:
    """Metadata for a single K-FAC-supported layer.

    Attributes
    ----------
    layer
        The framework-specific layer object.
    layer_name
        Layer/module name as reported by backend traversal.
    layer_idx
        Index of the layer in the selected collection order
        (top-level or recursive).
    weight_shape
        Shape of the weight tensor as reported by the framework.  In PyTorch
        this is ``(n_out, n_in)`` for Linear and ``(c_out, c_in, kH, kW)``
        for Conv2d.  In TensorFlow it is ``(n_in, n_out)`` for Dense and
        ``(kH, kW, c_in, c_out)`` for Conv2D.  Do **not** assume a fixed
        dimension ordering — use the Kronecker factor matrix shapes to derive
        ``n_out`` and ``n_in_eff`` instead.
    has_bias
        Whether the layer has a bias parameter.
    flat_start
        Start index of this layer's parameters in the flat gradient vector.
    flat_end
        End index (exclusive) of this layer's parameters in the flat gradient vector.
    """
    layer: Any
    layer_name: str
    layer_idx: int
    weight_shape: Tuple[int, ...]
    has_bias: bool
    flat_start: int
    flat_end: int


class LayerParameterMap:
    """Maps between flat gradient vectors and per-layer matrix representations.

    This mirrors the ``ForwardOverBackwardHVP._weight_slices`` pattern but
    adds layer-level bookkeeping needed by K-FAC: the map only includes
    *supported* (Linear / dense Conv2d) layers and records their position
    within the complete flat gradient vector.

    Parameters
    ----------
    model
        The influence model wrapper.
    backend
        The backend abstraction.
    target_layers
        If provided, restrict K-FAC to these layer indices only.
    layer_collection
        Layer traversal mode. ``"top_level"`` keeps previous behavior;
        ``"recursive"`` traverses nested submodules/layers.
    """

    def __init__(
        self,
        model: BaseInfluenceModel,
        backend: BaseBackend,
        target_layers: Optional[List[int]] = None,
        layer_collection: str = "top_level",
    ):
        self.backend = backend
        self.layers_info: List[LayerInfo] = []
        if layer_collection not in ("top_level", "recursive"):
            raise ValueError("layer_collection must be either 'top_level' or 'recursive'.")
        self.layer_collection = layer_collection

        all_named_layers = backend.get_named_layers(
            model.model,
            recursive=layer_collection == "recursive",
        )
        unique_named_layers: List[Tuple[str, Any]] = []
        seen_layer_ids = set()
        for layer_name, layer in all_named_layers:
            layer_id = id(layer)
            if layer_id in seen_layer_ids:
                continue
            seen_layer_ids.add(layer_id)
            unique_named_layers.append((layer_name, layer))
        all_named_layers = unique_named_layers

        weights = model.weights

        def _iter_weight_keys(weight: Any) -> List[Tuple[str, Any]]:
            """Return matching keys for a weight object across backend wrappers.

            PyTorch parameters are stable Python objects, so ``id(param)`` is
            sufficient. TensorFlow/Keras can expose wrapped variable objects
            (e.g. KerasVariable vs. ResourceVariable), so we include both
            object-id and name-based keys from common wrapper attributes.
            """
            keys: List[Tuple[str, Any]] = []
            to_visit = [weight]

            normalize_weights = getattr(backend, "normalize_weights_to_watch", None)
            if callable(normalize_weights):
                try:
                    normalized = normalize_weights([weight])
                    if isinstance(normalized, (list, tuple)):
                        for normalized_weight in normalized:
                            to_visit.append(normalized_weight)
                    elif normalized is not None:
                        to_visit.append(normalized)
                except Exception:  # pylint: disable=broad-except
                    pass

            seen_ids = set()
            while to_visit:
                current = to_visit.pop()
                if current is None:
                    continue

                current_id = id(current)
                if current_id in seen_ids:
                    continue
                seen_ids.add(current_id)

                keys.append(("id", current_id))

                current_name = getattr(current, "name", None)
                if current_name is not None:
                    keys.append(("name", str(current_name)))

                for attr_name in ("_variable", "variable", "_value", "value", "handle"):
                    attr_value = getattr(current, attr_name, None)
                    if callable(attr_value):
                        try:
                            attr_value = attr_value()
                        except TypeError:
                            continue
                    if attr_value is not None:
                        to_visit.append(attr_value)

            return keys

        # Build a mapping from backend-stable parameter keys to their position
        # in the flat gradient vector (same logic as ForwardOverBackwardHVP).
        weight_key_to_flat: Dict[Tuple[str, Any], Tuple[int, int, Tuple[int, ...]]] = {}
        offset = 0
        for w in weights:
            shape = backend.tensor_shape(w)
            size = 1
            for d in shape:
                size *= int(d)
            flat_entry = (offset, offset + size, shape)
            for key in _iter_weight_keys(w):
                if key not in weight_key_to_flat:
                    weight_key_to_flat[key] = flat_entry
            offset += size

        for layer_idx, (layer_name, layer) in enumerate(all_named_layers):
            if target_layers is not None and layer_idx not in target_layers:
                continue

            normalized_layer_name = str(layer_name) if layer_name else f"<layer_{layer_idx}>"

            if not backend.is_kfac_supported_layer(layer):
                if target_layers is not None:
                    warnings.warn(
                        f"Layer {layer_idx} ('{normalized_layer_name}', {type(layer).__name__}) "
                        f"is not a K-FAC-supported Linear/Dense or dense Conv2d layer and will be skipped by K-FAC.",
                        stacklevel=2,
                    )
                continue

            weight, bias = backend.get_layer_weight_and_bias(layer)
            weight_shape = backend.tensor_shape(weight)

            # Find the weight in the flat vector
            flat_entry = None
            for key in _iter_weight_keys(weight):
                if key in weight_key_to_flat:
                    flat_entry = weight_key_to_flat[key]
                    break

            if flat_entry is None:
                # Weight not tracked (frozen, outside watched range, or wrapper mismatch)
                continue

            flat_start = flat_entry[0]
            flat_end = flat_entry[1]

            has_bias = bias is not None
            if has_bias:
                for key in _iter_weight_keys(bias):
                    if key in weight_key_to_flat:
                        # Bias immediately follows weight in the flat vector
                        flat_end = weight_key_to_flat[key][1]
                        break

            self.layers_info.append(LayerInfo(
                layer=layer,
                layer_name=normalized_layer_name,
                layer_idx=layer_idx,
                weight_shape=weight_shape,
                has_bias=has_bias,
                flat_start=flat_start,
                flat_end=flat_end,
            ))

        if not self.layers_info:
            warnings.warn(
                "No supported layers (Linear/Dense, dense Conv2d) found for K-FAC. "
                "The IHVP will fall back to a zero approximation.",
                stacklevel=2,
            )

    @property
    def n_supported_layers(self) -> int:
        return len(self.layers_info)


# ---------------------------------------------------------------------------
# K-FAC factor computation
# ---------------------------------------------------------------------------

class KroneckerFactors:
    """Compute and store Kronecker factors A_l and G_l for each supported layer.

    For a fully-connected layer with weight W of shape ``(n_out, n_in)``:

    * ``A_l = E[a_l a_l^T]``  — input activation covariance ``(n_in, n_in)``
      (or ``(n_in+1, n_in+1)`` when bias is present, with the homogeneous "1"
      appended to activations).
    * ``G_l = E[g_l g_l^T]``  — output gradient covariance ``(n_out, n_out)``.

    For Conv2d the activations are unfolded (im2col) before computing A.

    Parameters
    ----------
    model
        The influence model wrapper.
    train_dataset
        Batched training dataset for estimating the factors.
    backend
        The backend abstraction.
    layer_map
        Pre-computed ``LayerParameterMap``.
    fisher_type
        Fisher variant used for curvature estimation: ``"empirical"`` (default)
        or ``"true"``.
    module_partition_size
        Optional number of supported layers to process per pass. ``None`` means
        process all supported layers together.
    offload_activations_to_cpu
        If ``True``, hook-captured activations and gradients are moved to CPU
        before accumulation and brought back to the compute device when needed.
    data_partition_size
        Optional number of batches per data partition.
    accumulator_offload_mode
        Accumulator checkpoint mode used at data-partition boundaries:
        ``"none"`` (default), ``"memory"``, or ``"disk"``.
    accumulator_offload_dir
        Optional directory where temporary disk-offload partition files are
        stored when ``accumulator_offload_mode="disk"``.
    keep_accumulator_offload_artifacts
        If ``True``, keep temporary disk-offload partition files after factor
        computation. Otherwise they are removed.
    """

    def __init__(
        self,
        model: BaseInfluenceModel,
        train_dataset: DatasetLike,
        backend: BaseBackend,
        layer_map: LayerParameterMap,
        fisher_type: str = "empirical",
        module_partition_size: Optional[int] = None,
        offload_activations_to_cpu: bool = False,
        data_partition_size: Optional[int] = None,
        accumulator_offload_mode: str = "none",
        accumulator_offload_dir: Optional[str] = None,
        keep_accumulator_offload_artifacts: bool = False,
    ):
        self.backend = backend
        self.model = model
        self.layer_map = layer_map
        if fisher_type not in ("empirical", "true"):
            raise ValueError("fisher_type must be either 'empirical' or 'true'.")
        if module_partition_size is not None and module_partition_size <= 0:
            raise ValueError("module_partition_size must be a strictly positive integer or None.")
        if data_partition_size is not None and data_partition_size <= 0:
            raise ValueError("data_partition_size must be a strictly positive integer or None.")
        if accumulator_offload_mode not in ("none", "memory", "disk"):
            raise ValueError(
                "accumulator_offload_mode must be one of 'none', 'memory', or 'disk'."
            )

        self.fisher_type = fisher_type
        self.module_partition_size = module_partition_size
        self.offload_activations_to_cpu = offload_activations_to_cpu
        self.data_partition_size = data_partition_size
        self.accumulator_offload_mode = accumulator_offload_mode
        self.accumulator_offload_dir = accumulator_offload_dir
        self.keep_accumulator_offload_artifacts = keep_accumulator_offload_artifacts
        self._true_fisher_warning_emitted = False
        self._factor_checkpoint: Optional[Dict[str, Any]] = None
        self.A: Dict[int, Tensor] = {}  # layer_idx -> A factor
        self.G: Dict[int, Tensor] = {}  # layer_idx -> G factor

        self._compute_factors(train_dataset)

    @staticmethod
    def checkpoint_exists(path: str) -> bool:
        """Return whether *path* points to a factor checkpoint directory."""
        metadata_path = os.path.join(path, _FACTOR_CHECKPOINT_METADATA_FILE)
        return os.path.isfile(metadata_path)

    def _build_checkpoint_metadata(self, method: str) -> Dict[str, Any]:
        """Build common checkpoint metadata payload."""
        return {
            "schema_version": _FACTOR_CHECKPOINT_SCHEMA_VERSION,
            "method": method,
            "fisher_type": self.fisher_type,
            "nb_params": int(self.model.nb_params),
            "layer_collection": self.layer_map.layer_collection,
            "module_partition_size": self.module_partition_size,
            "offload_activations_to_cpu": bool(self.offload_activations_to_cpu),
            "data_partition_size": self.data_partition_size,
            "true_fisher_fallback_used": bool(self._true_fisher_warning_emitted),
            "layers": _serialize_layer_infos(self.layer_map.layers_info),
        }

    def save_to_dir(self, path: str) -> None:
        """Serialize computed K-FAC factors to *path*.

        The checkpoint stores metadata and per-layer factor matrices.
        """

        def _writer(tmp_dir: str) -> None:
            metadata = self._build_checkpoint_metadata(method="kfac")
            for layer_entry in metadata["layers"]:
                layer_idx = int(layer_entry["layer_idx"])
                if layer_idx not in self.A or layer_idx not in self.G:
                    continue

                tensor_entries = {}
                for tensor_name, tensor in (("A", self.A[layer_idx]), ("G", self.G[layer_idx])):
                    tensor_array = np.asarray(self.backend.to_numpy(tensor))
                    file_name = f"{tensor_name}_layer_{layer_idx}.npy"
                    np.save(os.path.join(tmp_dir, file_name), tensor_array, allow_pickle=False)
                    tensor_entries[tensor_name] = _make_tensor_metadata(file_name, tensor_array)

                layer_entry["tensors"] = tensor_entries

            _write_checkpoint_metadata(tmp_dir, metadata)

        _atomic_write_directory(path, _writer)

    @classmethod
    def load_from_dir(
        cls,
        model: BaseInfluenceModel,
        backend: BaseBackend,
        layer_map: LayerParameterMap,
        path: str,
        fisher_type: str = "empirical",
        module_partition_size: Optional[int] = None,
        offload_activations_to_cpu: bool = False,
        data_partition_size: Optional[int] = None,
        accumulator_offload_mode: str = "none",
        accumulator_offload_dir: Optional[str] = None,
        keep_accumulator_offload_artifacts: bool = False,
    ) -> "KroneckerFactors":
        """Load precomputed K-FAC factors from *path* without recomputing."""
        if accumulator_offload_mode not in ("none", "memory", "disk"):
            raise ValueError(
                "accumulator_offload_mode must be one of 'none', 'memory', or 'disk'."
            )

        metadata = _read_checkpoint_metadata(path)
        _validate_checkpoint_compatibility(
            metadata=metadata,
            expected_method="kfac",
            expected_fisher_type=fisher_type,
            model=model,
            layer_map=layer_map,
        )

        instance = cls.__new__(cls)
        instance.backend = backend
        instance.model = model
        instance.layer_map = layer_map
        instance.fisher_type = fisher_type
        instance.module_partition_size = module_partition_size
        instance.offload_activations_to_cpu = offload_activations_to_cpu
        instance.data_partition_size = data_partition_size
        instance.accumulator_offload_mode = accumulator_offload_mode
        instance.accumulator_offload_dir = accumulator_offload_dir
        instance.keep_accumulator_offload_artifacts = keep_accumulator_offload_artifacts
        instance._true_fisher_warning_emitted = bool(metadata.get("true_fisher_fallback_used", False))
        instance._factor_checkpoint = None
        instance.A = {}
        instance.G = {}

        reference_tensor = instance._get_device_reference_tensor()
        for layer_entry in metadata["layers"]:
            tensors = layer_entry.get("tensors")
            if not isinstance(tensors, dict):
                continue

            layer_idx = int(layer_entry["layer_idx"])
            if "A" not in tensors or "G" not in tensors:
                raise ValueError(
                    f"Invalid checkpoint metadata for layer {layer_idx}: both 'A' and 'G' tensors are required."
                )

            instance.A[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "A",
                tensors["A"],
                reference_tensor,
            )
            instance.G[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "G",
                tensors["G"],
                reference_tensor,
            )

        return instance

    def _warn_true_fisher_fallback(self, loss_function: LossFunction) -> None:
        """Warn once when true Fisher sampling is not supported for a loss."""
        if self._true_fisher_warning_emitted:
            return

        warnings.warn(
            "fisher_type='true' is not supported for "
            f"loss {type(loss_function).__name__}; falling back to empirical Fisher.",
            stacklevel=2,
        )
        self._true_fisher_warning_emitted = True

    def _symmetrize_matrix(self, matrix: Tensor) -> Tensor:
        """Return the symmetric part of a square matrix."""
        return 0.5 * (matrix + self.backend.transpose(matrix))

    def _iter_layer_partitions(self) -> List[List[LayerInfo]]:
        """Return layer partitions according to ``module_partition_size``."""
        if self.module_partition_size is None:
            return [self.layer_map.layers_info]
        return _chunk_layer_infos(self.layer_map.layers_info, self.module_partition_size)

    def _get_device_reference_tensor(self) -> Optional[Tensor]:
        """Return a tensor living on the target compute device."""
        model_weights = getattr(self.model, "weights", None)
        if model_weights:
            return model_weights[0]
        return None

    def _make_accumulator_partition_dir(self, prefix: str) -> str:
        """Create a directory for temporary accumulator partition files."""
        if self.accumulator_offload_dir is None:
            return tempfile.mkdtemp(prefix=f"{prefix}_")

        os.makedirs(self.accumulator_offload_dir, exist_ok=True)
        return tempfile.mkdtemp(prefix=f"{prefix}_", dir=self.accumulator_offload_dir)

    def _write_accumulator_partition_to_disk(
        self,
        partition_dir: str,
        partition_idx: int,
        tensors: Dict[str, Dict[int, Tensor]],
        n_rows_per_layer: Dict[int, int],
    ) -> str:
        """Serialize one accumulator partition to a compressed ``.npz`` file."""
        layer_indices_set = set(n_rows_per_layer.keys())
        for tensor_dict in tensors.values():
            layer_indices_set.update(tensor_dict.keys())
        layer_indices = sorted(int(layer_idx) for layer_idx in layer_indices_set)

        payload: Dict[str, np.ndarray] = {
            "layer_indices": np.asarray(layer_indices, dtype=np.int64),
        }
        for layer_idx in layer_indices:
            payload[f"rows_{layer_idx}"] = np.asarray([int(n_rows_per_layer.get(layer_idx, 0))], dtype=np.int64)
            for tensor_name, tensor_dict in tensors.items():
                if layer_idx not in tensor_dict:
                    continue
                tensor_array = np.asarray(self.backend.to_numpy(tensor_dict[layer_idx]))
                payload[f"{tensor_name}_{layer_idx}"] = tensor_array

        partition_path = os.path.join(partition_dir, f"partition_{partition_idx:06d}.npz")
        np.savez(partition_path, **payload)
        return partition_path

    def _merge_accumulator_partitions_from_disk(
        self,
        partition_paths: List[str],
        tensor_names: Tuple[str, ...],
    ) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[int, int]]:
        """Load and merge accumulator partitions from disk."""
        merged_tensors: Dict[str, Dict[int, np.ndarray]] = {name: {} for name in tensor_names}
        merged_rows: Dict[int, int] = {}

        for partition_path in partition_paths:
            with np.load(partition_path, allow_pickle=False) as partition_data:
                if "layer_indices" not in partition_data:
                    raise ValueError(
                        f"Invalid accumulator partition file '{partition_path}': missing layer_indices."
                    )
                layer_indices = [int(v) for v in np.asarray(partition_data["layer_indices"]).tolist()]

                for layer_idx in layer_indices:
                    rows_key = f"rows_{layer_idx}"
                    if rows_key not in partition_data:
                        raise ValueError(
                            f"Invalid accumulator partition file '{partition_path}': missing '{rows_key}'."
                        )
                    n_rows = int(np.asarray(partition_data[rows_key]).reshape(-1)[0])
                    merged_rows[layer_idx] = merged_rows.get(layer_idx, 0) + n_rows

                    for tensor_name in tensor_names:
                        tensor_key = f"{tensor_name}_{layer_idx}"
                        if tensor_key not in partition_data:
                            continue

                        tensor_array = np.asarray(partition_data[tensor_key])
                        if layer_idx not in merged_tensors[tensor_name]:
                            merged_tensors[tensor_name][layer_idx] = tensor_array
                        else:
                            merged_tensors[tensor_name][layer_idx] = (
                                merged_tensors[tensor_name][layer_idx] + tensor_array
                            )

        return merged_tensors, merged_rows

    def _cleanup_accumulator_partition_dir(self, partition_dir: Optional[str]) -> None:
        """Cleanup temporary accumulator partition files if requested."""
        if partition_dir is None:
            return
        if self.keep_accumulator_offload_artifacts:
            return
        _remove_path(partition_dir)

    def _checkpoint_factor_accumulators(
        self,
        a_sums: Dict[int, Tensor],
        g_sums: Dict[int, Tensor],
        n_rows_per_layer: Dict[int, int],
    ) -> None:
        """Store in-memory factor checkpoint state at data-partition boundaries."""
        self._factor_checkpoint = {
            "mode": "memory",
            "a_sums": a_sums,
            "g_sums": g_sums,
            "n_rows_per_layer": n_rows_per_layer,
        }

    # ------------------------------------------------------------------
    # Factor computation
    # ------------------------------------------------------------------

    def _compute_factors(self, train_dataset: DatasetLike) -> None:
        """Estimate A and G from *train_dataset* using hooks."""
        for layer_partition in self._iter_layer_partitions():
            self._compute_factors_for_layer_partition(train_dataset, layer_partition)

    def _compute_factors_for_layer_partition(
        self,
        train_dataset: DatasetLike,
        layer_infos: List[LayerInfo],
    ) -> None:
        """Estimate A and G for a specific partition of layers."""
        backend = self.backend
        model = self.model
        device_reference = self._get_device_reference_tensor()

        def _to_int(value: Any) -> int:
            """Convert backend scalar values to Python int."""
            try:
                return int(value)
            except TypeError:
                return int(value.numpy())

        # Storage for running sums
        a_sums: Dict[int, Tensor] = {}
        g_sums: Dict[int, Tensor] = {}
        n_rows_per_layer: Dict[int, int] = {}

        # --- Register hooks -----------------------------------------------
        handles = []
        # Store one entry per hook invocation. Shared layers can be called
        # multiple times during a single forward pass, so overwriting by layer
        # index would pair unrelated activations and gradients.
        activations: Dict[int, List[Tensor]] = {}
        grad_outputs: Dict[int, List[Tensor]] = {}

        for info in layer_infos:
            idx = info.layer_idx

            def _fwd_hook(layer, inp, out, _idx=idx):
                # inp is a tuple; take the first element (the actual input tensor)
                a = inp if not isinstance(inp, tuple) else inp[0]
                if self.offload_activations_to_cpu:
                    a = backend.to_cpu(a)
                activations.setdefault(_idx, []).append(a)

            def _bwd_hook(layer, grad_inp, grad_out, _idx=idx):
                g = grad_out if not isinstance(grad_out, tuple) else grad_out[0]
                if self.offload_activations_to_cpu:
                    g = backend.to_cpu(g)
                grad_outputs.setdefault(_idx, []).append(g)

            handles.append(backend.register_forward_hook(info.layer, _fwd_hook))
            handles.append(backend.register_backward_hook(info.layer, _bwd_hook))

        # --- Iterate over dataset -----------------------------------------
        partition_batch_count = 0
        partition_idx = 0
        partition_paths: List[str] = []
        partition_dir: Optional[str] = None
        if self.accumulator_offload_mode == "disk":
            partition_dir = self._make_accumulator_partition_dir("kfac_accumulators")

        def _flush_partition_to_checkpoint() -> None:
            nonlocal a_sums, g_sums, n_rows_per_layer, partition_idx
            if not a_sums:
                return

            if self.accumulator_offload_mode == "memory":
                self._checkpoint_factor_accumulators(a_sums, g_sums, n_rows_per_layer)
                return

            if self.accumulator_offload_mode != "disk":
                return

            if partition_dir is None:
                raise RuntimeError("partition_dir is required for disk accumulator offload mode.")

            partition_path = self._write_accumulator_partition_to_disk(
                partition_dir=partition_dir,
                partition_idx=partition_idx,
                tensors={"a_sums": a_sums, "g_sums": g_sums},
                n_rows_per_layer=n_rows_per_layer,
            )
            partition_idx += 1
            partition_paths.append(partition_path)
            self._factor_checkpoint = {
                "mode": "disk",
                "partition_dir": partition_dir,
                "partition_files": list(partition_paths),
            }

            a_sums = {}
            g_sums = {}
            n_rows_per_layer = {}

        try:
            try:
                for batch in train_dataset:
                    if isinstance(batch, (list, tuple)):
                        batch_tuple = tuple(batch)
                    else:
                        batch_tuple = (batch,)

                    # Use model's preprocessing
                    model_inp, y_true, sample_weight = model.process_batch_for_loss_fn(batch_tuple)

                    # Forward pass
                    activations.clear()
                    grad_outputs.clear()

                    # We need gradients w.r.t. outputs to trigger backward hooks.
                    # Compute the per-sample loss and backprop.
                    self._forward_backward(model, model_inp, y_true, sample_weight,
                                           activations, grad_outputs, layer_infos=layer_infos)

                    # Accumulate factors
                    for info in layer_infos:
                        idx = info.layer_idx
                        if idx not in activations or idx not in grad_outputs:
                            continue

                        a_list = activations[idx]
                        g_list = grad_outputs[idx]
                        if len(a_list) != len(g_list):
                            raise ValueError(
                                f"Activation/gradient capture count mismatch for layer {idx}: "
                                f"{len(a_list)} vs {len(g_list)}."
                            )

                        # Backward hooks run in reverse execution order, so
                        # reverse gradients to align with forward activations.
                        for a, g in zip(a_list, reversed(g_list)):
                            if self.offload_activations_to_cpu:
                                a = backend.to_device(a, reference=device_reference)
                                g = backend.to_device(g, reference=device_reference)

                            a_mat, g_mat = self._prepare_activation_gradient(info, a, g)

                            a_rows = _to_int(backend.get_batch_size(a_mat))
                            g_rows = _to_int(backend.get_batch_size(g_mat))
                            if a_rows != g_rows:
                                raise ValueError(
                                    f"Activation/gradient row mismatch for layer {idx}: "
                                    f"{a_rows} vs {g_rows}."
                                )

                            n_rows_per_layer[idx] = n_rows_per_layer.get(idx, 0) + a_rows

                            if idx not in a_sums:
                                a_sums[idx] = backend.matmul(backend.transpose(a_mat), a_mat)
                                g_sums[idx] = backend.matmul(backend.transpose(g_mat), g_mat)
                            else:
                                a_sums[idx] = a_sums[idx] + backend.matmul(backend.transpose(a_mat), a_mat)
                                g_sums[idx] = g_sums[idx] + backend.matmul(backend.transpose(g_mat), g_mat)

                    partition_batch_count += 1
                    if self.data_partition_size is not None and partition_batch_count >= self.data_partition_size:
                        _flush_partition_to_checkpoint()
                        partition_batch_count = 0
            finally:
                # --- Cleanup hooks --------------------------------------------
                for h in handles:
                    backend.remove_hook(h)

            if self.data_partition_size is not None and partition_batch_count > 0:
                _flush_partition_to_checkpoint()

            if self.accumulator_offload_mode == "disk":
                _flush_partition_to_checkpoint()

            if self.accumulator_offload_mode == "disk":
                merged_tensors, merged_rows = self._merge_accumulator_partitions_from_disk(
                    partition_paths,
                    tensor_names=("a_sums", "g_sums"),
                )

                for info in layer_infos:
                    idx = info.layer_idx
                    if idx not in merged_tensors["a_sums"]:
                        continue

                    n_rows = merged_rows.get(idx, 0)
                    if n_rows <= 0:
                        continue

                    scale = float(n_rows)
                    a_factor = merged_tensors["a_sums"][idx] / scale
                    g_factor = merged_tensors["g_sums"][idx] / scale

                    a_tensor = backend.to_device(backend.convert_to_tensor(a_factor), reference=device_reference)
                    g_tensor = backend.to_device(backend.convert_to_tensor(g_factor), reference=device_reference)

                    self.A[idx] = self._symmetrize_matrix(a_tensor)
                    self.G[idx] = self._symmetrize_matrix(g_tensor)
            else:
                # --- Normalize and store ------------------------------------------
                for info in layer_infos:
                    idx = info.layer_idx
                    if idx not in a_sums:
                        continue

                    n_rows = n_rows_per_layer.get(idx, 0)
                    if n_rows <= 0:
                        continue

                    scale = float(n_rows)
                    self.A[idx] = self._symmetrize_matrix(a_sums[idx] / scale)
                    self.G[idx] = self._symmetrize_matrix(g_sums[idx] / scale)
        finally:
            self._cleanup_accumulator_partition_dir(partition_dir)

    def _forward_backward(
        self,
        model: BaseInfluenceModel,
        model_inp: Tensor,
        y_true: Tensor,
        sample_weight: Optional[Tensor],
        activations: Dict[int, List[Tensor]],
        grad_outputs: Dict[int, List[Tensor]],
        layer_infos: Optional[List[LayerInfo]] = None,
    ) -> None:
        """Run forward + backward to populate hook-captured activations and gradients.

        In PyTorch we can just do a normal forward/backward pass and the hooks
        fire automatically.  In TensorFlow we need a GradientTape approach;
        the forward hook wrapper on ``layer.call`` fires during the tape-traced
        forward pass, and we manually invoke backward hooks with the per-layer
        output gradients obtained from the tape.
        """
        backend = self.backend
        active_layer_infos = self.layer_map.layers_info if layer_infos is None else layer_infos

        if backend.framework.value == "pytorch":
            self._forward_backward_pytorch(
                model,
                model_inp,
                y_true,
                sample_weight,
                use_true_fisher=self.fisher_type == "true",
            )
        else:
            self._forward_backward_tensorflow(
                model,
                model_inp,
                y_true,
                sample_weight,
                grad_outputs,
                active_layer_infos,
                use_true_fisher=self.fisher_type == "true",
            )

    def _sample_true_fisher_targets_pytorch(
        self,
        predictions: Any,
        y_true: Any,
        loss_function: LossFunction,
    ) -> Any:
        """Sample labels from the model predictive distribution (PyTorch)."""
        import torch
        import torch.nn as nn

        detached_predictions = predictions.detach()

        if isinstance(loss_function, nn.MSELoss):
            noise = torch.randn_like(detached_predictions)
            return detached_predictions + noise

        if isinstance(loss_function, nn.CrossEntropyLoss):
            logits = detached_predictions.movedim(1, -1)
            n_classes = logits.shape[-1]
            logits_flat = logits.reshape(-1, n_classes)
            sampled_flat = torch.distributions.Categorical(logits=logits_flat).sample()
            sampled = sampled_flat.reshape(logits.shape[:-1])
            return sampled.to(dtype=torch.long)

        if isinstance(loss_function, nn.BCEWithLogitsLoss):
            probs = torch.sigmoid(detached_predictions)
            return torch.bernoulli(probs).to(dtype=detached_predictions.dtype)

        if isinstance(loss_function, nn.BCELoss):
            probs = torch.clamp(detached_predictions, min=1e-7, max=1.0 - 1e-7)
            return torch.bernoulli(probs).to(dtype=detached_predictions.dtype)

        self._warn_true_fisher_fallback(loss_function)
        return y_true

    def _sample_true_fisher_targets_tensorflow(
        self,
        predictions: Any,
        y_true: Any,
        loss_function: LossFunction,
    ) -> Any:
        """Sample labels from the model predictive distribution (TensorFlow)."""
        import tensorflow as tf

        detached_predictions = tf.stop_gradient(predictions)

        if isinstance(loss_function, tf.keras.losses.MeanSquaredError):
            noise = tf.random.normal(tf.shape(detached_predictions), dtype=detached_predictions.dtype)
            return detached_predictions + noise

        if isinstance(loss_function, tf.keras.losses.CategoricalCrossentropy):
            probs = detached_predictions
            if getattr(loss_function, "from_logits", False):
                probs = tf.nn.softmax(probs, axis=-1)
            probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
            probs = tf.clip_by_value(probs, 1e-12, 1.0)
            n_classes = tf.shape(probs)[-1]
            probs_flat = tf.reshape(probs, (-1, n_classes))
            sampled_flat = tf.random.categorical(tf.math.log(probs_flat), 1)
            sampled_flat = tf.squeeze(sampled_flat, axis=-1)
            sampled = tf.reshape(sampled_flat, tf.shape(probs)[:-1])
            return tf.one_hot(tf.cast(sampled, tf.int32), depth=n_classes, dtype=detached_predictions.dtype)

        if isinstance(loss_function, tf.keras.losses.SparseCategoricalCrossentropy):
            probs = detached_predictions
            if getattr(loss_function, "from_logits", False):
                probs = tf.nn.softmax(probs, axis=-1)
            probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
            probs = tf.clip_by_value(probs, 1e-12, 1.0)
            n_classes = tf.shape(probs)[-1]
            probs_flat = tf.reshape(probs, (-1, n_classes))
            sampled_flat = tf.random.categorical(tf.math.log(probs_flat), 1)
            sampled_flat = tf.squeeze(sampled_flat, axis=-1)
            sampled = tf.reshape(sampled_flat, tf.shape(probs)[:-1])
            target_dtype = y_true.dtype if y_true is not None else tf.int64
            return tf.cast(sampled, target_dtype)

        if isinstance(loss_function, tf.keras.losses.BinaryCrossentropy):
            probs = detached_predictions
            if getattr(loss_function, "from_logits", False):
                probs = tf.math.sigmoid(probs)
            probs = tf.clip_by_value(probs, 1e-7, 1.0 - 1e-7)
            uniforms = tf.random.uniform(tf.shape(probs), dtype=probs.dtype)
            return tf.cast(uniforms < probs, detached_predictions.dtype)

        self._warn_true_fisher_fallback(loss_function)
        return y_true

    def _forward_backward_pytorch(
        self,
        model: BaseInfluenceModel,
        model_inp: Any,
        y_true: Any,
        sample_weight: Optional[Any],
        use_true_fisher: bool = False,
    ) -> None:
        """PyTorch: standard forward + backward; hooks fire automatically."""
        custom_forward_backward = getattr(model, "kfac_forward_backward", None)
        if callable(custom_forward_backward):
            custom_forward_backward(model_inp, y_true, sample_weight)
            return

        predictions = model.model(model_inp)
        targets = y_true
        if use_true_fisher:
            targets = self._sample_true_fisher_targets_pytorch(predictions, y_true, model.loss_function)

        loss = model.loss_function(predictions, targets)
        if sample_weight is not None:
            loss = loss * sample_weight
        total_loss = loss.sum()
        total_loss.backward()

        # Zero grad after capture (we don't need optimizer updates)
        model.model.zero_grad()

    def _forward_backward_tensorflow(
        self,
        model: BaseInfluenceModel,
        model_inp: Any,
        y_true: Any,
        sample_weight: Optional[Any],
        grad_outputs: Dict[int, List[Any]],
        layer_infos: List[LayerInfo],
        use_true_fisher: bool = False,
    ) -> None:
        """TensorFlow: use GradientTape + per-layer gradient for backward hooks.

        Strategy
        --------
        1. Create a persistent ``GradientTape``.
        2. Wrap each supported layer's ``call`` to both record intermediate
           outputs and call ``tape.watch`` on them *immediately*, so the tape
           traces operations downstream of each layer output.
        3. Run the full forward pass and compute the loss inside the tape
           context.
        4. After the tape context, use the tape to obtain per-layer output
           gradients and manually fire the backward hooks stored on each layer.
        5. In ``finally``, restore the original ``call`` for every layer to
           prevent stacking wrappers across batches.
        """
        import tensorflow as tf

        # Collect intermediate outputs so we can compute per-layer gradients
        layer_outputs: Dict[int, Any] = {}

        # Save original calls — will be restored in `finally`.
        original_calls: Dict[int, Callable] = {}

        try:
            with tf.GradientTape(persistent=True) as tape:
                # Install wrappers *inside* the tape context so `tape` is
                # available to the closure for `tape.watch`.
                for info in layer_infos:
                    idx = info.layer_idx
                    _orig = info.layer.call
                    original_calls[idx] = _orig

                    def _record_and_watch(*a, _idx=idx, _orig_fn=_orig, _tape=tape, **kw):
                        out = _orig_fn(*a, **kw)
                        # TODO(shared-layer-tf): keep all outputs here and
                        # compute one gradient per output for reused layers.
                        layer_outputs[_idx] = out
                        _tape.watch(out)
                        return out

                    info.layer.call = _record_and_watch

                # Forward pass — the wrapped ``call`` fires, populates
                # *layer_outputs*, and calls ``tape.watch`` on each output.
                # The forward hooks installed by the caller also fire and
                # populate *activations*.
                predictions = model.model(model_inp, training=False)

                targets = y_true
                if use_true_fisher:
                    targets = self._sample_true_fisher_targets_tensorflow(
                        predictions,
                        y_true,
                        model.loss_function,
                    )

                loss = model.loss_function(targets, predictions)
                if sample_weight is not None:
                    loss = loss * sample_weight
                batch_size = tf.shape(predictions)[0]
                loss = tf.reshape(loss, (batch_size, -1))
                loss = tf.reduce_mean(loss, axis=1)
                total_loss = tf.reduce_sum(loss)

            # Compute gradient of total_loss w.r.t. each layer output and
            # manually invoke backward hooks.
            for info in layer_infos:
                idx = info.layer_idx
                if idx not in layer_outputs:
                    continue
                try:
                    g = tape.gradient(total_loss, layer_outputs[idx])
                    if g is not None:
                        # Fire backward hooks
                        hooks = getattr(info.layer, '_kfac_backward_hooks', [])
                        if hooks:
                            for hook in hooks:
                                hook(info.layer, None, (g,))
                        else:
                            grad_outputs.setdefault(idx, []).append(g)
                except Exception:  # pylint: disable=broad-except
                    pass
            del tape
        finally:
            # Restore original calls to prevent stacking wrappers
            for info in layer_infos:
                idx = info.layer_idx
                if idx in original_calls:
                    info.layer.call = original_calls[idx]

    def _prepare_activation_gradient(
        self, info: LayerInfo, activation: Tensor, grad_output: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Reshape activation and gradient for factor accumulation.

        For Linear/Dense layers:
        - activation: ``(batch, n_in)`` -> append 1 column if bias -> ``(batch, n_in+bias)``
        - grad_output: ``(batch, n_out)``

        For Conv2d layers:
        - activation: ``(batch, C_in, H, W)`` -> unfold -> ``(batch*H'*W', C_in*kH*kW+bias)``
        - grad_output: ``(batch, C_out, H', W')`` -> ``(batch*H'*W', C_out)``

        Returns ``(a_mat, g_mat)`` where rows correspond to samples.
        """
        backend = self.backend

        if backend.is_conv2d_layer(info.layer):
            return self._prepare_conv2d(info, activation, grad_output)

        # Linear / Dense
        a = activation
        g = grad_output

        # Ensure 2-D: (batch, features)
        if backend.tensor_ndim(a) > 2:
            shape = backend.tensor_shape(a)
            a = backend.reshape(a, (shape[0], -1))

        if backend.tensor_ndim(g) > 2:
            shape = backend.tensor_shape(g)
            g = backend.reshape(g, (shape[0], -1))
        elif backend.tensor_ndim(g) == 1:
            g = backend.expand_dims(g, axis=0)

        if info.has_bias:
            # Append column of ones for the bias term
            ones = backend.ones_like(a[:, :1])
            a = backend.concat([a, ones], axis=1)

        return a, g

    def _prepare_conv2d(
        self, info: LayerInfo, activation: Tensor, grad_output: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Unfold Conv2d activations (im2col) and reshape gradients."""
        backend = self.backend

        # For PyTorch: activation is (B, C_in, H, W), grad_output is (B, C_out, H', W')
        # For TF/Keras: activation is (B, H, W, C_in), grad_output is (B, H', W', C_out)
        if backend.framework.value == "pytorch":
            return self._prepare_conv2d_pytorch(info, activation, grad_output)
        return self._prepare_conv2d_tensorflow(info, activation, grad_output)

    def _prepare_conv2d_pytorch(
        self, info: LayerInfo, activation: Any, grad_output: Any
    ) -> Tuple[Any, Any]:
        """PyTorch Conv2d unfolding."""
        import torch
        import torch.nn.functional as F

        layer = info.layer
        # Unfold using the layer's kernel_size, stride, padding, dilation
        a_unfold = F.unfold(
            activation,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
                if isinstance(layer.padding, int) else (0, 0),
            dilation=layer.dilation,
        )
        # a_unfold: (B, C_in*kH*kW, L) where L = H'*W'
        # Transpose to (B, L, C_in*kH*kW)
        a_unfold = a_unfold.permute(0, 2, 1)
        b, l, cin_k = a_unfold.shape

        if info.has_bias:
            ones = torch.ones(b, l, 1, device=a_unfold.device, dtype=a_unfold.dtype)
            a_unfold = torch.cat([a_unfold, ones], dim=2)

        # Reshape to (B*L, cin_k+bias)
        a_mat = a_unfold.reshape(b * l, -1)

        # grad_output: (B, C_out, H', W') -> (B, H'*W', C_out) -> (B*L, C_out)
        g = grad_output.permute(0, 2, 3, 1).reshape(b * l, -1)

        return a_mat, g

    def _prepare_conv2d_tensorflow(
        self, info: LayerInfo, activation: Any, grad_output: Any
    ) -> Tuple[Any, Any]:
        """TensorFlow Conv2d unfolding via extract_patches."""
        import tensorflow as tf

        layer = info.layer
        kernel_size = layer.kernel_size
        strides = layer.strides
        # Keras stores padding as string ('valid' or 'same')
        padding = layer.padding.upper() if isinstance(layer.padding, str) else 'VALID'

        # activation: (B, H, W, C_in)
        # tf.image.extract_patches -> (B, H', W', kH*kW*C_in)
        a_patches = tf.image.extract_patches(
            activation,
            sizes=[1, kernel_size[0], kernel_size[1], 1],
            strides=[1, strides[0], strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=padding,
        )
        shape = tf.shape(a_patches)
        b, hp, wp, patch_dim = shape[0], shape[1], shape[2], shape[3]
        l = hp * wp

        a_mat = tf.reshape(a_patches, [b * l, patch_dim])
        if info.has_bias:
            ones = tf.ones([b * l, 1], dtype=a_mat.dtype)
            a_mat = tf.concat([a_mat, ones], axis=1)

        # grad_output: (B, H', W', C_out) -> (B*L, C_out)
        g_shape = tf.shape(grad_output)
        g = tf.reshape(grad_output, [b * l, g_shape[-1]])

        return a_mat, g


# ---------------------------------------------------------------------------
# EK-FAC factor computation
# ---------------------------------------------------------------------------

class EKFACFactors(KroneckerFactors):
    """Extends :class:`KroneckerFactors` with eigendecomposition and corrected eigenvalues.

    After computing the base Kronecker factors ``A_l`` and ``G_l``, this class:

    1. Eigendecomposes each: ``A_l = Q_A Lambda_A Q_A^T``,  ``G_l = Q_G Lambda_G Q_G^T``.
    2. Estimates *corrected* diagonal eigenvalues from a second pass over training
       data (or a subset), following the EK-FAC procedure in Grosse et al. (2023).

    Parameters
    ----------
    model
        The influence model wrapper.
    train_dataset
        Batched training dataset.
    backend
        The backend abstraction.
    layer_map
        Pre-computed ``LayerParameterMap``.
    n_ekfac_samples
        Number of samples (batches are consumed until this many samples are seen)
        used for corrected eigenvalue estimation.  ``None`` means use all data.
    fisher_type
        Fisher variant used for curvature estimation: ``"empirical"`` (default)
        or ``"true"``.
    module_partition_size
        Optional number of supported layers to process per pass.
    offload_activations_to_cpu
        Whether to offload hook-captured activations/gradients to CPU.
    data_partition_size
        Optional number of batches per data partition.
    accumulator_offload_mode
        Accumulator checkpoint mode used at data-partition boundaries:
        ``"none"`` (default), ``"memory"``, or ``"disk"``.
    accumulator_offload_dir
        Optional directory where temporary disk-offload partition files are
        stored when ``accumulator_offload_mode="disk"``.
    keep_accumulator_offload_artifacts
        If ``True``, keep temporary disk-offload partition files after factor
        computation. Otherwise they are removed.
    """

    def __init__(
        self,
        model: BaseInfluenceModel,
        train_dataset: DatasetLike,
        backend: BaseBackend,
        layer_map: LayerParameterMap,
        n_ekfac_samples: Optional[int] = None,
        fisher_type: str = "empirical",
        module_partition_size: Optional[int] = None,
        offload_activations_to_cpu: bool = False,
        data_partition_size: Optional[int] = None,
        accumulator_offload_mode: str = "none",
        accumulator_offload_dir: Optional[str] = None,
        keep_accumulator_offload_artifacts: bool = False,
    ):
        # Compute base K-FAC factors (A, G)
        super().__init__(
            model,
            train_dataset,
            backend,
            layer_map,
            fisher_type=fisher_type,
            module_partition_size=module_partition_size,
            offload_activations_to_cpu=offload_activations_to_cpu,
            data_partition_size=data_partition_size,
            accumulator_offload_mode=accumulator_offload_mode,
            accumulator_offload_dir=accumulator_offload_dir,
            keep_accumulator_offload_artifacts=keep_accumulator_offload_artifacts,
        )

        # Eigendecompose A and G
        self.Q_A: Dict[int, Tensor] = {}
        self.Lambda_A: Dict[int, Tensor] = {}
        self.Q_G: Dict[int, Tensor] = {}
        self.Lambda_G: Dict[int, Tensor] = {}

        for info in layer_map.layers_info:
            idx = info.layer_idx
            if idx in self.A:
                a_factor = self._symmetrize_matrix(self.A[idx])
                g_factor = self._symmetrize_matrix(self.G[idx])
                self.A[idx] = a_factor
                self.G[idx] = g_factor

                a_dtype = backend.get_dtype(a_factor)
                g_dtype = backend.get_dtype(g_factor)

                # Get the eigenvalues/eigenvectors in float64 for stability, then cast back
                lam_a, q_a = backend.eigh(backend.cast(a_factor, backend.float64_dtype()))
                lam_g, q_g = backend.eigh(backend.cast(g_factor, backend.float64_dtype()))
                self.Q_A[idx] = q_a
                self.Lambda_A[idx] = lam_a
                self.Q_G[idx] = q_g
                self.Lambda_G[idx] = lam_g

                self.Q_A[idx] = backend.cast(self.Q_A[idx], a_dtype)
                self.Lambda_A[idx] = backend.cast(self.Lambda_A[idx], a_dtype)
                self.Q_G[idx] = backend.cast(self.Q_G[idx], g_dtype)
                self.Lambda_G[idx] = backend.cast(self.Lambda_G[idx], g_dtype)

        # Corrected eigenvalues
        self.Lambda_corrected: Dict[int, Tensor] = {}
        self._corrected_checkpoint: Optional[Dict[str, Any]] = None
        self.n_ekfac_samples = n_ekfac_samples
        self._estimate_corrected_eigenvalues(train_dataset, n_ekfac_samples)

    def save_to_dir(self, path: str) -> None:
        """Serialize computed EK-FAC factors to *path*."""

        def _writer(tmp_dir: str) -> None:
            metadata = self._build_checkpoint_metadata(method="ekfac")
            metadata["n_ekfac_samples"] = self.n_ekfac_samples

            for layer_entry in metadata["layers"]:
                layer_idx = int(layer_entry["layer_idx"])
                if layer_idx not in self.A or layer_idx not in self.G:
                    continue
                if layer_idx not in self.Q_A or layer_idx not in self.Q_G:
                    continue
                if layer_idx not in self.Lambda_A or layer_idx not in self.Lambda_G:
                    continue
                if layer_idx not in self.Lambda_corrected:
                    continue

                tensors = {
                    "A": self.A[layer_idx],
                    "G": self.G[layer_idx],
                    "Q_A": self.Q_A[layer_idx],
                    "Q_G": self.Q_G[layer_idx],
                    "Lambda_A": self.Lambda_A[layer_idx],
                    "Lambda_G": self.Lambda_G[layer_idx],
                    "Lambda_corrected": self.Lambda_corrected[layer_idx],
                }

                tensor_entries = {}
                for tensor_name, tensor in tensors.items():
                    tensor_array = np.asarray(self.backend.to_numpy(tensor))
                    file_name = f"{tensor_name}_layer_{layer_idx}.npy"
                    np.save(os.path.join(tmp_dir, file_name), tensor_array, allow_pickle=False)
                    tensor_entries[tensor_name] = _make_tensor_metadata(file_name, tensor_array)

                layer_entry["tensors"] = tensor_entries

            _write_checkpoint_metadata(tmp_dir, metadata)

        _atomic_write_directory(path, _writer)

    @classmethod
    def load_from_dir(
        cls,
        model: BaseInfluenceModel,
        backend: BaseBackend,
        layer_map: LayerParameterMap,
        path: str,
        fisher_type: str = "empirical",
        module_partition_size: Optional[int] = None,
        offload_activations_to_cpu: bool = False,
        data_partition_size: Optional[int] = None,
        accumulator_offload_mode: str = "none",
        accumulator_offload_dir: Optional[str] = None,
        keep_accumulator_offload_artifacts: bool = False,
        n_ekfac_samples: Optional[int] = None,
    ) -> "EKFACFactors":
        """Load precomputed EK-FAC factors from *path* without recomputing."""
        if accumulator_offload_mode not in ("none", "memory", "disk"):
            raise ValueError(
                "accumulator_offload_mode must be one of 'none', 'memory', or 'disk'."
            )

        metadata = _read_checkpoint_metadata(path)
        _validate_checkpoint_compatibility(
            metadata=metadata,
            expected_method="ekfac",
            expected_fisher_type=fisher_type,
            model=model,
            layer_map=layer_map,
            expected_n_ekfac_samples=n_ekfac_samples,
        )

        instance = cls.__new__(cls)
        instance.backend = backend
        instance.model = model
        instance.layer_map = layer_map
        instance.fisher_type = fisher_type
        instance.module_partition_size = module_partition_size
        instance.offload_activations_to_cpu = offload_activations_to_cpu
        instance.data_partition_size = data_partition_size
        instance.accumulator_offload_mode = accumulator_offload_mode
        instance.accumulator_offload_dir = accumulator_offload_dir
        instance.keep_accumulator_offload_artifacts = keep_accumulator_offload_artifacts
        instance._true_fisher_warning_emitted = bool(metadata.get("true_fisher_fallback_used", False))
        instance._factor_checkpoint = None

        instance.A = {}
        instance.G = {}
        instance.Q_A = {}
        instance.Lambda_A = {}
        instance.Q_G = {}
        instance.Lambda_G = {}
        instance.Lambda_corrected = {}
        instance._corrected_checkpoint = None
        instance.n_ekfac_samples = n_ekfac_samples

        reference_tensor = instance._get_device_reference_tensor()
        required_tensor_names = (
            "A",
            "G",
            "Q_A",
            "Q_G",
            "Lambda_A",
            "Lambda_G",
            "Lambda_corrected",
        )

        for layer_entry in metadata["layers"]:
            tensors = layer_entry.get("tensors")
            if not isinstance(tensors, dict):
                continue

            layer_idx = int(layer_entry["layer_idx"])
            missing_names = [name for name in required_tensor_names if name not in tensors]
            if missing_names:
                raise ValueError(
                    f"Invalid EK-FAC checkpoint metadata for layer {layer_idx}: "
                    f"missing tensors {missing_names}."
                )

            instance.A[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "A",
                tensors["A"],
                reference_tensor,
            )
            instance.G[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "G",
                tensors["G"],
                reference_tensor,
            )
            instance.Q_A[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "Q_A",
                tensors["Q_A"],
                reference_tensor,
            )
            instance.Q_G[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "Q_G",
                tensors["Q_G"],
                reference_tensor,
            )
            instance.Lambda_A[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "Lambda_A",
                tensors["Lambda_A"],
                reference_tensor,
            )
            instance.Lambda_G[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "Lambda_G",
                tensors["Lambda_G"],
                reference_tensor,
            )
            instance.Lambda_corrected[layer_idx] = _load_tensor_from_checkpoint(
                backend,
                path,
                layer_idx,
                "Lambda_corrected",
                tensors["Lambda_corrected"],
                reference_tensor,
            )

        return instance

    def _estimate_corrected_eigenvalues(
        self, train_dataset: DatasetLike, n_ekfac_samples: Optional[int]
    ) -> None:
        """Estimate corrected eigenvalues, optionally using module partitions."""
        for layer_partition in self._iter_layer_partitions():
            self._estimate_corrected_eigenvalues_for_layer_partition(
                train_dataset,
                n_ekfac_samples,
                layer_partition,
            )

    def _checkpoint_corrected_accumulators(
        self,
        corrected_sums: Dict[int, Tensor],
        n_rows_per_layer: Dict[int, int],
    ) -> None:
        """Store in-memory EK-FAC checkpoint state at data-partition boundaries."""
        self._corrected_checkpoint = {
            "mode": "memory",
            "corrected_sums": corrected_sums,
            "n_rows_per_layer": n_rows_per_layer,
        }

    def _estimate_corrected_eigenvalues_for_layer_partition(
        self,
        train_dataset: DatasetLike,
        n_ekfac_samples: Optional[int],
        layer_infos: List[LayerInfo],
    ) -> None:
        """Estimate the corrected diagonal eigenvalues for EK-FAC.

        For each layer l, the corrected eigenvalue matrix ``Λ_corrected`` is a
        flattened vector of shape ``(n_out * n_in_eff,)`` where ``n_in_eff``
        includes bias.  Each entry ``(i,j)`` is estimated as:

            E[ (Q_G^T g)_i^2  *  (Q_A^T a)_j^2 ]

        i.e. the expected squared Kronecker-rotated per-sample quantities.
        """
        backend = self.backend
        model = self.model
        device_reference = self._get_device_reference_tensor()

        def _to_int(value: Any) -> int:
            """Convert backend scalar values to Python int."""
            try:
                return int(value)
            except TypeError:
                return int(value.numpy())

        # Storage for running sums: layer_idx -> (n_out, n_in_eff) accumulator
        corrected_sums: Dict[int, Tensor] = {}
        n_rows_per_layer: Dict[int, int] = {}
        n_samples = 0

        # Register hooks (same as factor computation)
        handles = []
        # Keep every hook invocation so reused layers contribute all calls.
        activations: Dict[int, List[Tensor]] = {}
        grad_outputs_captured: Dict[int, List[Tensor]] = {}

        for info in layer_infos:
            idx = info.layer_idx

            def _fwd_hook(layer, inp, out, _idx=idx):
                a = inp if not isinstance(inp, tuple) else inp[0]
                if self.offload_activations_to_cpu:
                    a = backend.to_cpu(a)
                activations.setdefault(_idx, []).append(a)

            def _bwd_hook(layer, grad_inp, grad_out, _idx=idx):
                g = grad_out if not isinstance(grad_out, tuple) else grad_out[0]
                if self.offload_activations_to_cpu:
                    g = backend.to_cpu(g)
                grad_outputs_captured.setdefault(_idx, []).append(g)

            handles.append(backend.register_forward_hook(info.layer, _fwd_hook))
            handles.append(backend.register_backward_hook(info.layer, _bwd_hook))

        partition_batch_count = 0
        partition_idx = 0
        partition_paths: List[str] = []
        partition_dir: Optional[str] = None
        if self.accumulator_offload_mode == "disk":
            partition_dir = self._make_accumulator_partition_dir("ekfac_accumulators")

        def _flush_partition_to_checkpoint() -> None:
            nonlocal corrected_sums, n_rows_per_layer, partition_idx
            if not corrected_sums:
                return

            if self.accumulator_offload_mode == "memory":
                self._checkpoint_corrected_accumulators(corrected_sums, n_rows_per_layer)
                return

            if self.accumulator_offload_mode != "disk":
                return

            if partition_dir is None:
                raise RuntimeError("partition_dir is required for disk accumulator offload mode.")

            partition_path = self._write_accumulator_partition_to_disk(
                partition_dir=partition_dir,
                partition_idx=partition_idx,
                tensors={"corrected_sums": corrected_sums},
                n_rows_per_layer=n_rows_per_layer,
            )
            partition_idx += 1
            partition_paths.append(partition_path)
            self._corrected_checkpoint = {
                "mode": "disk",
                "partition_dir": partition_dir,
                "partition_files": list(partition_paths),
            }

            corrected_sums = {}
            n_rows_per_layer = {}

        try:
            try:
                for batch in train_dataset:
                    if isinstance(batch, (list, tuple)):
                        batch_tuple = tuple(batch)
                    else:
                        batch_tuple = (batch,)

                    model_inp, y_true, sample_weight = model.process_batch_for_loss_fn(batch_tuple)
                    batch_size = _to_int(backend.get_batch_size(model_inp))
                    n_samples += batch_size

                    activations.clear()
                    grad_outputs_captured.clear()

                    self._forward_backward(
                        model, model_inp, y_true, sample_weight,
                        activations, grad_outputs_captured,
                        layer_infos=layer_infos,
                    )

                    for info in layer_infos:
                        idx = info.layer_idx
                        if idx not in activations or idx not in grad_outputs_captured:
                            continue
                        if idx not in self.Q_A:
                            continue

                        a_list = activations[idx]
                        g_list = grad_outputs_captured[idx]
                        if len(a_list) != len(g_list):
                            raise ValueError(
                                f"Activation/gradient capture count mismatch for layer {idx}: "
                                f"{len(a_list)} vs {len(g_list)}."
                            )

                        # Backward hooks run in reverse execution order, so
                        # reverse gradients to align with forward activations.
                        for a, g in zip(a_list, reversed(g_list)):
                            if self.offload_activations_to_cpu:
                                a = backend.to_device(a, reference=device_reference)
                                g = backend.to_device(g, reference=device_reference)

                            a_mat, g_mat = self._prepare_activation_gradient(info, a, g)

                            a_rows = _to_int(backend.get_batch_size(a_mat))
                            g_rows = _to_int(backend.get_batch_size(g_mat))
                            if a_rows != g_rows:
                                raise ValueError(
                                    f"Activation/gradient row mismatch for layer {idx}: "
                                    f"{a_rows} vs {g_rows}."
                                )

                            n_rows_per_layer[idx] = n_rows_per_layer.get(idx, 0) + a_rows

                            # Rotate into eigenbasis: (batch, n_in_eff) @ Q_A -> (batch, n_in_eff)
                            a_rot = backend.matmul(a_mat, self.Q_A[idx])
                            # (batch, n_out) @ Q_G -> (batch, n_out)
                            g_rot = backend.matmul(g_mat, self.Q_G[idx])

                            # Corrected eigenvalue: E[ g_rot_i^2 * a_rot_j^2 ]
                            # g_rot^2: (batch, n_out), a_rot^2: (batch, n_in_eff)
                            # Outer per sample then average: (batch, n_out, 1) * (batch, 1, n_in_eff)
                            # Sum across batch -> (n_out, n_in_eff)
                            g_sq = backend.multiply(g_rot, g_rot)  # (batch, n_out)
                            a_sq = backend.multiply(a_rot, a_rot)  # (batch, n_in_eff)

                            # Equivalent to summing per-sample outer products, but avoids
                            # materializing a large 3-D tensor of shape
                            # (batch, n_out, n_in_eff).
                            batch_sum = backend.matmul(backend.transpose(g_sq), a_sq)
                            if idx not in corrected_sums:
                                corrected_sums[idx] = batch_sum
                            else:
                                corrected_sums[idx] = corrected_sums[idx] + batch_sum

                    partition_batch_count += 1
                    if self.data_partition_size is not None and partition_batch_count >= self.data_partition_size:
                        _flush_partition_to_checkpoint()
                        partition_batch_count = 0

                    if n_ekfac_samples is not None and n_samples >= n_ekfac_samples:
                        break
            finally:
                for h in handles:
                    backend.remove_hook(h)

            if self.data_partition_size is not None and partition_batch_count > 0:
                _flush_partition_to_checkpoint()

            if self.accumulator_offload_mode == "disk":
                _flush_partition_to_checkpoint()

            if self.accumulator_offload_mode == "disk":
                merged_tensors, merged_rows = self._merge_accumulator_partitions_from_disk(
                    partition_paths,
                    tensor_names=("corrected_sums",),
                )
                merged_corrected = merged_tensors["corrected_sums"]

                for info in layer_infos:
                    idx = info.layer_idx
                    if idx not in merged_corrected:
                        continue

                    n_rows = merged_rows.get(idx, 0)
                    if n_rows <= 0:
                        continue

                    corrected_array = merged_corrected[idx] / float(n_rows)
                    corrected_tensor = backend.to_device(
                        backend.convert_to_tensor(corrected_array),
                        reference=device_reference,
                    )
                    self.Lambda_corrected[idx] = backend.reshape(corrected_tensor, (-1,))
            else:
                # Normalize and flatten
                for info in layer_infos:
                    idx = info.layer_idx
                    if idx not in corrected_sums:
                        continue

                    n_rows = n_rows_per_layer.get(idx, 0)
                    if n_rows <= 0:
                        continue

                    # (n_out, n_in_eff) -> flatten to (n_out * n_in_eff,)
                    corrected_factor = corrected_sums[idx] / float(n_rows)
                    self.Lambda_corrected[idx] = backend.reshape(corrected_factor, (-1,))
        finally:
            self._cleanup_accumulator_partition_dir(partition_dir)
