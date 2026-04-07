# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Utilities shared by factor and score partitioning code.
"""
import os
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from ..common.backend import BaseBackend
from ..types import Tensor


T = TypeVar("T")


def chunk_items(items: Sequence[T], chunk_size: int) -> List[List[T]]:
    """Split *items* into chunks of at most *chunk_size* entries."""
    return [list(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]


def remove_path(path: str) -> None:
    """Remove a file or directory if it exists."""
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def atomic_write_directory(target_dir: str, writer: Callable[[str], None], prefix: str = ".tmp_partition_") -> None:
    """Atomically write a checkpoint directory by replacing it at the end."""
    parent_dir = os.path.dirname(target_dir) or "."
    os.makedirs(parent_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=prefix, dir=parent_dir)
    try:
        writer(tmp_dir)
        remove_path(target_dir)
        os.replace(tmp_dir, target_dir)
    except Exception:
        remove_path(tmp_dir)
        raise


def make_partition_dir(prefix: str, base_dir: Optional[str] = None) -> str:
    """Create a directory for temporary partition files."""
    if base_dir is None:
        return tempfile.mkdtemp(prefix=f"{prefix}_")

    os.makedirs(base_dir, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{prefix}_", dir=base_dir)


def cleanup_partition_dir(partition_dir: Optional[str], keep_artifacts: bool) -> None:
    """Cleanup temporary partition files unless explicitly preserved."""
    if partition_dir is None or keep_artifacts:
        return
    remove_path(partition_dir)


def write_tensor_partition_to_disk(
    backend: BaseBackend,
    partition_dir: str,
    partition_idx: int,
    tensors: Dict[str, Dict[int, Tensor]],
    n_rows_per_layer: Dict[int, int],
) -> str:
    """Serialize one tensor partition to a compressed ``.npz`` file."""
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
            tensor_array = np.asarray(backend.to_numpy(tensor_dict[layer_idx]))
            payload[f"{tensor_name}_{layer_idx}"] = tensor_array

    partition_path = os.path.join(partition_dir, f"partition_{partition_idx:06d}.npz")
    np.savez(partition_path, **payload)
    return partition_path


def merge_tensor_partitions_from_disk(
    partition_paths: List[str],
    tensor_names: Tuple[str, ...],
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[int, int]]:
    """Load and merge serialized tensor partitions from disk."""
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
