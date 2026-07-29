# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Immutable, sharded storage for vector representations and sample payloads."""
# Manifest validation is intentionally kept together so cross-field constraints
# remain visible alongside the strict schema checks.
# pylint: disable=too-many-branches
import json
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np


_SCHEMA = "deel.influenciae.representation_store"
_SCHEMA_VERSION = 1
_MANIFEST = "manifest.json"
_MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "stage",
    "fingerprints",
    "vector_dim",
    "vector_dtype",
    "payload_shape",
    "payload_dtype",
    "count",
    "shard_size",
    "shards",
}
_SHARD_KEYS = {"ids", "vectors", "payload", "count"}


@dataclass(frozen=True)
class RepresentationShard:
    """A shard of sample IDs, vectors, and optional payload rows."""

    ids: np.ndarray
    vectors: np.ndarray
    payload: Optional[np.ndarray] = None

    def __iter__(self):
        """Allow tuple unpacking as ``ids, vectors, payload``."""
        return iter((self.ids, self.vectors, self.payload))


class RepresentationStore(ABC):
    """Read-only interface shared by memory and directory stores."""

    vector_dim: int
    vector_dtype: np.dtype
    payload_shape: Optional[Tuple[int, ...]]
    payload_dtype: Optional[np.dtype]
    stage: str
    fingerprints: Mapping[str, str]

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the number of stored representations."""

    def __len__(self) -> int:
        return self.count

    @abstractmethod
    def iter_shards(self) -> Iterator[RepresentationShard]:
        """Yield individual shards without materializing the complete store."""

    def gather(self, ids: Any) -> Optional[np.ndarray]:
        """Gather payload rows in the requested ID shape without loading vectors.

        ``None`` is returned for stores without a payload. Missing or duplicate
        stored IDs are rejected because either case makes an ID lookup ambiguous.
        """
        requested_array = _validate_ids(ids, "gather")
        requested_shape = requested_array.shape
        requested = requested_array.reshape(-1)
        if self.payload_shape is None:
            return None
        assert self.payload_dtype is not None
        output = np.empty((requested.size,) + self.payload_shape, dtype=self.payload_dtype)
        found = np.zeros(requested.size, dtype=bool)
        positions: dict[int, list[int]] = {}
        for position, sample_id in enumerate(requested.tolist()):
            positions.setdefault(sample_id, []).append(position)

        for shard in self.iter_shards():
            for row, sample_id in enumerate(shard.ids.tolist()):
                destinations = positions.get(sample_id)
                if destinations is None:
                    continue
                if found[destinations[0]]:
                    raise ValueError(f"Stored sample ID {sample_id} is duplicated.")
                assert shard.payload is not None
                output[destinations] = shard.payload[row]
                found[destinations] = True

        if not np.all(found):
            missing = requested[~found]
            raise KeyError(f"Sample IDs were not found: {missing.tolist()}.")
        return output.reshape(requested_shape + self.payload_shape)


class _RepresentationWriter(RepresentationStore):  # pylint: disable=too-many-instance-attributes
    """Shared fixed-buffer append implementation."""

    def __init__(
        self,
        vector_dim: int,
        *,
        shard_size: int = 1024,
        vector_dtype: Any = np.float32,
        payload_shape: Optional[Sequence[int]] = None,
        payload_dtype: Optional[Any] = None,
        stage: str = "representations",
        fingerprints: Optional[Mapping[str, str]] = None,
    ) -> None:
        if isinstance(vector_dim, bool) or not isinstance(vector_dim, (int, np.integer)) or vector_dim <= 0:
            raise ValueError("vector_dim must be a strictly positive integer.")
        if isinstance(shard_size, bool) or not isinstance(shard_size, (int, np.integer)) or shard_size <= 0:
            raise ValueError("shard_size must be a strictly positive integer.")
        self.vector_dim = int(vector_dim)
        self.vector_dtype = np.dtype(vector_dtype)
        if not np.issubdtype(self.vector_dtype, np.number) or self.vector_dtype.hasobject:
            raise ValueError("vector_dtype must be a non-object numeric dtype.")
        self.payload_shape = _validate_payload_shape(payload_shape)
        if (self.payload_shape is None) != (payload_dtype is None):
            raise ValueError("payload_shape and payload_dtype must either both be set or both be None.")
        self.payload_dtype = None if payload_dtype is None else np.dtype(payload_dtype)
        if self.payload_dtype is not None and self.payload_dtype.hasobject:
            raise ValueError("payload_dtype cannot contain Python objects.")
        self.stage = _validate_stage(stage)
        self.fingerprints = _validate_fingerprints(fingerprints or {})
        self.shard_size = int(shard_size)
        self._count = 0
        self._buffered = 0
        self._finalized = False
        self._seen_ids: set[int] = set()
        self._ids_buffer = np.empty(self.shard_size, dtype=np.int64)
        self._vectors_buffer = np.empty((self.shard_size, self.vector_dim), dtype=self.vector_dtype)
        self._payload_buffer = None
        if self.payload_shape is not None:
            assert self.payload_dtype is not None
            self._payload_buffer = np.empty(
                (self.shard_size,) + self.payload_shape, dtype=self.payload_dtype
            )

    @property
    def count(self) -> int:
        return self._count + self._buffered

    @property
    def finalized(self) -> bool:
        """Return whether the immutable store has been finalized."""
        return self._finalized

    def append(self, ids: Any, vectors: Any, payload: Optional[Any] = None) -> None:
        """Append rows, slicing large batches directly into fixed-size shards."""
        if self._finalized:
            raise RuntimeError("Cannot append to a finalized representation store.")
        ids_array = _validate_ids(ids, "append").reshape(-1)
        ids_list = ids_array.tolist()
        if len(set(ids_list)) != len(ids_list):
            raise ValueError("Sample IDs must be unique within each appended batch.")
        repeated = self._seen_ids.intersection(ids_list)
        if repeated:
            raise ValueError(f"Sample IDs are already stored: {sorted(repeated)}.")
        vectors_array = np.asarray(vectors)
        if vectors_array.ndim != 2 or vectors_array.shape[1] != self.vector_dim:
            raise ValueError(f"vectors must have shape (batch, {self.vector_dim}); got {vectors_array.shape}.")
        if vectors_array.shape[0] != ids_array.size:
            raise ValueError("ids and vectors must have the same batch size.")
        vectors_array = vectors_array.astype(self.vector_dtype, copy=False)

        payload_array = None
        if self.payload_shape is None:
            if payload is not None:
                raise ValueError("payload was supplied to a store configured without payloads.")
        else:
            if payload is None:
                raise ValueError("payload is required for this representation store.")
            assert self.payload_dtype is not None
            payload_array = np.asarray(payload)
            expected = (ids_array.size,) + self.payload_shape
            if payload_array.shape != expected:
                raise ValueError(f"payload must have shape {expected}; got {payload_array.shape}.")
            payload_array = payload_array.astype(self.payload_dtype, copy=False)

        self._seen_ids.update(ids_list)

        offset = 0
        while offset < ids_array.size:
            rows = min(self.shard_size - self._buffered, ids_array.size - offset)
            target = slice(self._buffered, self._buffered + rows)
            source = slice(offset, offset + rows)
            self._ids_buffer[target] = ids_array[source]
            self._vectors_buffer[target] = vectors_array[source]
            if self._payload_buffer is not None:
                assert payload_array is not None
                self._payload_buffer[target] = payload_array[source]
            self._buffered += rows
            offset += rows
            if self._buffered == self.shard_size:
                self._flush_buffer()

    def finalize(self) -> None:
        """Flush residual rows and make the store immutable."""
        if self._finalized:
            return
        if self._buffered:
            self._flush_buffer()
        self._publish()
        self._finalized = True

    def iter_shards(self) -> Iterator[RepresentationShard]:
        if not self._finalized:
            raise RuntimeError("Call finalize() before iterating a representation store.")
        yield from self._iter_published_shards()

    def _flush_buffer(self) -> None:
        rows = self._buffered
        ids = self._ids_buffer[:rows].copy()
        vectors = self._vectors_buffer[:rows].copy()
        payload = None if self._payload_buffer is None else self._payload_buffer[:rows].copy()
        self._write_shard(RepresentationShard(ids, vectors, payload))
        self._count += rows
        self._buffered = 0

    @abstractmethod
    def _write_shard(self, shard: RepresentationShard) -> None:
        pass

    @abstractmethod
    def _publish(self) -> None:
        pass

    @abstractmethod
    def _iter_published_shards(self) -> Iterator[RepresentationShard]:
        pass


class MemoryRepresentationStore(_RepresentationWriter):
    """Immutable representation store backed by independent in-memory shards."""

    def __init__(self, vector_dim: int, **kwargs: Any) -> None:
        super().__init__(vector_dim, **kwargs)
        self._shards: list[RepresentationShard] = []

    def _write_shard(self, shard: RepresentationShard) -> None:
        for array in (shard.ids, shard.vectors, shard.payload):
            if array is not None:
                array.flags.writeable = False
        self._shards.append(shard)

    def _publish(self) -> None:
        return

    def _iter_published_shards(self) -> Iterator[RepresentationShard]:
        yield from self._shards


class DirectoryRepresentationStore(_RepresentationWriter):  # pylint: disable=too-many-instance-attributes
    """Representation store atomically published as memory-mapped NumPy shards."""

    def __init__(self, path: Any, vector_dim: int, *, overwrite: bool = False, **kwargs: Any) -> None:
        self.path = Path(path)
        self._overwrite = bool(overwrite)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Representation store already exists at '{self.path}'.")
        parent = self.path.parent
        if not parent.is_dir():
            raise FileNotFoundError(f"Parent directory does not exist: '{parent}'.")
        self._staging_path: Optional[Path] = parent / f".{self.path.name}.staging-{uuid.uuid4().hex}"
        self._staging_path.mkdir()
        self._shard_entries: list[dict[str, Any]] = []
        try:
            super().__init__(vector_dim, **kwargs)
        except BaseException:
            shutil.rmtree(self._staging_path, ignore_errors=True)
            raise

    def _write_shard(self, shard: RepresentationShard) -> None:
        assert self._staging_path is not None
        index = len(self._shard_entries)
        ids_name = f"ids_{index:06d}.npy"
        vectors_name = f"vectors_{index:06d}.npy"
        payload_name = None if shard.payload is None else f"payload_{index:06d}.npy"
        np.save(self._staging_path / ids_name, shard.ids, allow_pickle=False)
        np.save(self._staging_path / vectors_name, shard.vectors, allow_pickle=False)
        if shard.payload is not None:
            assert payload_name is not None
            np.save(self._staging_path / payload_name, shard.payload, allow_pickle=False)
        self._shard_entries.append(
            {"ids": ids_name, "vectors": vectors_name, "payload": payload_name, "count": len(shard.ids)}
        )

    def _publish(self) -> None:
        assert self._staging_path is not None
        manifest = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "stage": self.stage,
            "fingerprints": dict(self.fingerprints),
            "vector_dim": self.vector_dim,
            "vector_dtype": self.vector_dtype.str,
            "payload_shape": None if self.payload_shape is None else list(self.payload_shape),
            "payload_dtype": None if self.payload_dtype is None else self.payload_dtype.str,
            "count": self._count,
            "shard_size": self.shard_size,
            "shards": self._shard_entries,
        }
        with open(self._staging_path / _MANIFEST, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

        backup = self.path.parent / f".{self.path.name}.backup-{uuid.uuid4().hex}"
        moved_previous = False
        try:
            if self.path.exists():
                if not self._overwrite:
                    raise FileExistsError(f"Representation store already exists at '{self.path}'.")
                os.replace(self.path, backup)
                moved_previous = True
            os.replace(self._staging_path, self.path)
        except BaseException:
            if moved_previous and backup.exists() and not self.path.exists():
                os.replace(backup, self.path)
            raise
        if moved_previous:
            shutil.rmtree(backup, ignore_errors=True)

    def _iter_published_shards(self) -> Iterator[RepresentationShard]:
        for entry in self._shard_entries:
            yield self._load_shard(self.path, entry)

    @classmethod
    def load(
        cls,
        path: Any,
        *,
        expected_stage: Optional[str] = None,
        expected_fingerprints: Optional[Mapping[str, str]] = None,
    ) -> "DirectoryRepresentationStore":
        """Load and fully validate an already-published directory store."""
        root = Path(path)
        manifest_path = root / _MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Representation store manifest not found at '{manifest_path}'.")
        try:
            with open(manifest_path, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid representation store manifest: {exc}.") from exc
        cls._validate_manifest(manifest, expected_stage, expected_fingerprints)

        instance = cls.__new__(cls)
        instance.path = root
        instance._staging_path = None
        instance._overwrite = False
        instance.vector_dim = manifest["vector_dim"]
        instance.vector_dtype = np.dtype(manifest["vector_dtype"])
        shape = manifest["payload_shape"]
        instance.payload_shape = None if shape is None else tuple(shape)
        dtype = manifest["payload_dtype"]
        instance.payload_dtype = None if dtype is None else np.dtype(dtype)
        instance.stage = manifest["stage"]
        instance.fingerprints = manifest["fingerprints"]
        instance.shard_size = manifest["shard_size"]
        instance._count = manifest["count"]
        instance._buffered = 0
        instance._finalized = True
        instance._shard_entries = manifest["shards"]

        total = 0
        for entry in instance._shard_entries:
            shard = cls._load_shard(root, entry)
            cls._validate_shard(instance, shard, entry)
            total += entry["count"]
        if total != instance._count:
            raise ValueError(f"Manifest count is {instance._count}, but shard counts total {total}.")
        return instance

    load_from_dir = load

    @staticmethod
    def _validate_manifest(
        manifest: Any,
        expected_stage: Optional[str],
        expected_fingerprints: Optional[Mapping[str, str]],
    ) -> None:
        if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
            raise ValueError("Representation store manifest has an invalid schema.")
        schema_version = manifest["schema_version"]
        if manifest["schema"] != _SCHEMA or isinstance(schema_version, bool) \
                or not isinstance(schema_version, int) or schema_version != _SCHEMA_VERSION:
            raise ValueError("Unsupported representation store schema or schema version.")
        _validate_stage(manifest["stage"])
        fingerprints = _validate_fingerprints(manifest["fingerprints"])
        _strict_positive_int(manifest["vector_dim"], "vector_dim")
        _strict_positive_int(manifest["shard_size"], "shard_size")
        _strict_nonnegative_int(manifest["count"], "count")
        try:
            vector_dtype = np.dtype(manifest["vector_dtype"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Manifest vector_dtype is invalid.") from exc
        if not np.issubdtype(vector_dtype, np.number) or vector_dtype.hasobject:
            raise ValueError("Manifest vector_dtype must be numeric and non-object.")
        shape = _validate_payload_shape(manifest["payload_shape"])
        if (shape is None) != (manifest["payload_dtype"] is None):
            raise ValueError("Manifest payload shape and dtype are inconsistent.")
        if manifest["payload_dtype"] is not None:
            try:
                payload_dtype = np.dtype(manifest["payload_dtype"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Manifest payload_dtype is invalid.") from exc
            if payload_dtype.hasobject:
                raise ValueError("Manifest payload_dtype cannot contain Python objects.")
        if not isinstance(manifest["shards"], list):
            raise ValueError("Manifest shards must be a list.")
        for entry in manifest["shards"]:
            if not isinstance(entry, dict) or set(entry) != _SHARD_KEYS:
                raise ValueError("Manifest contains an invalid shard entry.")
            _strict_positive_int(entry["count"], "shard count")
            if entry["count"] > manifest["shard_size"]:
                raise ValueError("Manifest shard count exceeds shard_size.")
            for key in ("ids", "vectors"):
                if not _safe_filename(entry[key]):
                    raise ValueError(f"Manifest shard {key} filename is invalid.")
            if entry["payload"] is not None and not _safe_filename(entry["payload"]):
                raise ValueError("Manifest shard payload filename is invalid.")
            if (shape is None) != (entry["payload"] is None):
                raise ValueError("Manifest shard payload is inconsistent with the store schema.")
        if expected_stage is not None and manifest["stage"] != expected_stage:
            raise ValueError(f"Store stage mismatch: expected {expected_stage!r}, got {manifest['stage']!r}.")
        if expected_fingerprints is not None and fingerprints != _validate_fingerprints(expected_fingerprints):
            raise ValueError("Store fingerprints do not match the expected fingerprints.")

    @staticmethod
    def _load_shard(root: Path, entry: Mapping[str, Any]) -> RepresentationShard:
        try:
            ids = np.load(root / entry["ids"], mmap_mode="r", allow_pickle=False)
            vectors = np.load(root / entry["vectors"], mmap_mode="r", allow_pickle=False)
            payload = None
            if entry["payload"] is not None:
                payload = np.load(root / entry["payload"], mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid representation shard: {exc}.") from exc
        return RepresentationShard(ids, vectors, payload)

    @staticmethod
    def _validate_shard(
        store: "DirectoryRepresentationStore",
        shard: RepresentationShard,
        entry: Mapping[str, Any],
    ) -> None:
        rows = entry["count"]
        if shard.ids.shape != (rows,) or shard.ids.dtype != np.dtype(np.int64):
            raise ValueError("Representation shard IDs have an invalid shape or dtype.")
        if shard.vectors.shape != (rows, store.vector_dim) or shard.vectors.dtype != store.vector_dtype:
            raise ValueError("Representation shard vectors have an invalid shape or dtype.")
        if store.payload_shape is not None:
            assert shard.payload is not None and store.payload_dtype is not None
            if shard.payload.shape != (rows,) + store.payload_shape or shard.payload.dtype != store.payload_dtype:
                raise ValueError("Representation shard payload has an invalid shape or dtype.")


def _validate_ids(ids: Any, operation: str) -> np.ndarray:
    array = np.asarray(ids)
    unsigned_overflow = (
        array.dtype.kind == "u" and array.size and np.max(array) > np.iinfo(np.int64).max
    )
    if array.dtype.kind not in "iu" or unsigned_overflow:
        raise ValueError(f"{operation} IDs must be integers representable as int64.")
    return array.astype(np.int64, copy=False)


def _validate_payload_shape(shape: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if shape is None:
        return None
    if not isinstance(shape, (list, tuple)):
        raise ValueError("payload_shape must be a sequence of non-negative integers or None.")
    result = tuple(shape)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0 for value in result):
        raise ValueError("payload_shape must contain only non-negative integers.")
    return tuple(int(value) for value in result)


def _validate_stage(stage: Any) -> str:
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string.")
    return stage


def _validate_fingerprints(fingerprints: Any) -> Mapping[str, str]:
    if not isinstance(fingerprints, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(value, str)
        for key, value in fingerprints.items()
    ):
        raise ValueError("fingerprints must map non-empty string keys to string values.")
    return dict(fingerprints)


def _strict_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Manifest {name} must be a strictly positive integer.")


def _strict_nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Manifest {name} must be a non-negative integer.")


def _safe_filename(value: Any) -> bool:
    return isinstance(value, str) and value not in ("", ".", "..") and Path(value).name == value


__all__ = [
    "DirectoryRepresentationStore",
    "MemoryRepresentationStore",
    "RepresentationShard",
    "RepresentationStore",
]
