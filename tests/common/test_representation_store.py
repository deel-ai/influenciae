# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Tests for immutable representation stores."""
import json
import os

import numpy as np
import pytest

from deel.influenciae.common.representation_store import (
    DirectoryRepresentationStore,
    MemoryRepresentationStore,
)


pytestmark = pytest.mark.backend_agnostic


def _contents(store):
    shards = list(store.iter_shards())
    return (
        np.concatenate([shard.ids for shard in shards]),
        np.concatenate([shard.vectors for shard in shards]),
        np.concatenate([shard.payload for shard in shards]),
    )


def _fill(store):
    ids = np.arange(11, dtype=np.int64) + 100
    vectors = np.arange(55, dtype=np.float64).reshape(11, 5)
    payload = np.arange(66, dtype=np.int16).reshape(11, 2, 3)
    store.append(ids[:2], vectors[:2], payload[:2])
    store.append(ids[2:], vectors[2:], payload[2:])
    store.finalize()
    return ids, vectors.astype(np.float32), payload


def test_memory_and_directory_parity_with_large_append(tmp_path):
    options = dict(
        vector_dim=5,
        shard_size=4,
        payload_shape=(2, 3),
        payload_dtype=np.int16,
        stage="projected",
        fingerprints={"model": "abc", "projection": "def"},
    )
    memory = MemoryRepresentationStore(**options)
    expected = _fill(memory)
    directory = DirectoryRepresentationStore(tmp_path / "representations", **options)
    _fill(directory)
    loaded = DirectoryRepresentationStore.load(
        tmp_path / "representations",
        expected_stage="projected",
        expected_fingerprints={"model": "abc", "projection": "def"},
    )

    for actual in (_contents(memory), _contents(loaded)):
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(value, reference)
    assert [len(shard.ids) for shard in loaded.iter_shards()] == [4, 4, 3]
    assert all(isinstance(shard.vectors, np.memmap) for shard in loaded.iter_shards())
    gathered = loaded.gather([[110, 100], [105, 103]])
    np.testing.assert_array_equal(gathered, expected[2][[10, 0, 5, 3]].reshape(2, 2, 2, 3))


def test_payload_is_optional_and_store_is_immutable():
    store = MemoryRepresentationStore(2, shard_size=2)
    store.append([4, 8, 9], np.ones((3, 2)))
    store.finalize()
    assert store.gather([4]) is None
    with pytest.raises(RuntimeError, match="finalized"):
        store.append([10], np.ones((1, 2)))
    with pytest.raises(ValueError, match="payload"):
        MemoryRepresentationStore(2).append([1], [[1, 2]], [3])


def test_duplicate_ids_are_rejected_during_append():
    store = MemoryRepresentationStore(2)
    with pytest.raises(ValueError, match="unique"):
        store.append([1, 1], [[1, 0], [0, 1]])
    store.append([1], [[1, 0]])
    with pytest.raises(ValueError, match="already stored"):
        store.append([1], [[0, 1]])


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda manifest: manifest.update(schema_version=9), "schema"),
        (lambda manifest: manifest.update(stage=3), "stage"),
        (lambda manifest: manifest.update(count=999), "counts total"),
        (lambda manifest: manifest.update(extra=True), "schema"),
        (lambda manifest: manifest.update(fingerprints={"model": 7}), "fingerprints"),
    ],
)
def test_manifest_corruption_is_rejected(tmp_path, mutation, match):
    path = tmp_path / "store"
    store = DirectoryRepresentationStore(path, 2, stage="projected", fingerprints={"model": "x"})
    store.append([1], [[1, 2]])
    store.finalize()
    manifest_path = path / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    mutation(manifest)
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream)
    with pytest.raises(ValueError, match=match):
        DirectoryRepresentationStore.load(path)


def test_shard_shape_and_dtype_corruption_is_rejected(tmp_path):
    path = tmp_path / "store"
    store = DirectoryRepresentationStore(path, 2)
    store.append([1, 2], [[1, 2], [3, 4]])
    store.finalize()
    np.save(path / "ids_000000.npy", np.array([1, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="IDs"):
        DirectoryRepresentationStore.load(path)


def test_failed_overwrite_restores_previous_store(tmp_path, monkeypatch):
    path = tmp_path / "store"
    original = DirectoryRepresentationStore(path, 2)
    original.append([7], [[1, 2]])
    original.finalize()
    replacement = DirectoryRepresentationStore(path, 2, overwrite=True)
    replacement.append([9], [[3, 4]])

    real_replace = os.replace
    calls = 0

    def fail_publication(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_publication)
    with pytest.raises(OSError, match="injected"):
        replacement.finalize()
    loaded = DirectoryRepresentationStore.load(path)
    np.testing.assert_array_equal(next(loaded.iter_shards()).ids, [7])


def test_validation_and_missing_gather_id():
    with pytest.raises(ValueError, match="int64"):
        MemoryRepresentationStore(2).append([1.5], [[1, 2]])
    with pytest.raises(ValueError, match="both"):
        MemoryRepresentationStore(2, payload_shape=(1,))
    store = MemoryRepresentationStore(2, payload_shape=(), payload_dtype=np.int64)
    store.append([3], [[1, 2]], [42])
    store.finalize()
    with pytest.raises(KeyError, match="4"):
        store.gather([4])
