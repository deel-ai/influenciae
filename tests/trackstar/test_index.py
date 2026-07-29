# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Tests for the exact streaming TrackStar vector index."""
import numpy as np
import pytest

from deel.influenciae.common.representation_store import (
    DirectoryRepresentationStore,
    MemoryRepresentationStore,
)
from deel.influenciae.trackstar.index import ExactStreamingVectorIndex


pytestmark = pytest.mark.backend_agnostic


@pytest.mark.parametrize("on_disk", [False, True])
@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_exact_search_matches_brute_force(tmp_path, on_disk, order):
    rng = np.random.default_rng(12)
    vectors = rng.normal(size=(37, 7)).astype(np.float32)
    queries = rng.normal(size=(9, 7)).astype(np.float32)
    ids = rng.permutation(np.arange(37, dtype=np.int64) + 200)
    payload = np.stack((ids, -ids), axis=1)
    options = dict(vector_dim=7, shard_size=8, payload_shape=(2,), payload_dtype=np.int64)
    if on_disk:
        store = DirectoryRepresentationStore(tmp_path / "index", **options)
    else:
        store = MemoryRepresentationStore(**options)
    store.append(ids, vectors, payload)
    store.finalize()

    result = ExactStreamingVectorIndex(store, max_score_block_elements=15).search(queries, 6, order=order)
    scores = queries @ vectors.T
    expected_positions = np.empty((len(queries), 6), dtype=np.int64)
    for row in range(len(queries)):
        primary = scores[row] if order == "ascending" else -scores[row]
        expected_positions[row] = np.lexsort((ids, primary))[:6]
    expected_scores = np.take_along_axis(scores, expected_positions, axis=1)
    expected_ids = ids[expected_positions]
    np.testing.assert_allclose(result.scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(result.ids, expected_ids)
    np.testing.assert_array_equal(result.payload, np.stack((expected_ids, -expected_ids), axis=-1))


def test_ties_are_ordered_by_id_across_shards():
    store = MemoryRepresentationStore(2, shard_size=2)
    store.append([40, 10, 30, 20], np.ones((4, 2)))
    store.finalize()
    index = ExactStreamingVectorIndex(store, max_score_block_elements=1)
    for order in ("ascending", "descending"):
        result = index.search([1, 1], 4, order=order)
        np.testing.assert_array_equal(result.ids, [10, 20, 30, 40])


def test_k_and_query_validation():
    store = MemoryRepresentationStore(3)
    store.append([1, 2], np.eye(2, 3))
    store.finalize()
    index = ExactStreamingVectorIndex(store)
    with pytest.raises(ValueError, match="strictly positive"):
        index.search([1, 2, 3], 0)
    with pytest.raises(ValueError, match="exceeds"):
        index.search([1, 2, 3], 3)
    with pytest.raises(ValueError, match="shape"):
        index.search([1, 2], 1)
    with pytest.raises(ValueError, match="order"):
        index.search([1, 2, 3], 1, order="largest")


def test_payload_gather_can_be_skipped(monkeypatch):
    store = MemoryRepresentationStore(2, payload_shape=(), payload_dtype=np.int64)
    store.append([5, 6], [[1, 0], [0, 1]], [50, 60])
    store.finalize()

    def unexpected_gather(ids):
        raise AssertionError(f"payload gathered for {ids}")

    monkeypatch.setattr(store, "gather", unexpected_gather)
    result = ExactStreamingVectorIndex(store).search([1, 0], 1, return_payload=False)
    assert result.payload is None
    np.testing.assert_array_equal(result.ids, [5])
