# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Backend-agnostic tests for the optional FAISS nearest-neighbor backend."""
import numpy as np
import pytest

from deel.influenciae.utils.nearest_neighbors import (
    FaissConfig,
    FaissNearestNeighbors,
    is_faiss_available,
)


pytestmark = pytest.mark.backend_agnostic


faiss_only = pytest.mark.skipif(
    not is_faiss_available(),
    reason="Optional FAISS backend not installed; install with 'pip install influenciae[faiss]'.",
)


def test_faiss_nearest_neighbors_raises_when_faiss_missing():
    """Instantiation without faiss should raise with an install hint."""
    if is_faiss_available():
        pytest.skip("FAISS is installed; this test only runs when the optional dep is missing.")
    with pytest.raises(ImportError, match="faiss"):
        FaissNearestNeighbors()


@faiss_only
def test_faiss_inner_product_top_k_matches_brute_force():
    """FAISS flat index should return the same top-k as a NumPy brute-force search."""
    rng = np.random.default_rng(0)
    gradients = rng.standard_normal((32, 8)).astype(np.float32)
    queries = rng.standard_normal((3, 8)).astype(np.float32)
    payload = np.arange(gradients.shape[0], dtype=np.int64) + 100

    nn = FaissNearestNeighbors(FaissConfig(factory_string="Flat", metric="inner_product"))
    nn.build(dataset=(payload, gradients), k=4)
    scores, indices = nn.query(queries)

    reference_scores = queries @ gradients.T
    reference_top_k = np.argsort(-reference_scores, axis=1)[:, :4]
    reference_payload = payload[reference_top_k]

    np.testing.assert_array_equal(np.sort(indices, axis=1), np.sort(reference_payload, axis=1))
    reference_top_scores = np.take_along_axis(reference_scores, reference_top_k, axis=1)
    np.testing.assert_allclose(np.sort(scores, axis=1), np.sort(reference_top_scores, axis=1), atol=1e-5)


@faiss_only
def test_faiss_l2_metric_returns_descending_order_when_requested():
    """L2 metric should honor descending order override for cross-lib compatibility."""
    from deel.influenciae.utils.sorted_dict import ORDER

    rng = np.random.default_rng(1)
    gradients = rng.standard_normal((10, 4)).astype(np.float32)

    nn = FaissNearestNeighbors(FaissConfig(factory_string="Flat", metric="l2"))
    nn.build(dataset=gradients, k=3, order=ORDER.DESCENDING)
    scores, _ = nn.query(gradients[:2])

    assert scores.shape == (2, 3)
    assert np.all(scores[:, 0] >= scores[:, -1])


@faiss_only
def test_faiss_normalize_queries_yields_cosine_similarity():
    """normalize_queries=True and inner product should yield cosine similarity."""
    rng = np.random.default_rng(2)
    gradients = rng.standard_normal((16, 5)).astype(np.float32)
    query = rng.standard_normal((1, 5)).astype(np.float32)

    nn = FaissNearestNeighbors(
        FaissConfig(factory_string="Flat", metric="inner_product", normalize_queries=True)
    )
    nn.build(dataset=gradients, k=1)
    scores, indices = nn.query(query)

    gradients_normalized = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)
    query_normalized = query / np.linalg.norm(query, axis=1, keepdims=True)
    reference_scores = query_normalized @ gradients_normalized.T
    best_index = int(np.argmax(reference_scores))

    assert int(indices[0, 0]) == best_index
    np.testing.assert_allclose(float(scores[0, 0]), float(reference_scores[0, best_index]), atol=1e-5)


@faiss_only
def test_faiss_query_before_build_raises():
    """Querying before building should raise a clear error."""
    nn = FaissNearestNeighbors()
    with pytest.raises(ValueError, match="not built"):
        nn.query(np.zeros((1, 4), dtype=np.float32))


@faiss_only
def test_faiss_build_rejects_wrong_input_type():
    """Unsupported dataset types should raise TypeError."""
    nn = FaissNearestNeighbors()
    with pytest.raises(TypeError, match="numpy array"):
        nn.build(dataset={"grads": np.zeros((2, 2))})
