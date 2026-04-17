# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the BatchSort class with PyTorch backend.
"""
import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.pytorch


@pytest.fixture
def pytorch_backend():
    """Get the PyTorch backend."""
    from deel.influenciae.common import Framework, get_backend
    return get_backend(Framework.PYTORCH)


def test_batched_sorted_dict_pytorch_descending(pytorch_backend):
    """Test BatchSort with PyTorch backend in DESCENDING order."""
    from deel.influenciae.utils.sorted_dict import BatchSort, ORDER

    bsd = BatchSort(batch_shape=(2,), k_shape=(1, 4), dtype=torch.float32,
                    order=ORDER.DESCENDING, backend=pytorch_backend)

    # 1
    v = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float32)
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32)
    bsd.add_all(k, v)

    # 2
    v = torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.float32) * 2
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 10
    bsd.add_all(k, v)

    # 3
    v = torch.tensor([[1, 1, -1, -1, -1]], dtype=torch.float32) * 3
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 100
    bsd.add_all(k, v)

    # 4
    v = torch.tensor([[1, -1, -1, -1, -1]], dtype=torch.float32) * 2.5
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 1000
    bsd.add_all(k, v)

    key, vals = bsd.get()

    values_expected = torch.tensor([[3., 3., 2.5, 2.]])
    key_expected = torch.tensor([[[200, 200], [300, 300], [2000, 2000], [60, 60]]], dtype=torch.float32)

    assert torch.max(torch.abs(key - key_expected)) < 1E-6
    assert torch.max(torch.abs(vals - values_expected)) < 1E-6


def test_batched_sorted_dict_pytorch_ascending(pytorch_backend):
    """Test BatchSort with PyTorch backend in ASCENDING order."""
    from deel.influenciae.utils.sorted_dict import BatchSort, ORDER

    bsd = BatchSort(batch_shape=(2,), k_shape=(1, 4), dtype=torch.float32,
                    order=ORDER.ASCENDING, backend=pytorch_backend)

    # 1
    v = torch.tensor([[-1, -1, -1, -1, -1]], dtype=torch.float32)
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32)
    bsd.add_all(k, v)

    # 2
    v = torch.tensor([[0, 0, 0, 0, -1]], dtype=torch.float32) * 2
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 10
    bsd.add_all(k, v)

    # 3
    v = torch.tensor([[-1, -1, 1, 1, 1]], dtype=torch.float32) * 3
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 100
    bsd.add_all(k, v)

    # 4
    v = torch.tensor([[-1, 1, 1, 1, 1]], dtype=torch.float32) * 2.5
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 1000
    bsd.add_all(k, v)

    key, vals = bsd.get()

    values_expected = torch.tensor([[-3., -3., -2.5, -2.]])
    key_expected = torch.tensor([[[200, 200], [300, 300], [2000, 2000], [60, 60]]], dtype=torch.float32)

    assert torch.max(torch.abs(key - key_expected)) < 1E-6
    assert torch.max(torch.abs(vals - values_expected)) < 1E-6


def test_batched_sorted_dict_pytorch_multiple_rows(pytorch_backend):
    """Test BatchSort with PyTorch backend with multiple rows in k_shape."""
    from deel.influenciae.utils.sorted_dict import BatchSort, ORDER

    bsd = BatchSort(batch_shape=(2,), k_shape=(2, 4), dtype=torch.float32,
                    order=ORDER.DESCENDING, backend=pytorch_backend)

    # 1
    v = torch.tensor([[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]], dtype=torch.float32)
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                      [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32)
    bsd.add_all(k, v)

    # 2
    v = torch.tensor([[-1, 1, -1, -1, -1], [-1, -1, -1, -1, -1]], dtype=torch.float32) * 2
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                      [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 10
    bsd.add_all(k, v)

    # 3
    v = torch.tensor([[-1, -1, 1, 1, 1], [1, 1, 1, -1, -1]], dtype=torch.float32) * 3
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                      [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 100
    bsd.add_all(k, v)

    # 4
    v = torch.tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, 1, -1]], dtype=torch.float32) * 2.5
    k = torch.tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                      [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=torch.float32) * 1000
    bsd.add_all(k, v)

    key, vals = bsd.get()

    values_expected = torch.tensor([[3., 3., 3., 2.], [3., 3., 3.0, 2.5]])
    key_expected = torch.tensor([[[400, 400], [500, 500], [600, 600], [30, 30]],
                                 [[200, 200], [300, 300], [400, 400], [5000, 5000]]], dtype=torch.float32)

    assert torch.max(torch.abs(key - key_expected)) < 1E-6
    assert torch.max(torch.abs(vals - values_expected)) < 1E-6


def test_batched_sorted_dict_pytorch_reset(pytorch_backend):
    """Test BatchSort reset functionality with PyTorch backend."""
    from deel.influenciae.utils.sorted_dict import BatchSort, ORDER

    bsd = BatchSort(batch_shape=(2,), k_shape=(1, 4), dtype=torch.float32,
                    order=ORDER.DESCENDING, backend=pytorch_backend)

    # Add some values
    v = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
    k = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]], dtype=torch.float32)
    bsd.add_all(k, v)

    # Reset
    bsd.reset()

    # Verify reset
    key, vals = bsd.get()

    # After reset, all values should be zeros and -inf
    assert torch.all(key == 0)
    assert torch.all(vals == float('-inf'))


def test_batch_sort_supports_distinct_payload_and_score_dtypes():
    """BatchSort should keep integer payloads and floating-point scores separate."""
    from deel.influenciae.common import get_backend_for_model
    from deel.influenciae.utils.sorted_dict import BatchSort

    backend = get_backend_for_model(nn.Linear(1, 1))
    sorter = BatchSort(
        batch_shape=(),
        k_shape=(1, 2),
        batch_dtype=torch.int64,
        value_dtype=torch.float32,
        backend=backend,
    )
    sorter.add_all(
        torch.tensor([[3, 9]], dtype=torch.int64),
        torch.tensor([[0.4, 0.8]], dtype=torch.float32),
    )
    best_payloads, best_values = sorter.get()

    assert sorter.dtype == torch.float32
    assert sorter.batch_dtype == torch.int64
    assert sorter.value_dtype == torch.float32
    assert best_payloads.dtype == torch.int64
    assert best_values.dtype == torch.float32
    assert best_payloads.shape == (1, 2)
    assert best_values.shape == (1, 2)
