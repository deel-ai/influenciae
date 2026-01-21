# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the BatchSort class with PyTorch backend.
"""
import pytest

# Check if PyTorch is available
try:
    import torch
    HAS_PYTORCH = True
except (ImportError, OSError):
    HAS_PYTORCH = False
    torch = None

pytestmark = pytest.mark.skipif(
    not HAS_PYTORCH,
    reason="PyTorch is required for these tests"
)


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
