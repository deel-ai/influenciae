# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for LinearNearestNeighbors with PyTorch backend.
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


def test_linear_nearest_pytorch(pytorch_backend):
    """Test LinearNearestNeighbors with PyTorch backend."""
    from deel.influenciae.common import Framework
    from deel.influenciae.utils.nearest_neighbors import LinearNearestNeighbors
    from deel.influenciae.utils.sorted_dict import ORDER

    linear_nearest = LinearNearestNeighbors(backend=Framework.PYTORCH)

    def dot_product_fun(x1, x2):
        influence_values = torch.matmul(x1, x2.T)
        return influence_values

    x = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=torch.float32) * 10
    y = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    v = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=torch.float32)

    # Create a list of batched tuples to simulate PyTorch DataLoader behavior
    batch_size = 2
    dataset = []
    for i in range(0, len(x), batch_size):
        batch_x = x[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        batch_v = v[i:i + batch_size]
        dataset.append(((batch_x, batch_y), batch_v))

    linear_nearest.build(
        dataset=dataset,
        dot_product_fun=dot_product_fun,
        k=4,
        query_batch_size=3,
        d_type=torch.float32,
        order=ORDER.DESCENDING
    )

    vector_to_find = torch.tensor([[1, 1], [-0.5, 1], [0, -1]], dtype=torch.float32)
    influences_values, training_samples = linear_nearest.query(vector_to_find=vector_to_find)

    influences_values_expected = torch.tensor(
        [[12, 10, 8, 6], [3., 2.5, 2., 1.5], [-1, -2, -3, -4]],
        dtype=torch.float32
    )
    training_samples_expected = torch.tensor(
        [
            [[60, 60], [50, 50], [40, 40], [30, 30]],
            [[60, 60], [50, 50], [40, 40], [30, 30]],
            [[10, 10], [20, 20], [30, 30], [40, 40]]
        ],
        dtype=torch.float32
    )

    assert torch.max(torch.abs(influences_values - influences_values_expected)) < 1E-6
    assert torch.max(torch.abs(training_samples - training_samples_expected)) < 1E-6


def test_linear_nearest_pytorch_ascending(pytorch_backend):
    """Test LinearNearestNeighbors with PyTorch backend in ASCENDING order."""
    from deel.influenciae.common import Framework
    from deel.influenciae.utils.nearest_neighbors import LinearNearestNeighbors
    from deel.influenciae.utils.sorted_dict import ORDER

    linear_nearest = LinearNearestNeighbors(backend=Framework.PYTORCH)

    def dot_product_fun(x1, x2):
        influence_values = torch.matmul(x1, x2.T)
        return influence_values

    x = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=torch.float32) * 10
    y = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    v = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=torch.float32)

    # Create a list of batched tuples to simulate PyTorch DataLoader behavior
    batch_size = 2
    dataset = []
    for i in range(0, len(x), batch_size):
        batch_x = x[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        batch_v = v[i:i + batch_size]
        dataset.append(((batch_x, batch_y), batch_v))

    linear_nearest.build(
        dataset=dataset,
        dot_product_fun=dot_product_fun,
        k=4,
        query_batch_size=3,
        d_type=torch.float32,
        order=ORDER.ASCENDING
    )

    vector_to_find = torch.tensor([[1, 1], [-0.5, 1], [0, -1]], dtype=torch.float32)
    influences_values, training_samples = linear_nearest.query(vector_to_find=vector_to_find)

    # For ascending order, we expect the smallest values first
    influences_values_expected = torch.tensor(
        [[2, 4, 6, 8], [0.5, 1., 1.5, 2.], [-6, -5, -4, -3]],
        dtype=torch.float32
    )
    training_samples_expected = torch.tensor(
        [
            [[10, 10], [20, 20], [30, 30], [40, 40]],
            [[10, 10], [20, 20], [30, 30], [40, 40]],
            [[60, 60], [50, 50], [40, 40], [30, 30]]
        ],
        dtype=torch.float32
    )

    assert torch.max(torch.abs(influences_values - influences_values_expected)) < 1E-6
    assert torch.max(torch.abs(training_samples - training_samples_expected)) < 1E-6


def test_linear_nearest_pytorch_one_pass_iterator(pytorch_backend):
    """Test one-pass iterators are materialized in build()."""
    from deel.influenciae.common import Framework
    from deel.influenciae.utils.nearest_neighbors import LinearNearestNeighbors
    from deel.influenciae.utils.sorted_dict import ORDER

    linear_nearest = LinearNearestNeighbors(backend=Framework.PYTORCH)

    def dot_product_fun(x1, x2):
        return torch.matmul(x1, x2.T)

    x = torch.tensor([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=torch.float32)
    y = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    v = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float32)

    def dataset_iter():
        batch_size = 2
        for idx in range(0, len(x), batch_size):
            batch_x = x[idx:idx + batch_size]
            batch_y = y[idx:idx + batch_size]
            batch_v = v[idx:idx + batch_size]
            yield ((batch_x, batch_y), batch_v)

    with pytest.warns(RuntimeWarning, match="one-pass iterator"):
        linear_nearest.build(
            dataset=dataset_iter(),
            dot_product_fun=dot_product_fun,
            k=2,
            query_batch_size=1,
            d_type=torch.float32,
            order=ORDER.DESCENDING,
        )

    query = torch.tensor([[1, 1]], dtype=torch.float32)
    values_first, samples_first = linear_nearest.query(vector_to_find=query)
    values_second, samples_second = linear_nearest.query(vector_to_find=query)

    assert torch.max(torch.abs(values_first - values_second)) < 1E-6
    assert torch.max(torch.abs(samples_first - samples_second)) < 1E-6
