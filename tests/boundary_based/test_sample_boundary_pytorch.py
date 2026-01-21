# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch-specific tests for SampleBoundaryCalculator.
"""
import pytest

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

from deel.influenciae.boundary_based import SampleBoundaryCalculator


def test_compute_influence_shape_pytorch():
    """Test that influence scores have the correct shape with PyTorch model."""
    # Create a simple PyTorch model
    model = nn.Sequential(
        nn.Linear(3, 2)
    )
    # Initialize weights similar to TF test
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
        model[0].bias.copy_(torch.tensor([4.0, 0.0]))

    calculator = SampleBoundaryCalculator(model)

    inputs_train = torch.randn(10, 3)
    targets_train = torch.zeros(10, dtype=torch.long)
    targets_one_hot = torch.nn.functional.one_hot(targets_train, num_classes=2).float()

    dataset = TensorDataset(inputs_train, targets_one_hot)
    train_loader = DataLoader(dataset, batch_size=5)

    influence_computed_score = calculator._compute_influence_values(train_loader)

    assert influence_computed_score.shape == (10, 1)


def test_compute_influence_values_pytorch():
    """Test that influence values are computed correctly with PyTorch model."""
    # Create a simple PyTorch model
    model = nn.Sequential(
        nn.Linear(3, 2)
    )
    # Initialize weights similar to TF test
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 1.0, 0.0],  # w0
                                            [1.0, 0.0, 0.0]]))  # w1
        model[0].bias.copy_(torch.tensor([4.0, 0.0]))

    calculator = SampleBoundaryCalculator(model)

    inputs_train = torch.zeros(1, 3)
    targets_train = torch.zeros(1, dtype=torch.long)
    targets_one_hot = torch.nn.functional.one_hot(targets_train, num_classes=2).float()

    dataset = TensorDataset(inputs_train, targets_one_hot)
    train_loader = DataLoader(dataset, batch_size=1)

    influence_computed_score = calculator._compute_influence_values(train_loader)

    # Check that the influence score is negative (indicating distance to boundary)
    # and has reasonable magnitude. Due to algorithmic differences between
    # TensorFlow and PyTorch implementations, exact values may differ slightly.
    assert influence_computed_score.shape == (1, 1)
    assert influence_computed_score[0, 0].item() == -4
