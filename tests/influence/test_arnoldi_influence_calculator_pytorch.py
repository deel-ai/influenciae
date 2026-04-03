# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for ArnoldiInfluenceCalculator with PyTorch backend.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.influence import ArnoldiInfluenceCalculator, FirstOrderInfluenceCalculator
from deel.influenciae.common import InfluenceModel
from ..utils_test import mse_loss_no_reduction


pytestmark = pytest.mark.pytorch


class SimpleLinearModel(nn.Module):
    """Simple linear model for testing."""
    def __init__(self, input_dim, output_dim, dtype=torch.float64):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


class SimpleCNNModel(nn.Module):
    """Simple CNN model for testing."""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=2, dtype=dtype)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 4, 1, bias=False, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        return self.fc(x)


def test_inverse_exact_hessian_pytorch():
    """Test that the Arnoldi approximation of the inverse Hessian is accurate."""
    torch.manual_seed(0)
    dtype = torch.float64

    model = SimpleLinearModel(10, 1, dtype=dtype)

    # Initialize weights
    nn.init.normal_(model.linear.weight)

    loss_function = mse_loss_no_reduction
    nb_sample = 100
    batch_size = 10

    inputs_train = torch.randn((nb_sample, 10), dtype=dtype)
    targets_train = torch.randn((nb_sample, 1), dtype=dtype)

    # Compute analytical Hessian for MSE loss: H = 2 * X^T X / n
    hessian = 2 * torch.matmul(inputs_train.unsqueeze(2), inputs_train.unsqueeze(1))
    hessian = torch.mean(hessian, dim=0)

    train_dataset = DataLoader(
        TensorDataset(inputs_train, targets_train),
        batch_size=batch_size
    )

    k_largest_eig_vals = 10
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    arnoldi = ArnoldiInfluenceCalculator(
        influence_model, train_dataset, 10,
        False, k_largest_eig_vals, dtype=dtype
    )

    # Reconstruct the inverse Hessian from Arnoldi decomposition
    # H_inv = G^T * diag(1/eig_vals) * G
    H_inv = torch.matmul(
        arnoldi.G.T,
        torch.matmul(torch.diag(1.0 / arnoldi.eig_vals), arnoldi.G)
    )

    # Get real parts if complex
    if H_inv.is_complex():
        H_inv = H_inv.real
    if hessian.is_complex():
        hessian = hessian.real

    hessian_inv = torch.linalg.inv(hessian)
    max_diff = torch.max(torch.abs(H_inv.to(dtype) - hessian_inv)).item()

    assert max_diff < 1E-6, f"Max difference: {max_diff}"


def test_exact_influence_values_pytorch():
    """Test that Arnoldi influence values match first-order influence values."""
    torch.manual_seed(0)
    dtype = torch.float64

    model = SimpleLinearModel(10, 1, dtype=dtype)
    nn.init.normal_(model.linear.weight)

    loss_function = mse_loss_no_reduction
    nb_sample = 100
    batch_size = 10

    inputs_train = torch.randn((nb_sample, 10), dtype=dtype)
    targets_train = torch.randn((nb_sample, 1), dtype=dtype)
    train_dataset = DataLoader(
        TensorDataset(inputs_train, targets_train),
        batch_size=batch_size
    )

    k_largest_eig_vals = 10
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    arnoldi_calculator = ArnoldiInfluenceCalculator(
        influence_model, train_dataset, 10,
        False, k_largest_eig_vals, dtype=dtype
    )

    arnoldi_influence_values = arnoldi_calculator._compute_influence_values(train_dataset)

    first_order_calculator = FirstOrderInfluenceCalculator(influence_model, train_dataset)
    first_order_influence_values = first_order_calculator._compute_influence_values(train_dataset)

    # Get real parts if complex
    if arnoldi_influence_values.is_complex():
        arnoldi_influence_values = arnoldi_influence_values.real
    if first_order_influence_values.is_complex():
        first_order_influence_values = first_order_influence_values.real

    max_diff = torch.max(torch.abs(arnoldi_influence_values - first_order_influence_values)).item()
    assert max_diff < 1E-5, f"Max difference: {max_diff}"


def test_hermitian_mode_pytorch():
    """Test Arnoldi calculator with force_hermitian=True."""
    torch.manual_seed(0)
    dtype = torch.float64

    model = SimpleLinearModel(10, 1, dtype=dtype)
    nn.init.normal_(model.linear.weight)

    loss_function = mse_loss_no_reduction
    nb_sample = 50
    batch_size = 10

    inputs_train = torch.randn((nb_sample, 10), dtype=dtype)
    targets_train = torch.randn((nb_sample, 1), dtype=dtype)
    train_dataset = DataLoader(
        TensorDataset(inputs_train, targets_train),
        batch_size=batch_size
    )

    k_largest_eig_vals = 8
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)

    # Test with force_hermitian=True
    arnoldi_hermitian = ArnoldiInfluenceCalculator(
        influence_model, train_dataset, 10,
        True, k_largest_eig_vals, dtype=dtype
    )

    # Compute influence values - should not raise
    influence_values = arnoldi_hermitian._compute_influence_values(train_dataset)

    # Ensure eigenvalues are real when force_hermitian is True
    assert not arnoldi_hermitian.eig_vals.is_complex(), "Eigenvalues should be real with force_hermitian=True"
    assert influence_values is not None


def test_compute_influence_vector_pytorch():
    """Test _compute_influence_vector method."""
    torch.manual_seed(0)
    dtype = torch.float64

    model = SimpleLinearModel(5, 1, dtype=dtype)
    nn.init.normal_(model.linear.weight)

    loss_function = mse_loss_no_reduction
    nb_sample = 20
    batch_size = 5

    inputs_train = torch.randn((nb_sample, 5), dtype=dtype)
    targets_train = torch.randn((nb_sample, 1), dtype=dtype)
    train_dataset = DataLoader(
        TensorDataset(inputs_train, targets_train),
        batch_size=batch_size
    )

    k_largest_eig_vals = 5
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    arnoldi = ArnoldiInfluenceCalculator(
        influence_model, train_dataset, 5,
        False, k_largest_eig_vals, dtype=dtype
    )

    # Get a batch and compute influence vector
    for batch in train_dataset:
        influence_vector = arnoldi._compute_influence_vector(batch)
        assert influence_vector.shape[0] == batch[0].shape[0]
        assert influence_vector.shape[1] == k_largest_eig_vals
        break


def test_preprocess_samples_pytorch():
    """Test _preprocess_samples method."""
    torch.manual_seed(0)
    dtype = torch.float64

    model = SimpleLinearModel(5, 1, dtype=dtype)
    nn.init.normal_(model.linear.weight)

    loss_function = mse_loss_no_reduction
    nb_sample = 20
    batch_size = 5

    inputs_train = torch.randn((nb_sample, 5), dtype=dtype)
    targets_train = torch.randn((nb_sample, 1), dtype=dtype)
    train_dataset = DataLoader(
        TensorDataset(inputs_train, targets_train),
        batch_size=batch_size
    )

    k_largest_eig_vals = 5
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    arnoldi = ArnoldiInfluenceCalculator(
        influence_model, train_dataset, 5,
        False, k_largest_eig_vals, dtype=dtype
    )

    # Get a batch and preprocess
    for batch in train_dataset:
        preprocessed = arnoldi._preprocess_samples(batch)
        assert preprocessed.shape[0] == batch[0].shape[0]
        assert preprocessed.shape[1] == k_largest_eig_vals
        break
