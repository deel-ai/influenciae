# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the TracIn method with PyTorch backend.
"""
from functools import partial

import pytest
import numpy as np
from ..utils_test import mse_loss_no_reduction, relative_almost_equal, set_seed_torch_numpy

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except (ImportError, OSError):
    HAS_PYTORCH = False
    torch = None
    nn = None

pytestmark = pytest.mark.skipif(
    not HAS_PYTORCH,
    reason="PyTorch is required for these tests"
)


_seed_all = partial(set_seed_torch_numpy, include_cuda=True)


class FeatureModel(nn.Module):
    """
    Feature extractor: Conv2d -> ReLU -> Flatten.
    Input shape: (batch, 3, 5, 5)
    Output shape: (batch, 64) since Conv2d(3, 4, kernel_size=2) on 5x5 -> 4x4 -> 4*4*4=64
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=2, dtype=torch.float64)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.flatten(x)
        return x


class FullModel(nn.Module):
    """
    Full model combining feature extractor and linear head.
    Uses ones initializer for the linear layer (no bias) to match TF tests.
    """
    def __init__(self, feature_model):
        super().__init__()
        self.features = feature_model
        # Linear layer with ones initialization, no bias
        self.linear = nn.Linear(64, 1, bias=False, dtype=torch.float64)
        nn.init.ones_(self.linear.weight)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x


def test_compute_influence_vector():
    """Test that _compute_influence_vector returns correct gradients scaled by sqrt(lr)."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(42)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    # Warm up the model
    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = torch.randn((10, 3, 5, 5), dtype=torch.float64)
    targets_train = torch.randn((10, 1), dtype=torch.float64)
    train_dataset = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)

    # Compute expected influence vectors manually
    with torch.no_grad():
        f_train = feature_model(inputs_train)  # (10, 64)
        pred = torch.sum(f_train, dim=1, keepdim=True)  # model predicts sum since weights are ones
        # Gradient of MSE: d/dw (pred - target)^2 = 2 * (pred - target) * d(pred)/dw
        # d(pred)/dw_i = f_train_i (since pred = sum(f_train * w) and w=1)
        g_train = 2 * (pred - targets_train) * f_train  # (10, 64)

        expected_inf_vect = torch.cat([
            g_train * np.sqrt(lr),
            g_train * np.sqrt(2 * lr)
        ], dim=1)

    inf_vect = []
    for batch in train_loader:
        batched_inf_vec = tracin._compute_influence_vector(batch)
        assert batched_inf_vec.shape == (5, 2 * if_model.nb_params)  # (batch_size, nb_model * nb_params)
        inf_vect.append(batched_inf_vec)
    inf_vect = torch.cat(inf_vect, dim=0)

    assert relative_almost_equal(expected_inf_vect, inf_vect, percent=1e-6)


def test_compute_influence_value_from_influence_vector():
    """Test that influence values are computed correctly from influence vectors."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(43)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    # Warm up the model
    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = torch.randn((10, 3, 5, 5), dtype=torch.float64)
    inputs_test = torch.randn((50, 3, 5, 5), dtype=torch.float64)
    targets_train = torch.randn((10, 1), dtype=torch.float64)
    targets_test = torch.randn((50, 1), dtype=torch.float64)

    train_dataset = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    test_dataset = TensorDataset(inputs_test, targets_test)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Compute expected values manually
    expected_values = []

    with torch.no_grad():
        f_train = feature_model(inputs_train)
        pred_train = torch.sum(f_train, dim=1, keepdim=True)
        g_train = 2 * (pred_train - targets_train) * f_train

        train_inf_vect = torch.cat([
            g_train * np.sqrt(lr),
            g_train * np.sqrt(2 * lr)
        ], dim=1)

        for batch_inputs_test, batch_targets_test in test_loader:
            f_test = feature_model(batch_inputs_test)
            pred_test = torch.sum(f_test, dim=1, keepdim=True)
            g_test = 2 * (pred_test - batch_targets_test) * f_test
            test_inf_vect = torch.cat([
                g_test * np.sqrt(lr),
                g_test * np.sqrt(2 * lr)
            ], dim=1)
            v = torch.matmul(test_inf_vect, train_inf_vect.T)
            expected_values.append(v)

        expected_values = torch.cat(expected_values, dim=0)

    # Compute using TracIn methods
    inf_vect = []
    for train_batch in train_loader:
        batch_inf_vec = tracin._compute_influence_vector(train_batch)
        inf_vect.append(batch_inf_vec)
    inf_vect = torch.cat(inf_vect, dim=0)

    computed_values = []
    for test_batch in test_loader:
        preproc_test_batch = tracin._preprocess_samples(test_batch)
        inf_values = tracin._estimate_influence_value_from_influence_vector(preproc_test_batch, inf_vect)
        computed_values.append(inf_values)
    computed_values = torch.cat(computed_values, dim=0)

    assert computed_values.shape == (50, 10)
    assert relative_almost_equal(expected_values, computed_values, percent=1e-6)


def test_compute_pairwise_influence_value():
    """Test that pairwise (self) influence values are computed correctly."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(44)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    # Warm up the model
    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = torch.randn((10, 3, 5, 5), dtype=torch.float64)
    targets_train = torch.randn((10, 1), dtype=torch.float64)
    train_dataset = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)

    # Compute expected pairwise influence values manually
    with torch.no_grad():
        f_train = feature_model(inputs_train)
        pred_train = torch.sum(f_train, dim=1, keepdim=True)
        g_train = 2 * (pred_train - targets_train) * f_train

        expected_inf_vect = torch.cat([
            g_train * np.sqrt(lr),
            g_train * np.sqrt(2 * lr)
        ], dim=1)
        expected_pairwise_inf_vect = torch.sum(expected_inf_vect * expected_inf_vect, dim=1, keepdim=True)

    pairwise_inf = []
    for batch in train_loader:
        loc_pairwise_inf = tracin._compute_influence_value_from_batch(batch)
        assert loc_pairwise_inf.shape == (5, 1)
        pairwise_inf.append(loc_pairwise_inf)
    pairwise_inf = torch.cat(pairwise_inf, dim=0)

    assert pairwise_inf.shape == (10, 1)
    assert relative_almost_equal(expected_pairwise_inf_vect, pairwise_inf, percent=1e-6)


def test_estimate_individual_influence_values_from_batch():
    """Test the _estimate_individual_influence_values_from_batch method."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(45)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    # Warm up the model
    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = torch.randn((5, 3, 5, 5), dtype=torch.float64)
    inputs_test = torch.randn((10, 3, 5, 5), dtype=torch.float64)
    targets_train = torch.randn((5, 1), dtype=torch.float64)
    targets_test = torch.randn((10, 1), dtype=torch.float64)

    train_batch = (inputs_train, targets_train)
    test_batch = (inputs_test, targets_test)

    influence_values = tracin._estimate_individual_influence_values_from_batch(
        train_samples=train_batch,
        samples_to_evaluate=test_batch
    )

    # Shape should be (test_batch_size, train_batch_size)
    assert influence_values.shape == (10, 5)


def test_multiple_learning_rates():
    """Test TracIn with different learning rates for each checkpoint."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(46)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    # Warm up the model
    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)

    # Test with list of learning rates
    lrs = [1.0, 0.1, 0.01]
    tracin = TracIn([if_model, if_model, if_model], lrs)
    assert tracin.learning_rates == lrs

    # Test with single learning rate
    single_lr = 0.001
    tracin_single = TracIn([if_model, if_model], single_lr)
    assert tracin_single.learning_rates == [single_lr, single_lr]


def test_tracin_with_single_model():
    """Test TracIn with a single checkpoint model."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(47)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    # Warm up the model
    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    lr = 0.01
    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)
    tracin = TracIn([if_model], lr)

    inputs_train = torch.randn((5, 3, 5, 5), dtype=torch.float64)
    targets_train = torch.randn((5, 1), dtype=torch.float64)
    train_batch = (inputs_train, targets_train)

    # Should work with single model
    influence_vector = tracin._compute_influence_vector(train_batch)
    assert influence_vector.shape == (5, if_model.nb_params)


def test_empty_models_raises_error():
    """Test that TracIn raises an error when no models are provided."""
    from deel.influenciae.trac_in.tracin import TracIn

    with pytest.raises(ValueError, match="At least one model must be provided"):
        TracIn([], 0.01)


def test_learning_rates_length_mismatch():
    """Test that TracIn raises an error when learning_rates list doesn't match models length."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(48)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)

    with pytest.raises(AssertionError):
        TracIn([if_model, if_model], [0.01])  # 2 models but only 1 learning rate


def test_compute_influence_values():
    """Test the _compute_influence_values method on a dataset."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(49)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    # Warm up the model
    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    lr = 3.0
    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)
    tracin = TracIn([if_model, if_model], [lr, 2 * lr])

    inputs_train = torch.randn((10, 3, 5, 5), dtype=torch.float64)
    targets_train = torch.randn((10, 1), dtype=torch.float64)
    train_dataset = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)

    # Compute influence values
    influence_values = tracin._compute_influence_values(train_loader)

    assert influence_values.shape == (10, 1)
    # All self-influence values should be positive (sum of squared gradients)
    assert torch.all(influence_values >= 0)


def test_compute_influence_vector_batched_consistency():
    """Test that computing influence vectors in batches gives consistent results."""
    from deel.influenciae.common import InfluenceModel
    from deel.influenciae.trac_in.tracin import TracIn

    _seed_all(50)

    feature_model = FeatureModel()
    model = FullModel(feature_model)

    _ = model(torch.randn((50, 3, 5, 5), dtype=torch.float64))

    lr = 0.01
    if_model = InfluenceModel(model, start_layer=-1, loss_function=mse_loss_no_reduction)
    tracin = TracIn([if_model], lr)

    inputs_train = torch.randn((20, 3, 5, 5), dtype=torch.float64)
    targets_train = torch.randn((20, 1), dtype=torch.float64)

    # Compute with batch size 5
    train_dataset_5 = TensorDataset(inputs_train, targets_train)
    train_loader_5 = DataLoader(train_dataset_5, batch_size=5, shuffle=False)

    inf_vect_5 = []
    for batch in train_loader_5:
        inf_vect_5.append(tracin._compute_influence_vector(batch))
    inf_vect_5 = torch.cat(inf_vect_5, dim=0)

    # Compute with batch size 10
    train_loader_10 = DataLoader(train_dataset_5, batch_size=10, shuffle=False)

    inf_vect_10 = []
    for batch in train_loader_10:
        inf_vect_10.append(tracin._compute_influence_vector(batch))
    inf_vect_10 = torch.cat(inf_vect_10, dim=0)

    # Results should be the same regardless of batch size
    assert relative_almost_equal(inf_vect_5, inf_vect_10, percent=1e-6)
