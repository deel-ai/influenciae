# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the model wrappers with PyTorch backend.
These tests mirror the TensorFlow tests in test_model_wrapper.py.
"""
import itertools

import pytest

from deel.influenciae.common import BaseInfluenceModel
from deel.influenciae.common import InfluenceModel
from ..utils_test import almost_equal, assert_tensor_equal

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

def test_loss_reduction():
    """Ensure we can instantiate with proper loss for PyTorch.

    Note: PyTorch loss functions use reduction='none' as string,
    unlike TensorFlow's Reduction.NONE enum.
    """
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 20),  # 5x5 input -> 4x4 after conv
        nn.Linear(20, 10)
    )

    # We should be able to instantiate with loss using reduction='none'
    BaseInfluenceModel(model, loss_function=nn.MSELoss(reduction='none'))
    BaseInfluenceModel(model, loss_function=nn.L1Loss(reduction='none'))
    BaseInfluenceModel(model, loss_function=nn.CrossEntropyLoss(reduction='none'))


def test_loss_calculation():
    """Ensure the wrapper can properly compute the loss for PyTorch."""
    # Create a simple model: f(x) = x^2 + x
    # Using a linear layer with identity weights followed by a custom module
    class SquarePlusX(nn.Module):
        def __init__(self):
            super().__init__()
            # Linear layer that acts as identity (weight=1, bias=0)
            self.linear = nn.Linear(1, 1, bias=False)
            nn.init.ones_(self.linear.weight)

        def forward(self, x):
            x = self.linear(x)
            return x ** 2 + x

    dummy_model = SquarePlusX()

    # MSE(f(1), 1) = MSE(2, 1) = 1
    # MSE(f(0), 1) = MSE(0, 1) = 1
    # MSE(f(2), -1) = MSE(6, -1) = 7 (using MAE: |6 - (-1)| = 7)
    x = torch.tensor([[1.0], [0.0], [2.0]])
    y = torch.tensor([[1.0], [1.0], [-1.0]])

    dataset = DataLoader(TensorDataset(x, y), batch_size=3)

    mae_influence_model = BaseInfluenceModel(
        dummy_model,
        weights_to_watch=list(dummy_model.linear.parameters()),
        loss_function=nn.L1Loss(reduction='none')
    )
    loss_score = mae_influence_model.batch_loss(dataset)

    assert almost_equal(loss_score.squeeze(), [1, 1, 7])

    # Also test with cross entropy (just checking shape)
    # Note: CrossEntropyLoss expects different input format
    dataset2 = DataLoader(TensorDataset(x, y), batch_size=3)

    mse_influence_model = BaseInfluenceModel(
        dummy_model,
        weights_to_watch=list(dummy_model.linear.parameters()),
        loss_function=nn.MSELoss(reduction='none')
    )
    loss_score = mse_influence_model.batch_loss(dataset2)

    assert loss_score.shape == (3, 1)


def test_grad_calculation():
    """Ensure the wrapper can properly compute the gradients for PyTorch."""
    # l(y_pred, y) = (y - y_pred)^2
    # f(x) = (x * W) * 2
    # grad_W(l(f(x), y)) = 8 * x^2 * W - 4 * x * W * y
    # With W=1: grad = 8 * x^2 - 4 * x * y

    class DoubleLin(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=False)
            nn.init.ones_(self.linear.weight)

        def forward(self, x):
            return self.linear(x) * 2

    grad_func = lambda x, y: 8 * x ** 2 - 4 * x * y

    dummy_model = DoubleLin()

    x = torch.tensor([[1.0], [0.0], [0.0]])
    y = torch.tensor([[10.0], [0.0], [0.0]])

    mse_influence_model = BaseInfluenceModel(
        dummy_model,
        loss_function=nn.MSELoss(reduction='none')
    )

    # Using only one batch
    ds1 = DataLoader(TensorDataset(x, y), batch_size=len(x))

    gradients1 = mse_influence_model.batch_gradient(ds1)
    real_gradients1 = torch.sum(torch.stack([grad_func(xi, yi) for xi, yi in zip(x, y)]))

    assert almost_equal(real_gradients1, gradients1, epsilon=1e-4)

    # Using one element per batch
    ds2 = DataLoader(TensorDataset(x, y), batch_size=1)

    gradients2 = mse_influence_model.batch_gradient(ds2)
    real_gradients2 = [grad_func(xi, yi).item() for xi, yi in zip(x, y)]

    assert almost_equal(real_gradients2, gradients2.squeeze(), epsilon=1e-4)


def test_jacobian_calculation():
    """Ensure the wrapper can properly compute the jacobians for PyTorch."""
    # f(x) = (x * W) * 2
    # grad(f, x, y) = 8 * x^2 - 4 * x * y (when W=1)
    grad_func = lambda x, y: 8 * x ** 2 - 4 * x * y

    class DoubleLin(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=False)
            nn.init.ones_(self.linear.weight)

        def forward(self, x):
            return self.linear(x) * 2

    dummy_model = DoubleLin()

    x = torch.tensor([[1.0], [0.0], [0.0]])
    y = torch.tensor([[10.0], [0.0], [0.0]])

    mse_influence_model = BaseInfluenceModel(
        dummy_model,
        loss_function=nn.MSELoss(reduction='none')
    )

    # Multiple batches or one batch should return the same result (one per x)
    ds1 = DataLoader(TensorDataset(x, y), batch_size=len(x))
    ds2 = DataLoader(TensorDataset(x, y), batch_size=1)

    jacobian1 = mse_influence_model.batch_jacobian(ds1)
    jacobian2 = mse_influence_model.batch_jacobian(ds2)

    real_jacobian = torch.stack([grad_func(xi, yi) for xi, yi in zip(x, y)])

    assert almost_equal(real_jacobian.squeeze(), jacobian1.squeeze(), epsilon=1e-4)
    assert almost_equal(real_jacobian.squeeze(), jacobian2.squeeze(), epsilon=1e-4)


def test_weights_default_targeting():
    """Ensure we target the correct theta / weights by default -- last layer with weights."""
    from deel.influenciae.common import InfluenceModel

    # Create a model with multiple layers, ending with Flatten (no weights)
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=2),  # layer 0
        nn.ReLU(),                        # layer 1
        nn.Flatten(),                     # layer 2
        nn.Linear(64, 20),                # layer 3 (5x5 -> 4x4 -> 4*4*4=64)
        nn.Linear(20, 10),                # layer 4
        nn.Flatten()                      # layer 5 (no weights)
    )

    # Warmup (JIT)
    _ = model(torch.randn(1, 1, 5, 5))

    # Should skip the last flatten layer (no weights) and target Linear(20, 10)
    theta = list(model[4].parameters())
    influence_model = InfluenceModel(model)

    for w, theta_w in zip(influence_model.weights, theta):
        assert_tensor_equal(w, theta_w)

    # Model 2: last layer has weights
    model2 = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64, 20),
        nn.Flatten(),
        nn.Linear(20, 10)  # Last layer with weights
    )

    _ = model2(torch.randn(1, 1, 5, 5))

    # Default target layer should be the Linear(20, 10) (last layer with weights)
    theta2 = list(model2[-1].parameters())
    influence_model2 = InfluenceModel(model2)

    for w, theta_w in zip(influence_model2.weights, theta2):
        assert_tensor_equal(w, theta_w)


def test_weights_targeting():
    """Ensure we target the right weights when specifying a layer by name or index."""
    from deel.influenciae.common import InfluenceModel

    # For PyTorch with Sequential, we can use integer indices
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=2),  # layer 0
        nn.Flatten(),                     # layer 1
        nn.Linear(64, 10)                 # layer 2
    )

    _ = model(torch.randn(1, 1, 5, 5))

    # Test targeting by index (first layer - Conv2d)
    theta = list(model[0].parameters())
    influence_model = InfluenceModel(model, start_layer=0)

    for w, theta_w in zip(influence_model.weights, theta):
        assert_tensor_equal(w, theta_w)


def test_targeting_multiple_layers():
    """Assert that everything runs smoothly when we have multiple layers to check on."""
    # Create a Sequential model for layer indexing
    d_test_0 = nn.Linear(2, 2)
    nn.init.ones_(d_test_0.weight)
    nn.init.zeros_(d_test_0.bias)

    d_test_1 = nn.Linear(2, 2, bias=False)

    d_test_2 = nn.Linear(2, 1)
    nn.init.ones_(d_test_2.weight)
    nn.init.ones_(d_test_2.bias)

    model = nn.Sequential(d_test_0, d_test_1, d_test_2)
    _ = model(torch.randn(1, 2))

    layer_0_weights = list(model[0].parameters())
    layer_1_weights = list(model[1].parameters())
    layer_2_weights = list(model[2].parameters())

    ## only start_layer is passed
    # first layer only (by index)
    influence_model = InfluenceModel(model, start_layer=0)
    for w, theta_w in zip(influence_model.weights, layer_0_weights):
        assert_tensor_equal(w, theta_w)

    # second layer only
    influence_model = InfluenceModel(model, start_layer=1)
    for w, theta_w in zip(influence_model.weights, layer_1_weights):
        assert_tensor_equal(w, theta_w)

    # last layer only
    influence_model = InfluenceModel(model, start_layer=2)
    for w, theta_w in zip(influence_model.weights, layer_2_weights):
        assert_tensor_equal(w, theta_w)

    # negative index for last layer
    influence_model = InfluenceModel(model, start_layer=-1)
    for w, theta_w in zip(influence_model.weights, layer_2_weights):
        assert_tensor_equal(w, theta_w)

    ## only last_layer is passed

    # should have last layer only (default start is last layer with weights)
    influence_model = InfluenceModel(model, last_layer=2)
    theoric_weights = layer_2_weights
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)

    influence_model = InfluenceModel(model, last_layer=-1)
    theoric_weights = layer_2_weights
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)

    # should raise an error (start_layer defaults to last layer, so last_layer can't be before it)
    with pytest.raises(AssertionError):
        influence_model = InfluenceModel(model, last_layer=0)
    with pytest.raises(AssertionError):
        influence_model = InfluenceModel(model, last_layer=-3)
    with pytest.raises(AssertionError):
        influence_model = InfluenceModel(model, last_layer=1)

    ## use of both start_layer and last_layer
    influence_model = InfluenceModel(model, start_layer=0, last_layer=1)
    theoric_weights = list(itertools.chain(layer_0_weights, layer_1_weights))
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-1)
    theoric_weights = list(itertools.chain(layer_0_weights, layer_1_weights, layer_2_weights))
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=2)
    theoric_weights = list(itertools.chain(layer_0_weights, layer_1_weights, layer_2_weights))
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)

    influence_model = InfluenceModel(model, start_layer=0, last_layer=-2)
    theoric_weights = list(itertools.chain(layer_0_weights, layer_1_weights))
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)

    influence_model = InfluenceModel(model, start_layer=1, last_layer=2)
    theoric_weights = list(itertools.chain(layer_1_weights, layer_2_weights))
    for w, theta_w in zip(influence_model.weights, theoric_weights):
        assert_tensor_equal(w, theta_w)


def test_forward_pass():
    """Ensure the model wrapper correctly forwards inputs through the model."""
    model = nn.Sequential(
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, 2)
    )

    influence_model = BaseInfluenceModel(model)

    inputs = torch.randn(4, 5)
    outputs = influence_model(inputs)

    assert outputs.shape == (4, 2)
    # Verify it's the same as calling model directly
    expected = model(inputs)
    assert torch.allclose(outputs, expected)


def test_nb_params():
    """Ensure we correctly count the number of parameters."""
    model = nn.Sequential(
        nn.Linear(5, 3),  # 5*3 + 3 = 18 params
        nn.ReLU(),
        nn.Linear(3, 2)   # 3*2 + 2 = 8 params
    )

    influence_model = BaseInfluenceModel(model)

    # Total: 18 + 8 = 26 params
    assert influence_model.nb_params == 26


def test_layers_property():
    """Ensure the layers property returns the model layers."""
    model = nn.Sequential(
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, 2)
    )

    influence_model = BaseInfluenceModel(model)
    layers = influence_model.layers

    # Should include all modules (container + children)
    assert len(layers) > 0
