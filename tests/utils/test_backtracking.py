# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.utils.backtracking_line_search import BacktrackingLineSearch, BacktrackingLineSearchPyTorch
from ..utils_test import almost_equal


def test_backtracking_line_search():
    # Define a simple least squares problem y = m * x + b + e, e dist normal(0, 1)
    # This is a convex problem, so the optimizer should converge to the global optimum quite easily
    t = tf.linspace(0., 10., 100)
    m = 0.42
    b = 0.66
    y = m * t + b + tf.random.normal((100,), stddev=0.1)

    # Define the linear problem as Ax=b
    inputs = Input(shape=(1,))
    x = Dense(1, use_bias=True)(inputs)
    model = Model(inputs=inputs, outputs=x)
    optimizer = BacktrackingLineSearch(batches_per_epoch=10, scaling_factor=0.1)
    model.compile(optimizer=optimizer, loss=MeanSquaredError())

    # Optimize using backtracking line-search SGD
    epochs = 100
    train_set = tf.data.Dataset.from_tensor_slices((t, y)).shuffle(100)
    loss_fn = MeanSquaredError()
    for e in range(epochs):
        for t_batch, y_batch in train_set.batch(10):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_weights)
                y_pred = model(t_batch)
                loss = loss_fn(y_batch, y_pred)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.step(model, loss, t_batch, y_batch, grads)

    # Get the estimated m and b
    m_estimated = model.weights[0]
    b_estimated = model.weights[1]
    assert almost_equal(m_estimated, tf.cast(m, tf.float32), epsilon=5e-2)
    assert almost_equal(b_estimated, tf.cast(b, tf.float32), epsilon=5e-2)


def test_backtracking_line_search_pytorch():
    """
    PyTorch equivalent test for BacktrackingLineSearchPyTorch optimizer.
    Tests on a simple least squares problem y = m * x + b + e, e ~ normal(0, 0.1)
    """
    # Define a simple least squares problem
    torch.manual_seed(42)
    t = torch.linspace(0., 10., 100).unsqueeze(1)  # Shape: (100, 1)
    m_true = 0.42
    b_true = 0.66
    y = m_true * t + b_true + torch.randn(100, 1) * 0.1

    # Create dataset and dataloader
    dataset = TensorDataset(t, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Define a simple linear model
    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=True)

        def forward(self, x):
            return self.linear(x)

    model = LinearModel()
    loss_fn = nn.MSELoss()

    # Create the optimizer
    optimizer = BacktrackingLineSearchPyTorch(
        params=model.parameters(),
        batches_per_epoch=10,
        scaling_factor=0.1
    )

    # Training loop
    epochs = 100
    for _ in range(epochs):
        for t_batch, y_batch in dataloader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(t_batch)
            loss = loss_fn(y_pred, y_batch)

            # Backward pass to compute gradients
            loss.backward()

            # Collect gradients
            gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]

            # Define closure for loss re-evaluation
            def closure():
                with torch.no_grad():
                    return loss_fn(model(t_batch), y_batch)

            # Optimizer step with backtracking line search
            optimizer.step(
                model=model,
                current_loss=loss.detach(),
                x_inputs=t_batch,
                labels=y_batch,
                gradients=gradients,
                closure=closure
            )

    # Get the estimated m and b
    m_estimated = model.linear.weight.item()
    b_estimated = model.linear.bias.item()

    # Check that the estimated values are close to the true values
    assert abs(m_estimated - m_true) < 5e-2, f"m_estimated={m_estimated}, m_true={m_true}"
    assert abs(b_estimated - b_true) < 5e-2, f"b_estimated={b_estimated}, b_true={b_true}"


def test_backtracking_line_search_pytorch_state_dict():
    """
    Test that state_dict and load_state_dict work correctly for BacktrackingLineSearchPyTorch.
    """
    # Create a simple parameter
    param = nn.Parameter(torch.randn(3, 3))

    # Create optimizer
    optimizer = BacktrackingLineSearchPyTorch(
        params=[param],
        batches_per_epoch=10,
        scaling_factor=0.2,
        beta=0.8,
        max_eta=5.0,
        min_eta=1e-5
    )

    # Modify eta to test state saving
    optimizer.parameters.eta = 0.5

    # Get state dict
    state = optimizer.state_dict()

    # Verify state dict contents
    assert state['eta'] == 0.5
    assert state['beta'] == 0.8
    assert state['scaling_factor'] == 0.2
    assert state['max_eta'] == 5.0
    assert state['min_eta'] == 1e-5
    assert state['batches_per_epoch'] == 10

    # Create a new optimizer and load state
    optimizer2 = BacktrackingLineSearchPyTorch(
        params=[param],
        batches_per_epoch=5,  # Different initial value
        scaling_factor=0.1   # Different initial value
    )
    optimizer2.load_state_dict(state)

    # Verify state was loaded correctly
    assert optimizer2.parameters.eta == 0.5
    assert optimizer2.parameters.beta == 0.8
    assert optimizer2.scaling_factor == 0.2
    assert optimizer2.parameters.max_eta == 5.0
    assert optimizer2.parameters.min_eta == 1e-5
    assert optimizer2.batches_per_epoch == 10

