# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Tests for SecondOrderInfluenceCalculator with PyTorch backend."""

from functools import partial

import pytest

try:
    import torch
    import torch.nn as nn

    HAS_PYTORCH = True
except (ImportError, OSError):
    HAS_PYTORCH = False
    torch = None
    nn = None

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ConjugateGradientDescentIHVP
from deel.influenciae.influence.second_order_influence_calculator import SecondOrderInfluenceCalculator
from ..utils_test import (
    assert_close,
    build_loader_torch,
    build_regression_tensors_torch,
    ground_truth_grads_hessian_last_layer_torch,
    make_linear_model_torch,
    set_seed_torch,
)


pytestmark = [
    pytest.mark.pytorch,
    pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch is required for these tests"),
]


set_seed = set_seed_torch
make_linear_model = make_linear_model_torch
build_regression_tensors = build_regression_tensors_torch
build_loader = build_loader_torch
grads_hessians_last_layer = partial(
    ground_truth_grads_hessian_last_layer_torch,
    return_hessian_stack=True,
)


def second_order_ground_truth(inv_hessian, grads_group, hessians_group, train_size):
    """Compute analytical second-order additive/pairwise and final influence vector."""
    group_size = grads_group.shape[1]
    reduced_grads = torch.sum(grads_group, dim=1, keepdim=True)
    additive_raw = inv_hessian @ reduced_grads

    hihvp = torch.matmul(hessians_group, additive_raw)
    pairwise_raw = inv_hessian @ torch.sum(hihvp, dim=0)

    fraction = float(group_size) / float(train_size)
    coeff_add = (1.0 - 2.0 * fraction) / (((1.0 - fraction) ** 2) * float(train_size))
    coeff_pair = 1.0 / ((((1.0 - fraction) ** 2) * float(train_size) * float(train_size)))

    influence_group = (coeff_add * additive_raw + coeff_pair * pairwise_raw).T
    return additive_raw, pairwise_raw, influence_group


def build_calculators(influence_model, train_loader):
    """Build IHVP calculators with associated tolerances."""
    return [
        (ExactIHVP(influence_model, train_loader), 1e-3),
        (ConjugateGradientDescentIHVP(influence_model, -1, train_loader, n_opt_iters=80), 2e-1),
    ]


def test__compute_additive_term():
    """Test _compute_additive_term against analytical ground truth."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=1)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    group_loader = build_loader(inputs_train[:5], targets_train[:5], batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    grads_train, hess_train = grads_hessians_last_layer(model, inputs_train, targets_train)
    inv_hessian = torch.linalg.pinv(torch.mean(hess_train, dim=0))
    additive_gt, _, _ = second_order_ground_truth(inv_hessian, grads_train[:, :5], hess_train[:5], train_size=25)

    for ihvp_calculator, tolerance in build_calculators(influence_model, train_loader):
        calculator = SecondOrderInfluenceCalculator(
            influence_model,
            train_loader,
            ihvp_calculator,
            n_samples_for_hessian=25,
            shuffle_buffer_size=25,
        )
        additive_term = calculator._compute_additive_term(group_loader)
        assert additive_term.shape == additive_gt.shape
        assert_close(additive_term, additive_gt, epsilon=tolerance)


def test__compute_pairwise_interactions():
    """Test _compute_pairwise_interactions against analytical ground truth."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=2)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    group_loader = build_loader(inputs_train[:5], targets_train[:5], batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    grads_train, hess_train = grads_hessians_last_layer(model, inputs_train, targets_train)
    inv_hessian = torch.linalg.pinv(torch.mean(hess_train, dim=0))
    _, pairwise_gt, _ = second_order_ground_truth(inv_hessian, grads_train[:, :5], hess_train[:5], train_size=25)

    for ihvp_calculator, tolerance in build_calculators(influence_model, train_loader):
        calculator = SecondOrderInfluenceCalculator(
            influence_model,
            train_loader,
            ihvp_calculator,
            n_samples_for_hessian=25,
            shuffle_buffer_size=25,
        )
        pairwise_term = calculator._compute_pairwise_interactions(group_loader)
        assert pairwise_term.shape == pairwise_gt.shape
        assert_close(pairwise_term, pairwise_gt, epsilon=tolerance)


def test_compute_influence_group():
    """Test compute_influence_vector_group against analytical second-order vector."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=3)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    group_loader = build_loader(inputs_train[:5], targets_train[:5], batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    grads_train, hess_train = grads_hessians_last_layer(model, inputs_train, targets_train)
    inv_hessian = torch.linalg.pinv(torch.mean(hess_train, dim=0))
    _, _, influence_gt = second_order_ground_truth(inv_hessian, grads_train[:, :5], hess_train[:5], train_size=25)

    for ihvp_calculator, tolerance in build_calculators(influence_model, train_loader):
        calculator = SecondOrderInfluenceCalculator(
            influence_model,
            train_loader,
            ihvp_calculator,
            n_samples_for_hessian=25,
            shuffle_buffer_size=25,
        )
        influence_group = calculator.compute_influence_vector_group(group_loader)
        assert influence_group.shape == (1, 2)
        assert_close(influence_group, influence_gt, epsilon=tolerance)


def test_compute_influence_values_group():
    """Test estimate_influence_values_group for explicit and self influence."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=4)
    inputs_test, targets_test = build_regression_tensors(25, seed=5)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=5)

    group_train_loader = build_loader(inputs_train[:5], targets_train[:5], batch_size=5)
    group_test_loader = build_loader(inputs_test[:5], targets_test[:5], batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    grads_train, hess_train = grads_hessians_last_layer(model, inputs_train, targets_train)
    grads_test, _ = grads_hessians_last_layer(model, inputs_test, targets_test)

    inv_hessian = torch.linalg.pinv(torch.mean(hess_train, dim=0))
    _, _, influence_gt = second_order_ground_truth(inv_hessian, grads_train[:, :5], hess_train[:5], train_size=25)

    reduced_train = torch.sum(grads_train[:, :5], dim=1, keepdim=True)
    reduced_test = torch.sum(grads_test[:, :5], dim=1, keepdim=True)
    influence_values_gt = reduced_test.T @ influence_gt.T
    self_influence_gt = reduced_train.T @ influence_gt.T

    for ihvp_calculator, tolerance in build_calculators(influence_model, train_loader):
        calculator = SecondOrderInfluenceCalculator(
            influence_model,
            train_loader,
            ihvp_calculator,
            n_samples_for_hessian=25,
            shuffle_buffer_size=25,
        )
        influence_values = calculator.estimate_influence_values_group(group_train_loader, group_test_loader)
        assert influence_values.shape == (1, 1)
        assert_close(influence_values, influence_values_gt, epsilon=tolerance)

        self_influence = calculator.estimate_influence_values_group(group_train_loader)
        assert self_influence.shape == (1, 1)
        assert_close(self_influence, self_influence_gt, epsilon=tolerance)


def test_compute_influence_values_group_raises_on_mismatched_group_size():
    """Test estimate_influence_values_group raises when group sizes mismatch."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=6)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)

    group_train = build_loader(inputs_train[:5], targets_train[:5], batch_size=5)
    group_eval = build_loader(inputs_train[:4], targets_train[:4], batch_size=4)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    calculator = SecondOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    with pytest.raises(ValueError, match="must match"):
        calculator.estimate_influence_values_group(group_train, group_eval)


@pytest.mark.parametrize("ihvp_name", ["exact", "conjugate_gradient"])
def test_cnn_shapes(ihvp_name):
    """Shape checks on a more challenging CNN model."""
    set_seed(0)

    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=2, dtype=torch.float64),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 10, bias=True, dtype=torch.float64),
        nn.Linear(10, 10, bias=True, dtype=torch.float64),
    )

    def mse_per_sample(predictions, targets):
        return torch.mean((predictions - targets) ** 2, dim=1)

    x_train = torch.randn((10, 3, 5, 5), dtype=torch.float64)
    y_train_idx = torch.randint(0, 10, (10,))
    y_train = torch.nn.functional.one_hot(y_train_idx, num_classes=10).to(torch.float64)

    x_test = torch.randn((10, 3, 5, 5), dtype=torch.float64)
    y_test_idx = torch.randint(0, 10, (10,))
    y_test = torch.nn.functional.one_hot(y_test_idx, num_classes=10).to(torch.float64)

    train_loader = build_loader(x_train, y_train, batch_size=5)
    influence_model = InfluenceModel(model, loss_function=mse_per_sample)

    if ihvp_name == "exact":
        ihvp_calculator = ExactIHVP(influence_model, train_loader)
    else:
        ihvp_calculator = ConjugateGradientDescentIHVP(
            influence_model,
            -2,
            train_loader,
            n_opt_iters=50,
        )

    calculator = SecondOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ihvp_calculator,
        n_samples_for_hessian=10,
        shuffle_buffer_size=10,
    )

    group_train_loader = build_loader(x_train[:5], y_train[:5], batch_size=5)
    group_test_loader = build_loader(x_test[:5], y_test[:5], batch_size=5)

    influence = calculator.compute_influence_vector_group(group_train_loader)
    assert influence.shape == (1, 110)

    influence_values = calculator.estimate_influence_values_group(group_train_loader, group_test_loader)
    assert influence_values.shape == (1, 1)
