# tests/test_ihvp_pytorch.py
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils_test import (
    build_regression_tensors_torch,
    ground_truth_grads_hessian_last_layer_torch,
    make_linear_model_torch,
    max_abs_almost_equal as almost_equal,
)

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP


make_linear_model = partial(make_linear_model_torch, dtype=torch.float32)
make_dataset = partial(build_regression_tensors_torch, dtype=torch.float32)


def _per_sample_pred_scalar(W1: torch.Tensor, W2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """
    W1: (2, 3), W2: (1, 2), x3: (3,)
    pred = W2 * (W1 * x)
    """
    z = W1 @ x3  # (2,)
    pred = (W2.squeeze(0) @ z)  # scalar
    return pred


ground_truth_grads_hessian_last_layer = ground_truth_grads_hessian_last_layer_torch


def ground_truth_grads_hessian_first_layer(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
):
    """
    Ground-truth for start_layer=0: derivatives wrt first layer weights W1 only (6 params).

    pred = sum_{j,k} W1[j,k] * (W2[j] * x[k])  (purely linear)
    Let a = kron(W2, x) with ordering matching PyTorch flatten for W1 (row-major):
      [W2[0]*x0, W2[0]*x1, W2[0]*x2, W2[1]*x0, W2[1]*x1, W2[1]*x2]

    grad_W1_flat = 2 (pred - y) a
    Hess_W1 = 2 a a^T
    """
    W1 = model[0].weight.detach()  # (2,3)
    W2 = model[1].weight.detach()  # (1,2)

    grads = []
    hess = []
    n = inputs.shape[0]

    w2 = W2.squeeze(0)  # (2,)

    for i in range(n):
        x3 = inputs[i].squeeze(0)         # (3,)
        y = targets[i].reshape(-1)[0]     # scalar

        pred = _per_sample_pred_scalar(W1, W2, x3)
        err = pred - y

        a = torch.kron(w2, x3)             # (6,)
        g = 2.0 * err * a                  # (6,)
        H = 2.0 * torch.outer(a, a)        # (6,6)

        grads.append(g)
        hess.append(H)

    grads_mat = torch.stack(grads, dim=0).T          # (6, N)
    hess_mean = torch.stack(hess, dim=0).mean(dim=0) # (6,6)
    return grads_mat, hess_mean


# -------------------------
# Tests (PyTorch)
# -------------------------
def test_compute_ihvp_single_batch_torch():
    torch.manual_seed(42)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    # Keep this test reasonably fast for iterative methods.
    n = 12
    inputs, targets = make_dataset(n, seed=123)
    ds = TensorDataset(inputs, targets)

    train_loader = DataLoader(ds, batch_size=4, shuffle=False)
    loader_bs1 = DataLoader(ds, batch_size=1, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    # Ground truth wrt last layer (2 params)
    gt_grads, gt_hess = ground_truth_grads_hessian_last_layer(model, inputs, targets)
    gt_inv_hess = torch.linalg.pinv(gt_hess)
    gt_ihvp = gt_inv_hess @ gt_grads  # (2, n)

    # ---- ExactIHVP
    ihvp_calculator = ExactIHVP(influence_model, train_loader)

    ihvp_list = []
    for batch in loader_bs1:
        batch_ihvp = ihvp_calculator._compute_ihvp_single_batch(batch)
        assert batch_ihvp.shape == (2, 1)
        ihvp_list.append(batch_ihvp)
    ihvp_batch = torch.cat(ihvp_list, dim=1)
    assert almost_equal(ihvp_batch, gt_ihvp, epsilon=1e-2)

    # ---- ConjugateGradientDescentIHVP (check a few samples only)
    ihvp_calculator = ConjugateGradientDescentIHVP(
        influence_model,
        extractor_layer=-1,
        train_dataset=train_loader,
        n_opt_iters=50,
    )
    ihvp_list = []
    for k, batch in enumerate(loader_bs1):
        if k >= 3:
            break
        batch_ihvp = ihvp_calculator._compute_ihvp_single_batch(batch)
        assert batch_ihvp.shape == (2, 1)
        ihvp_list.append(batch_ihvp)
    ihvp_batch = torch.cat(ihvp_list, dim=1)  # (2, 3)
    assert almost_equal(ihvp_batch, gt_ihvp[:, :3], epsilon=1e-1)

    # ---- LissaIHVP (check a few samples only)
    ihvp_calculator = LissaIHVP(
        influence_model,
        extractor_layer=-1,
        train_dataset=train_loader,
        scale=4.0,
        damping=1e-4,
        n_opt_iters=150,
    )
    ihvp_list = []
    for k, batch in enumerate(loader_bs1):
        if k >= 3:
            break
        batch_ihvp = ihvp_calculator._compute_ihvp_single_batch(batch)
        assert batch_ihvp.shape == (2, 1)
        ihvp_list.append(batch_ihvp)
    ihvp_batch = torch.cat(ihvp_list, dim=1)
    assert almost_equal(ihvp_batch, gt_ihvp[:, :3], epsilon=2e-1)


def test_compute_hvp_single_batch_torch():
    torch.manual_seed(123)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    n = 12
    inputs, targets = make_dataset(n, seed=321)
    ds = TensorDataset(inputs, targets)

    train_loader = DataLoader(ds, batch_size=4, shuffle=False)
    loader_bs1 = DataLoader(ds, batch_size=1, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    # Ground truth wrt last layer (2 params)
    gt_grads, gt_hess = ground_truth_grads_hessian_last_layer(model, inputs, targets)
    gt_hvp = gt_hess @ gt_grads  # (2, n)

    # ---- ExactIHVP (for HVP we need the Hessian)
    ihvp_calculator = ExactIHVP(influence_model, train_loader)
    ihvp_calculator.hessian = torch.linalg.pinv(ihvp_calculator.inv_hessian)

    hvp_list = []
    for batch in loader_bs1:
        batch_hvp = ihvp_calculator._compute_hvp_single_batch(batch)
        assert batch_hvp.shape == (2, 1)
        hvp_list.append(batch_hvp)
    hvp_batch = torch.cat(hvp_list, dim=1)
    assert almost_equal(hvp_batch, gt_hvp, epsilon=1e-2)

    # ---- ConjugateGradientDescentIHVP
    ihvp_calculator = ConjugateGradientDescentIHVP(
        influence_model, extractor_layer=-1, train_dataset=train_loader, n_opt_iters=50
    )
    hvp_list = []
    for batch in loader_bs1:
        batch_hvp = ihvp_calculator._compute_hvp_single_batch(batch)
        assert batch_hvp.shape == (2, 1)
        hvp_list.append(batch_hvp)
    hvp_batch = torch.cat(hvp_list, dim=1)
    assert almost_equal(hvp_batch, gt_hvp, epsilon=1e-2)

    # ---- LissaIHVP
    ihvp_calculator = LissaIHVP(
        influence_model,
        extractor_layer=-1,
        train_dataset=train_loader,
        scale=4.0,
        damping=1e-4,
        n_opt_iters=150,
    )
    hvp_list = []
    for batch in loader_bs1:
        batch_hvp = ihvp_calculator._compute_hvp_single_batch(batch)
        assert batch_hvp.shape == (2, 1)
        hvp_list.append(batch_hvp)
    hvp_batch = torch.cat(hvp_list, dim=1)
    assert almost_equal(hvp_batch, gt_hvp, epsilon=1e-2)


def test_stochastic_cgd_ihvp_close_to_full_torch():
    torch.manual_seed(7)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    n = 8
    inputs, targets = make_dataset(n, seed=7)
    ds = TensorDataset(inputs, targets)

    train_loader = DataLoader(ds, batch_size=4, shuffle=False)
    loader_bs2 = DataLoader(ds, batch_size=2, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    full_cgd = ConjugateGradientDescentIHVP(
        influence_model,
        extractor_layer=-1,
        train_dataset=train_loader,
        n_opt_iters=50,
    )
    stochastic_cgd = ConjugateGradientDescentIHVP(
        influence_model,
        extractor_layer=-1,
        train_dataset=train_loader,
        n_opt_iters=50,
        stochastic_hvp=True,
        hvp_steps_per_iter=4,
        hvp_batch_size=2,
    )

    full_list = []
    stochastic_list = []
    for batch in loader_bs2:
        full_list.append(full_cgd._compute_ihvp_single_batch(batch))
        stochastic_list.append(stochastic_cgd._compute_ihvp_single_batch(batch))

    full_ihvp = torch.cat(full_list, dim=1)
    stochastic_ihvp = torch.cat(stochastic_list, dim=1)
    assert almost_equal(stochastic_ihvp, full_ihvp, epsilon=1e-2)


def test_batched_rhs_hvp_matches_scalar_torch():
    torch.manual_seed(11)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs, targets = make_dataset(6, seed=11)
    ds = TensorDataset(inputs, targets)
    train_loader = DataLoader(ds, batch_size=3, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    ihvp_calculator = ConjugateGradientDescentIHVP(
        influence_model,
        extractor_layer=-1,
        train_dataset=train_loader,
        n_opt_iters=20,
    )

    rhs = torch.randn(ihvp_calculator.model.nb_params, 3)
    batched = ihvp_calculator.hessian_vector_product(rhs)
    single = [ihvp_calculator.hessian_vector_product(rhs[:, i:i + 1]) for i in range(3)]
    stacked = torch.cat(single, dim=1)

    assert almost_equal(batched, stacked, epsilon=1e-5)

def test_exact_hessian_torch():
    torch.manual_seed(7)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    # Small set: Hessian checks are exact and fast here.
    inputs, targets = make_dataset(5, seed=7)
    ds = TensorDataset(inputs, targets)
    train_loader = DataLoader(ds, batch_size=5, shuffle=False)

    # ---- First layer (6 params)
    influence_model = InfluenceModel(model, start_layer=0, loss_function=loss_fn)
    ihvp_calculator = ExactIHVP(influence_model, train_loader)
    inv_hessian = ihvp_calculator.inv_hessian
    assert inv_hessian.shape == (6, 6)

    gt_grads, gt_hess = ground_truth_grads_hessian_first_layer(model, inputs, targets)
    gt_inv = torch.linalg.pinv(gt_hess)
    assert almost_equal(inv_hessian, gt_inv, epsilon=1e-3)

    # ---- Last layer (2 params)
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    ihvp_calculator = ExactIHVP(influence_model, DataLoader(ds, batch_size=3, shuffle=False))
    inv_hessian = ihvp_calculator.inv_hessian
    assert inv_hessian.shape == (2, 2)

    gt_grads, gt_hess = ground_truth_grads_hessian_last_layer(model, inputs, targets)
    gt_inv = torch.linalg.pinv(gt_hess)
    assert almost_equal(inv_hessian, gt_inv, epsilon=1e-3)


def test_exact_ihvp_torch():
    torch.manual_seed(42)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    n = 25
    inputs, targets = make_dataset(n, seed=42)
    ds = TensorDataset(inputs, targets)

    train_loader = DataLoader(ds, batch_size=5, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    # Compute IHVP using the implementation and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_loader)
    ihvp_ds = ihvp_calculator.compute_ihvp(train_loader)  # PyTorch backend returns a list

    ihvp_list = []
    for elt in ihvp_ds:
        assert elt.shape == (2, 5)  # nb_params x batch_size
        ihvp_list.append(elt)
    ihvp = torch.cat(ihvp_list, dim=1)
    assert ihvp.shape == (2, n)

    # Ground truth
    gt_grads, gt_hess = ground_truth_grads_hessian_last_layer(model, inputs, targets)
    gt_inv = torch.linalg.pinv(gt_hess)
    gt_ihvp = gt_inv @ gt_grads
    assert almost_equal(ihvp, gt_ihvp, epsilon=1e-2)

    # Init ExactIHVP from a provided Hessian
    ihvp_calculator2 = ExactIHVP(influence_model, train_hessian=gt_hess)
    ihvp_ds2 = ihvp_calculator2.compute_ihvp(train_loader)

    ihvp_list2 = []
    for elt in ihvp_ds2:
        assert elt.shape == (2, 5)
        ihvp_list2.append(elt)
    ihvp2 = torch.cat(ihvp_list2, dim=1)
    assert ihvp2.shape == (2, n)
    assert almost_equal(ihvp2, gt_ihvp, epsilon=1e-2)

    # Vector mode (use_gradient=False)
    vectors = torch.randn(n, 2)
    vec_loader = DataLoader(TensorDataset(vectors), batch_size=5, shuffle=False)

    ihvp_vec_ds = ihvp_calculator.compute_ihvp(group=vec_loader, use_gradient=False)
    ihvp_vec = torch.cat([elt for elt in ihvp_vec_ds], dim=1)
    assert ihvp_vec.shape == (2, n)

    gt_ihvp_vec = gt_inv @ vectors.T
    assert almost_equal(ihvp_vec, gt_ihvp_vec, epsilon=1e-3)


def test_exact_hvp_torch():
    torch.manual_seed(1234)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    n = 25
    inputs, targets = make_dataset(n, seed=1234)
    ds = TensorDataset(inputs, targets)

    train_loader = DataLoader(ds, batch_size=5, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    hvp_calculator = ExactIHVP(influence_model, train_loader)
    hvp_ds = hvp_calculator.compute_hvp(train_loader)

    hvp = torch.cat([elt for elt in hvp_ds], dim=1)
    assert hvp.shape == (2, n)

    # Ground truth
    gt_grads, gt_hess = ground_truth_grads_hessian_last_layer(model, inputs, targets)
    gt_hvp = gt_hess @ gt_grads
    assert almost_equal(hvp, gt_hvp, epsilon=1e-3)

    # Init from provided Hessian
    hvp_calculator2 = ExactIHVP(influence_model, train_hessian=gt_hess)
    hvp2 = torch.cat([elt for elt in hvp_calculator2.compute_hvp(train_loader)], dim=1)
    assert hvp2.shape == (2, n)
    assert almost_equal(hvp2, gt_hvp, epsilon=1e-3)

    # Vector mode (use_gradient=False)
    vectors = torch.randn(n, 2)
    vec_loader = DataLoader(TensorDataset(vectors), batch_size=5, shuffle=False)

    hvp_vec = torch.cat([elt for elt in hvp_calculator.compute_hvp(vec_loader, use_gradient=False)], dim=1)
    assert hvp_vec.shape == (2, n)

    gt_hvp_vec = gt_hess @ vectors.T
    assert almost_equal(hvp_vec, gt_hvp_vec, epsilon=1e-3)


@pytest.mark.slow
def test_cgd_hvp_torch():
    torch.manual_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    n = 12
    inputs, targets = make_dataset(n, seed=0)
    ds = TensorDataset(inputs, targets)

    train_loader = DataLoader(ds, batch_size=4, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    ihvp_calculator = ConjugateGradientDescentIHVP(
        influence_model, extractor_layer=1, train_dataset=train_loader, n_opt_iters=50
    )
    hvp_ds = ihvp_calculator.compute_hvp(train_loader)
    hvp = torch.cat([elt for elt in hvp_ds], dim=1)
    assert hvp.shape == (2, n)

    gt_grads, gt_hess = ground_truth_grads_hessian_last_layer(model, inputs, targets)
    gt_hvp = gt_hess @ gt_grads
    assert almost_equal(hvp, gt_hvp, epsilon=1e-3)

    vectors = torch.randn(n, 2)
    vec_loader = DataLoader(TensorDataset(vectors), batch_size=4, shuffle=False)
    hvp_vec = torch.cat([elt for elt in ihvp_calculator.compute_hvp(vec_loader, use_gradient=False)], dim=1)
    assert hvp_vec.shape == (2, n)
    assert almost_equal(hvp_vec, gt_hess @ vectors.T, epsilon=1e-3)


@pytest.mark.slow
def test_cgd_ihvp_torch():
    torch.manual_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    n = 12
    inputs, targets = make_dataset(n, seed=1)
    ds = TensorDataset(inputs, targets)

    train_loader = DataLoader(ds, batch_size=4, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    ihvp_calculator = ConjugateGradientDescentIHVP(
        influence_model, extractor_layer=-1, train_dataset=train_loader, n_opt_iters=50
    )
    ihvp_ds = ihvp_calculator.compute_ihvp(train_loader)
    ihvp = torch.cat([elt for elt in ihvp_ds], dim=1)
    assert ihvp.shape == (2, n)

    gt_grads, gt_hess = ground_truth_grads_hessian_last_layer(model, inputs, targets)
    gt_inv = torch.linalg.pinv(gt_hess)
    gt_ihvp = gt_inv @ gt_grads
    assert almost_equal(ihvp, gt_ihvp, epsilon=1e-1)

    vectors = torch.randn(n, 2)
    vec_loader = DataLoader(TensorDataset(vectors), batch_size=4, shuffle=False)
    ihvp_vec = torch.cat([elt for elt in ihvp_calculator.compute_ihvp(vec_loader, use_gradient=False)], dim=1)
    assert ihvp_vec.shape == (2, n)
    assert almost_equal(ihvp_vec, gt_inv @ vectors.T, epsilon=1e-1)


@pytest.mark.slow
def test_lissa_ihvp_torch():
    torch.manual_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    n = 12
    inputs, targets = make_dataset(n, seed=2)
    ds = TensorDataset(inputs, targets)

    train_loader = DataLoader(ds, batch_size=4, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)

    ihvp_calculator = LissaIHVP(
        influence_model,
        extractor_layer=-1,
        train_dataset=train_loader,
        damping=1e-4,
        scale=4.0,
        n_opt_iters=150,
    )
    ihvp_ds = ihvp_calculator.compute_ihvp(train_loader)
    ihvp = torch.cat([elt for elt in ihvp_ds], dim=1)
    assert ihvp.shape == (2, n)

    gt_grads, gt_hess = ground_truth_grads_hessian_last_layer(model, inputs, targets)
    gt_inv = torch.linalg.pinv(gt_hess)
    gt_ihvp = gt_inv @ gt_grads
    assert almost_equal(ihvp, gt_ihvp, epsilon=2e-1)

    vectors = torch.randn(n, 2)
    vec_loader = DataLoader(TensorDataset(vectors), batch_size=4, shuffle=False)
    ihvp_vec = torch.cat([elt for elt in ihvp_calculator.compute_ihvp(vec_loader, use_gradient=False)], dim=1)
    assert ihvp_vec.shape == (2, n)
    assert almost_equal(ihvp_vec, gt_inv @ vectors.T, epsilon=2e-1)
