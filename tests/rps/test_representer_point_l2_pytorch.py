# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.rps.rps_l2 import RepresenterPointL2
from ..utils_test import set_seed_torch


pytestmark = pytest.mark.pytorch


_seed_all = partial(set_seed_torch, include_cuda=True)


def _make_multiclass_problem(n=100, num_classes=4, batch_train=32, batch_eval=20):
    x = torch.randn((n, 3, 32, 32), dtype=torch.float32)
    y = torch.randint(0, num_classes, (n,), dtype=torch.long)
    train_set = TensorDataset(x, y)
    train_loader = DataLoader(train_set, batch_size=batch_train, shuffle=True)
    eval_loader = DataLoader(train_set, batch_size=batch_eval, shuffle=False)
    return x, y, train_loader, eval_loader


def _make_binary_problem(n=100, batch_train=32, batch_eval=20):
    x = torch.randn((n, 3, 32, 32), dtype=torch.float32)
    y = torch.randint(0, 2, (n, 1), dtype=torch.float32)
    train_set = TensorDataset(x, y)
    train_loader = DataLoader(train_set, batch_size=batch_train, shuffle=True)
    eval_loader = DataLoader(train_set, batch_size=batch_eval, shuffle=False)
    return x, y, train_loader, eval_loader


def _train_model_multiclass(model, train_loader, epochs=40, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
    model.eval()


def _train_model_binary(model, train_loader, epochs=40, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
    model.eval()


def test_surrogate_model_multiclass_learns_teacher_logits():
    """
    Mirrors TF test_surrogate_model:
      - surrogate has correct in/out dims
      - after training, surrogate predicts similar logits as teacher head on train data
    """
    _seed_all(0)

    x_train, y_train, train_loader, eval_loader = _make_multiclass_problem()
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 4, 1),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4, bias=False),
    )

    _train_model_multiclass(model, train_loader, epochs=40)

    rps_l2 = RepresenterPointL2(
        model,
        eval_loader,
        loss_function=nn.CrossEntropyLoss(reduction="none"),
        lambda_regularization=0.1,
    )

    surrogate = rps_l2._create_surrogate_model()
    assert surrogate.in_features == model[-1].in_features
    assert surrogate.out_features == model[-1].out_features

    # retrain surrogate (more epochs than init) and check it mimics teacher logits
    rps_l2._train_last_layer(epochs=120)
    surrogate = rps_l2.linear_layer

    with torch.no_grad():
        z = rps_l2.feature_extractor(x_train)
        teacher_logits = rps_l2.original_head(z)
        surrogate_logits = surrogate(z)

        teacher_p = torch.softmax(teacher_logits, dim=1)
        surrogate_logp = torch.log_softmax(surrogate_logits, dim=1)
        kl = torch.sum(teacher_p * (torch.log(teacher_p + 1e-8) - surrogate_logp), dim=1).mean().item()

    assert kl < 0.20, f"Surrogate did not match teacher distribution (KL={kl:.4f})"


def test_alpha_matches_closed_form_cross_entropy_gradient():
    """
    Mirrors TF test_gradients:
    Compare _compute_alpha output to a closed-form expression for CE gradients.

    In PyTorch:
      - targets are integer class labels
      - for CE on logits: dL/dW = (softmax(logits)-one_hot(y))^T @ z
      - Your method then scales by (-2*lambda*n_train) and divides by z (elementwise) and sums over in_features.
    """
    _seed_all(1)

    n = 100
    num_classes = 4
    x_train, y_train, train_loader, eval_loader = _make_multiclass_problem(n=n, num_classes=num_classes, batch_eval=50)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 4, 0),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, num_classes, bias=False),
    )
    _train_model_multiclass(model, train_loader, epochs=40)

    lambda_reg = 0.01
    rps_l2 = RepresenterPointL2(
        model,
        eval_loader,
        loss_function=nn.CrossEntropyLoss(reduction="none"),
        lambda_regularization=lambda_reg,
    )
    surrogate = rps_l2.linear_layer

    eps = 1e-5

    with torch.no_grad():
        z = rps_l2.feature_extractor(x_train)          # (N, in_features)
        logits = surrogate(z)                          # (N, C)
        p = torch.softmax(logits, dim=1)               # (N, C)
        y_onehot = F.one_hot(y_train, num_classes=num_classes).float()

        # ground_truth_gradients: (N, C, in_features)
        # = outer(z, p - y_onehot) with proper broadcasting
        gt_grad = torch.matmul(
            z.unsqueeze(-1),                           # (N, in_features, 1)
            (p - y_onehot).unsqueeze(1)                # (N, 1, C)
        ).transpose(1, 2)                              # (N, C, in_features)

        denom = (-2.0 * lambda_reg * float(n)) + eps
        gt_inf = gt_grad / denom                       # (N, C, in_features)

        # divide by z elementwise + eps, then sum over in_features
        z_denom = z + eps
        gt_alpha = (gt_inf * (1.0 / z_denom).unsqueeze(1)).sum(dim=2)  # (N, C)

    # AD-based alpha from the implementation
    grads = []
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=50, shuffle=False)
    for xb, yb in loader:
        with torch.no_grad():
            zb = rps_l2.feature_extractor(xb)
        ab = rps_l2._compute_alpha(zb, yb)
        grads.append(ab.detach())
    alpha = torch.cat(grads, dim=0)

    assert alpha.shape == (n, num_classes)

    max_err = torch.max(torch.abs(gt_alpha - alpha)).item()
    assert max_err < 5e-3, f"Alpha mismatch too large (max_err={max_err:.6f})"


def test_influence_values_matches_closed_form_gathered_alpha():
    """
    Mirrors TF test_influence_values:
      - compute alpha in closed-form
      - gather per-sample predicted-class alpha
      - compare absolute values to _compute_influence_values
    """
    _seed_all(2)

    n = 100
    num_classes = 4
    x_train, y_train, train_loader, eval_loader = _make_multiclass_problem(n=n, num_classes=num_classes, batch_eval=20)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 4, 0),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, num_classes, bias=False),
    )
    _train_model_multiclass(model, train_loader, epochs=40)

    lambda_reg = 0.1
    rps_l2 = RepresenterPointL2(
        model,
        eval_loader,
        loss_function=nn.CrossEntropyLoss(reduction="none"),
        lambda_regularization=lambda_reg,
    )
    surrogate = rps_l2.linear_layer

    eps = 1e-5

    with torch.no_grad():
        z = rps_l2.feature_extractor(x_train)     # (N, in_features)
        logits_surr = surrogate(z)               # (N, C)
        p = torch.softmax(logits_surr, dim=1)
        y_onehot = F.one_hot(y_train, num_classes=num_classes).float()

        gt_grad = torch.matmul(
            z.unsqueeze(-1),
            (p - y_onehot).unsqueeze(1)
        ).transpose(1, 2)                        # (N, C, in_features)

        denom = (-2.0 * lambda_reg * float(n)) + eps
        gt_inf = gt_grad / denom

        z_denom = z + eps
        gt_alpha = (gt_inf * (1.0 / z_denom).unsqueeze(1)).sum(dim=2)  # (N, C)

        pred_idx = torch.argmax(model(x_train), dim=1)                 # teacher pred class
        gt_val = gt_alpha[torch.arange(n), pred_idx].abs()             # (N,)

    influence = rps_l2._compute_influence_values(eval_loader)          # expected shape (N,) or (N,1) depending impl
    influence = influence.view(-1)

    assert influence.shape[0] == n

    max_err = torch.max(torch.abs(gt_val - influence)).item()
    assert max_err < 5e-2, f"Influence mismatch too large (max_err={max_err:.6f})"


def test_predict_with_kernel_binary_is_finite_and_close_to_teacher_logits():
    """
    Mirrors TF test_predict_with_kernel:
      - ensure finite
      - ensure kernel preds approximate teacher logits reasonably (BCE between teacher and kernel)
    """
    _seed_all(3)

    x_train, y_train, train_loader, eval_loader = _make_binary_problem()

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 5, 0),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 1, bias=False),
    )
    _train_model_binary(model, train_loader, epochs=40)

    lambda_reg = 10.0
    rps_l2 = RepresenterPointL2(
        model,
        eval_loader,
        loss_function=nn.BCEWithLogitsLoss(reduction="none"),
        lambda_regularization=lambda_reg,
    )

    # kernel preds over whole dataset
    preds = []
    for xb, yb in eval_loader:
        out = rps_l2.predict_with_kernel((xb, yb))
        preds.append(out.detach())
    kernel_preds = torch.cat(preds, dim=0).view(-1)

    assert kernel_preds.shape[0] == x_train.shape[0]
    assert torch.isfinite(kernel_preds).all(), "Kernel predictions contain NaN/Inf values"

    with torch.no_grad():
        teacher_logits = model(x_train).view(-1)

    # Compare in a loss space like TF does: BCE between teacher logits and kernel preds should be small-ish
    # Here we treat teacher logits as "targets" in logit-space; simplest stable check is MSE.
    # If you *really* want BCE, you need probabilities. We'll mimic TF intent (close logits).
    mse = torch.mean((teacher_logits - kernel_preds) ** 2).item()
    assert mse < 1.0, f"Kernel preds too far from teacher logits (mse={mse:.4f})"


def test_inheritance_shapes_basic():
    """
    Keep the inheritance test but make it stricter and simpler:
      - validate shapes returned by key base methods
      - validate no NaNs
    """
    _seed_all(4)

    class Permute(nn.Module):
        def forward(self, x):
            return x.permute(0, 3, 1, 2)

    model = nn.Sequential(
        Permute(),
        nn.Conv2d(3, 4, kernel_size=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 1, bias=False),
    )
    nn.init.ones_(model[-1].weight)

    # Warm-up forward to ensure shapes are set (some code relies on this)
    _ = model(torch.randn((2, 5, 5, 3)))

    inputs_train = torch.randn((10, 5, 5, 3))
    inputs_test = torch.randn((50, 5, 5, 3))
    targets_train = torch.randn((10, 1))
    targets_test = torch.randn((50, 1))

    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)
    test_loader = DataLoader(TensorDataset(inputs_test, targets_test), batch_size=10, shuffle=False)

    method = RepresenterPointL2(
        model,
        train_loader,
        loss_function=nn.BCEWithLogitsLoss(reduction="none"),
        lambda_regularization=10.0,
    )

    test_batch = next(iter(test_loader))   # (x_test_b, y_test_b)
    train_batch = next(iter(train_loader)) # (x_train_b, y_train_b)

    inf = method._estimate_individual_influence_values_from_batch(
        train_samples=train_batch,
        samples_to_evaluate=test_batch,
    )
    assert inf.shape == (10, 5)
    assert torch.isfinite(inf).all()

    alpha, z = method._compute_influence_vector(train_batch)
    assert alpha.shape[0] == 5
    assert z.shape[0] == 5
    assert torch.isfinite(alpha).all()
    assert torch.isfinite(z).all()

    self_inf = method._compute_influence_values(train_loader)
    self_inf = self_inf.view(-1)
    assert self_inf.shape[0] == 10
    assert torch.isfinite(self_inf).all()
