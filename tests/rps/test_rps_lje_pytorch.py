# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch test suite for RepresenterPointLJE.

This mirrors the TensorFlow tests (test_alpha, influence vector, preprocess, influence
value, pairwise influence, inheritance) but uses PyTorch models + DataLoaders and
PyTorch loss functions with reduction='none' behavior.

Save as something like:
  tests/rps/test_representer_point_lje_pytorch.py
(or next to the TF test file with a matching naming convention).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ExactIHVPFactory
from deel.influenciae.rps import RepresenterPointLJE



# -------------------------
# Helpers
# -------------------------
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, rtol=1e-5, atol=1e-6):
    a = a.detach().cpu()
    b = b.detach().cpu()
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        abs_err = (a - b).abs().max().item()
        rel_err = ((a - b).abs() / (b.abs() + 1e-12)).max().item()
        raise AssertionError(
            f"Not close: max_abs={abs_err:.3e}, max_rel={rel_err:.3e}, rtol={rtol}, atol={atol}"
        )


def _assert_relative_almost_equal(a: torch.Tensor, b: torch.Tensor, percent: float = 0.1):
    a = a.detach().cpu()
    b = b.detach().cpu()
    denom = torch.clamp(b.abs(), min=1e-12)
    rel = (a - b).abs() / denom
    max_rel = rel.max().item()
    if max_rel >= percent:
        raise AssertionError(f"Relative error too large: max_rel={max_rel:.3e} >= {percent:.3e}")


def _make_model(out_features: int, dtype=torch.float64) -> nn.Module:
    """
    Keep the head small so ExactIHVP (and PyTorch Hessian code) stays fast in CI.
    Input: (N, 3, 5, 5)
    Conv(3x3, out_channels=2) -> (N, 2, 3, 3) -> flatten 18 -> Linear(18->out_features, bias=False)
    """
    model = nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(18, out_features, bias=False),
    )
    return model.to(dtype=dtype)


def _feature_extractor_and_head(model: nn.Sequential):
    layers = list(model.children())
    feature_extractor = nn.Sequential(*layers[:-1])
    head = nn.Sequential(layers[-1])  # keep as a Sequential container like the backend does
    return feature_extractor, head


def _cce_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Soft-label categorical cross-entropy from logits.
    Matches tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=NONE)
    up to numeric differences (targets can be non-normalized in the tests, as in the TF suite).
    Returns shape (B,).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1)


def _bce_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy from logits with no reduction.
    Matches tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=NONE) in spirit.
    Returns shape (B,).
    """
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(dim=-1)


class PermuteNHWCtoNCHW(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()


def _make_model_nhwc(out_features: int, dtype=torch.float64) -> nn.Module:
    """
    Mirrors the TF test architecture for inheritance expectations:
      NHWC (5,5,3) -> NCHW -> Conv2d(out=4, k=2) => (4,4,4) -> Flatten => 64 -> Linear(64->out)
    """
    return nn.Sequential(
        PermuteNHWCtoNCHW(),             # (N,5,5,3) -> (N,3,5,5)
        nn.Conv2d(3, 4, kernel_size=2),  # -> (N,4,4,4)  => 64 features after flatten
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64, out_features, bias=False),
    ).to(dtype=dtype)



# -------------------------
# Tests (PyTorch)
# -------------------------
def test_alpha_pytorch():
    torch.manual_seed(0)
    device = _device()
    dtype = torch.float64

    model = _make_model(out_features=4, dtype=dtype).to(device)
    loss_function = _cce_from_logits

    inputs_train = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_train = torch.randn(50, 4, dtype=dtype, device=device)

    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_loader, ExactIHVPFactory(), target_layer=-1)

    feature_extractor, _ = _feature_extractor_and_head(model)
    feature_extractor = feature_extractor.to(device).to(dtype)
    feature_maps = feature_extractor(inputs_train)

    # Alpha from implementation
    alpha = rps_lje._compute_alpha(feature_maps, targets_train)

    # ---- Manual alpha (mirrors TF test logic) ----
    # 1) Perturb the head with 1 SGD step on -mean(loss)
    perturbed_head = copy.deepcopy(model[-1]).to(device).to(dtype)
    perturbed_head.train()
    opt = torch.optim.SGD(perturbed_head.parameters(), lr=1e-4)

    opt.zero_grad(set_to_none=True)
    logits = perturbed_head(feature_maps)
    loss = (-loss_function(logits, targets_train)).mean()
    loss.backward()
    opt.step()
    perturbed_head.eval()

    # 2) Build ExactIHVP on the perturbed head
    dataset_for_hessian = DataLoader(
        TensorDataset(feature_maps.detach(), targets_train.detach()),
        batch_size=5,
        shuffle=False,
    )
    ihvp = ExactIHVP(InfluenceModel(perturbed_head, start_layer=0, loss_function=loss_function), dataset_for_hessian)

    # 3) Per-sample gradients wrt weights, divide by (B*z + eps), IHVP, sum over in_features
    W = next(p for p in perturbed_head.parameters() if p.requires_grad)  # (out, in)
    out_features, in_features = W.shape
    B = feature_maps.shape[0]
    eps = 1e-5

    logits = perturbed_head(feature_maps)  # keep graph wrt weights
    losses = loss_function(logits, targets_train)  # (B,)

    grads = []
    for i in range(B):
        g = torch.autograd.grad(losses[i], W, retain_graph=True, create_graph=False)[0]  # (out, in)
        grads.append(g)
    grads = torch.stack(grads, dim=0)  # (B, out, in)
    grads = grads.permute(0, 2, 1)  # (B, in, out)

    divisor = (B * feature_maps) + eps  # (B, in)
    grads_div = grads / divisor.unsqueeze(-1)  # (B, in, out)

    second_term_list = []
    for i in range(B):
        grad_flat = grads_div[i].permute(1, 0).reshape(1, -1)  # (1, out*in), same as implementation
        ihvp_res = ihvp._compute_ihvp_single_batch((grad_flat,), use_gradient=False)  # protected in TF test too
        second_term_list.append(ihvp_res.squeeze())
    second_term = torch.stack(second_term_list, dim=0)  # (B, out*in)
    second_term = second_term.view(B, out_features, in_features).permute(0, 2, 1)  # (B, in, out)
    second_term_summed = second_term.sum(dim=1)  # (B, out)

    # First term: W^T / divisor, sum over in_features
    first_term = (W.T.unsqueeze(0) / divisor.unsqueeze(-1)).sum(dim=1)  # (B, out)

    alpha_test = first_term - second_term_summed

    assert alpha.shape == alpha_test.shape
    _assert_relative_almost_equal(alpha, alpha_test, percent=0.1)


def test_compute_influence_vector_pytorch():
    torch.manual_seed(0)
    device = _device()
    dtype = torch.float64

    model = _make_model(out_features=4, dtype=dtype).to(device)
    loss_function = _cce_from_logits

    inputs_train = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_train = torch.randn(50, 4, dtype=dtype, device=device)
    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_loader, ExactIHVPFactory(), target_layer=-1)

    alpha, z_batch = rps_lje._compute_influence_vector((inputs_train, targets_train))

    feature_extractor, _ = _feature_extractor_and_head(model)
    z_batch_test = feature_extractor(inputs_train)
    alpha_test = rps_lje._compute_alpha(z_batch_test, targets_train)

    _assert_allclose(z_batch, z_batch_test, rtol=1e-6, atol=1e-6)
    _assert_allclose(alpha, alpha_test, rtol=1e-6, atol=1e-6)


def test_preprocess_sample_to_evaluate_pytorch():
    torch.manual_seed(0)
    device = _device()
    dtype = torch.float64

    model = _make_model(out_features=4, dtype=dtype).to(device)
    loss_function = _cce_from_logits

    inputs_train = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_train = torch.randn(50, 4, dtype=dtype, device=device)
    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)

    inputs_test = torch.randn(60, 3, 5, 5, dtype=dtype, device=device)
    targets_test = torch.randn(60, 1, dtype=dtype, device=device)  # same “shape mismatch” as TF test

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_loader, ExactIHVPFactory(), target_layer=-1)
    pre = rps_lje._preprocess_samples((inputs_test, targets_test))

    feature_extractor, _ = _feature_extractor_and_head(model)
    feature_maps = feature_extractor(inputs_test)

    _assert_allclose(pre[0], feature_maps, rtol=1e-6, atol=1e-6)
    _assert_allclose(pre[1], targets_test, rtol=1e-6, atol=1e-6)


def test_compute_influence_value_from_influence_vector_binary_pytorch():
    torch.manual_seed(0)
    device = _device()
    dtype = torch.float64

    model = _make_model(out_features=1, dtype=dtype).to(device)
    loss_function = _bce_from_logits

    inputs_train = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_train = torch.randn(50, 1, dtype=dtype, device=device)
    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_loader, ExactIHVPFactory(), target_layer=-1)

    influence_values_computed = rps_lje._compute_influence_value_from_batch((inputs_train, targets_train))

    alpha, _ = rps_lje._compute_influence_vector((inputs_train, targets_train))
    influence_values = alpha.abs()

    _assert_allclose(influence_values_computed, influence_values, rtol=0.0, atol=1e-3)


def test_compute_influence_value_from_influence_vector_multiclass_pytorch():
    torch.manual_seed(0)
    device = _device()
    dtype = torch.float64

    model = _make_model(out_features=4, dtype=dtype).to(device)
    loss_function = _cce_from_logits

    inputs_train = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_train = torch.randn(50, 4, dtype=dtype, device=device)
    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_loader, ExactIHVPFactory(), target_layer=-1)

    influence_values_computed = rps_lje._compute_influence_value_from_batch((inputs_train, targets_train))
    influence_values_computed = influence_values_computed.squeeze(-1)

    alpha, z_batch = rps_lje._compute_influence_vector((inputs_train, targets_train))
    indices = torch.argmax(rps_lje.perturbed_head(z_batch), dim=1)  # (B,)
    alpha_i = alpha.gather(1, indices.view(-1, 1)).squeeze(1)  # (B,)
    influence_values = alpha_i.abs()

    _assert_relative_almost_equal(influence_values_computed, influence_values, percent=0.05)


def test_compute_pairwise_influence_value_binary_pytorch():
    torch.manual_seed(0)
    device = _device()
    dtype = torch.float64

    model = _make_model(out_features=1, dtype=dtype).to(device)
    loss_function = _bce_from_logits

    inputs_train = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_train = torch.randn(50, 1, dtype=dtype, device=device)

    inputs_test = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_test = torch.randn(50, 1, dtype=dtype, device=device)

    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_loader, ExactIHVPFactory(), target_layer=-1)

    v_test = rps_lje._preprocess_samples((inputs_test, targets_test))
    influence_vector = rps_lje._compute_influence_vector((inputs_train, targets_train))
    influence_values_computed = rps_lje._estimate_influence_value_from_influence_vector(v_test, influence_vector)

    feature_extractor, _ = _feature_extractor_and_head(model)
    feature_maps_train = feature_extractor(inputs_train)
    feature_maps_test = feature_extractor(inputs_test)
    alpha_train = influence_vector[0]  # (N_train, 1)

    K = torch.matmul(feature_maps_train, feature_maps_test.T)  # (N_train, N_test)
    influence_values_test = (alpha_train * K).T  # (N_test, N_train)

    _assert_relative_almost_equal(influence_values_computed, influence_values_test, percent=0.1)


def test_compute_pairwise_influence_value_multiclass_pytorch():
    torch.manual_seed(0)
    device = _device()
    dtype = torch.float64

    model = _make_model(out_features=4, dtype=dtype).to(device)
    loss_function = _cce_from_logits

    inputs_train = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_train = torch.randn(50, 4, dtype=dtype, device=device)

    inputs_test = torch.randn(50, 3, 5, 5, dtype=dtype, device=device)
    targets_test = torch.randn(50, 4, dtype=dtype, device=device)

    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_loader, ExactIHVPFactory(), target_layer=-1)

    v_test = rps_lje._preprocess_samples((inputs_test, targets_test))
    influence_vector = rps_lje._compute_influence_vector((inputs_train, targets_train))
    influence_values_computed = rps_lje._estimate_influence_value_from_influence_vector(v_test, influence_vector)

    feature_extractor, _ = _feature_extractor_and_head(model)
    feature_maps_train = feature_extractor(inputs_train)
    feature_maps_test = feature_extractor(inputs_test)

    # Match TF test: train and test have same batch size, so we can do row-wise gather ("batch_dims=1" equivalent)
    indices = torch.argmax(rps_lje.perturbed_head(feature_maps_test), dim=1)  # (50,)
    alpha = influence_vector[0]  # (50, 4)
    alpha_rowwise = alpha.gather(1, indices.view(-1, 1)).squeeze(1)  # (50,)

    K = torch.matmul(feature_maps_train, feature_maps_test.T)  # (50, 50)
    influence_values_test = (alpha_rowwise.view(-1, 1) * K).T  # (50, 50)

    _assert_relative_almost_equal(influence_values_computed, influence_values_test, percent=0.1)


def test_inheritance_pytorch():
    torch.manual_seed(0)
    device = _device()
    dtype = torch.float64

    model = _make_model_nhwc(out_features=1, dtype=dtype).to(device)
    loss_function = _bce_from_logits

    inputs_train = torch.randn(10, 5, 5, 3, dtype=dtype, device=device)
    targets_train = torch.randn(10, 1, dtype=dtype, device=device)

    inputs_test = torch.randn(50, 5, 5, 3, dtype=dtype, device=device)
    targets_test = torch.randn(50, 1, dtype=dtype, device=device)

    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=5, shuffle=False)
    test_loader = DataLoader(TensorDataset(inputs_test, targets_test), batch_size=10, shuffle=False)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_function)
    rps_lje = RepresenterPointLJE(influence_model, train_loader, ExactIHVPFactory(), target_layer=-1)

    nb_params = influence_model.nb_params

    # Import here to avoid module-level TensorFlow import in PyTorch-only tests
    from ..utils_test import assert_inheritance
    assert_inheritance(rps_lje, nb_params, train_loader, test_loader)
