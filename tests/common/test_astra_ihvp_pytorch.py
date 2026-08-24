# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Focused ASTRA IHVP tests for PyTorch."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.common import AstraConfig, AstraIHVP, EkfacIHVP, InfluenceModel
from deel.influenciae.common.ggn import GeneralizedGaussNewtonOperator


pytestmark = pytest.mark.pytorch


@pytest.fixture(scope="module")
def astra_setup():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(2, 1, bias=False, dtype=torch.float64))
    inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]], dtype=torch.float64)
    targets = torch.tensor([[1.], [-1.], [0.5], [0.]], dtype=torch.float64)
    dataset = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    influence_model = InfluenceModel(
        model, start_layer=0, last_layer=-1, loss_function=nn.MSELoss(reduction="none")
    )
    ekfac = EkfacIHVP(influence_model, dataset, damping=0.2)
    return influence_model, dataset, ekfac, (inputs[:2], targets[:2])


def _astra(setup, **config_kwargs):
    model, dataset, ekfac, _ = setup
    config = AstraConfig(preconditioner_damping=0.2, **config_kwargs)
    return AstraIHVP(model, dataset, config=config, ekfac_factors=ekfac.factors)


def test_zero_iteration_and_one_step_initializations(astra_setup):
    _, _, ekfac, batch = astra_setup
    rhs = torch.tensor([[1., -2.], [0.5, 3.]], dtype=torch.float64)
    expected = ekfac.precondition_gradient(rhs)
    initialized = _astra(astra_setup, n_iterations=0).precondition_gradient(rhs)
    one_step = _astra(
        astra_setup, n_iterations=1, learning_rate=1.0,
        damping=0.2, initialize_from_ekfac=False,
    ).precondition_gradient(rhs)
    torch.testing.assert_close(initialized, expected)
    torch.testing.assert_close(one_step, expected)

    from_gradient = _astra(astra_setup, n_iterations=0)._compute_ihvp_single_batch(batch)
    grads = astra_setup[0].batch_jacobian_tensor(batch)
    from_vectors = _astra(astra_setup, n_iterations=0).precondition_gradient(grads)
    torch.testing.assert_close(from_gradient, from_vectors)


def test_matrix_rhs_chunk_parity_and_shared_sampler(astra_setup):
    model, dataset, ekfac, batch = astra_setup
    rhs = torch.tensor([[1., 2.], [-2., 1.], [0.5, -1.]], dtype=torch.float64)
    calls = []

    def sampler(step):
        calls.append(step)
        return batch

    common = dict(
        n_iterations=3, learning_rate=0.05, damping=0.2,
        preconditioner_damping=0.2, initialize_from_ekfac=False,
    )
    chunked = AstraIHVP(
        model, dataset, config=AstraConfig(rhs_chunk_size=1, **common),
        ekfac_factors=ekfac.factors, curvature_batch_sampler=sampler,
    ).precondition_gradient(rhs)
    unchunked = AstraIHVP(
        model, dataset, config=AstraConfig(**common),
        ekfac_factors=ekfac.factors, curvature_batch_sampler=lambda _: batch,
    ).precondition_gradient(rhs)
    assert calls == [0, 1, 2]
    assert tuple(chunked.shape) == (2, 3)
    torch.testing.assert_close(chunked, unchunked)


def test_controlled_ggn_step_improves_residual(astra_setup):
    model, dataset, ekfac, batch = astra_setup
    rhs = torch.tensor([[1., -0.5]], dtype=torch.float64)
    config = AstraConfig(
        n_iterations=1,
        learning_rate=0.05,
        damping=0.2,
        preconditioner_damping=0.2,
        initialize_from_ekfac=False,
    )
    astra = AstraIHVP(
        model, dataset, config=config, ekfac_factors=ekfac.factors,
        curvature_batch_sampler=lambda _: batch,
    )
    solution = astra.precondition_gradient(rhs).T
    ggn = GeneralizedGaussNewtonOperator(model, batch, "mean")
    residual = ggn.matmat(solution) + config.damping * solution - rhs
    assert torch.linalg.vector_norm(residual) < torch.linalg.vector_norm(rhs)


def test_astra_improves_toward_exact_damped_ggn_solve():
    """Full-batch ASTRA refinement should improve its EK-FAC initialization."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, 2, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(2, 1, dtype=torch.float64),
    )
    inputs = torch.randn(8, 2, dtype=torch.float64)
    targets = torch.randn(8, 1, dtype=torch.float64)
    dataset = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    influence_model = InfluenceModel(
        model, start_layer=0, last_layer=-1, loss_function=nn.MSELoss(reduction="none")
    )
    damping = 0.1
    ekfac = EkfacIHVP(influence_model, dataset, damping=damping)
    rhs = torch.randn(1, influence_model.nb_params, dtype=torch.float64)
    ggn = GeneralizedGaussNewtonOperator(
        influence_model, (inputs, targets), "mean"
    )
    basis = torch.eye(influence_model.nb_params, dtype=torch.float64)
    system = ggn.matmat(basis) + damping * basis
    exact = torch.linalg.solve(system, rhs.T).T
    initial = ekfac.precondition_gradient(rhs).T
    astra = AstraIHVP(
        influence_model,
        dataset,
        config=AstraConfig(
            damping=damping,
            n_iterations=50,
            learning_rate=0.05,
        ),
        ekfac_factors=ekfac.factors,
        curvature_batch_sampler=lambda _step: (inputs, targets),
    )

    refined = astra.precondition_gradient(rhs).T

    assert torch.linalg.vector_norm(refined - exact) < torch.linalg.vector_norm(initial - exact)
    assert torch.linalg.vector_norm(refined @ system.T - rhs) < torch.linalg.vector_norm(
        initial @ system.T - rhs
    )


def test_strict_coverage_rejects_an_empty_layer_selection(astra_setup):
    model, dataset, _, _ = astra_setup
    with pytest.warns(UserWarning, match="No supported layers"):
        with pytest.raises(ValueError, match="Strict EK-FAC coverage"):
            AstraIHVP(model, dataset, target_layers=[])


def test_dataset_factor_and_operation_validation(astra_setup, tmp_path):
    model, dataset, ekfac, _ = astra_setup
    with pytest.raises(ValueError, match="restartable"):
        AstraIHVP(model, iter(dataset))
    empty_dataset = DataLoader(
        TensorDataset(
            torch.empty(0, 2, dtype=torch.float64),
            torch.empty(0, 1, dtype=torch.float64),
        ),
        batch_size=2,
    )
    with pytest.raises(ValueError, match="empty"):
        AstraIHVP(model, empty_dataset)
    with pytest.raises(ValueError, match="cannot be combined"):
        AstraIHVP(
            model,
            dataset,
            ekfac_factors=ekfac.factors,
            factors_path=str(tmp_path / "factors"),
        )

    astra = _astra(astra_setup, n_iterations=0)
    with pytest.raises(NotImplementedError, match="not a direct HVP"):
        astra._compute_hvp_single_batch((torch.ones(1, model.nb_params),), False)
    with pytest.raises(ValueError, match="at least one right-hand-side"):
        astra.precondition_gradient(torch.empty(0, model.nb_params, dtype=torch.float64))
