# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import pytest
import torch

from deel.influenciae.common import BaseInfluenceModel
from deel.influenciae.trackstar.optimizer_state import extract_optimizer_second_moments


pytestmark = pytest.mark.pytorch


@pytest.mark.parametrize("optimizer_type", (torch.optim.Adam, torch.optim.AdamW))
def test_extracts_raw_second_moments_in_model_order(optimizer_type):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    influence_model = BaseInfluenceModel(
        model, loss_function=torch.nn.MSELoss(reduction="none")
    )
    reversed_parameters = list(reversed(list(model.parameters())))
    optimizer = optimizer_type(reversed_parameters, lr=0.01, betas=(0.9, 0.5))
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index))
    optimizer.step()

    snapshot = extract_optimizer_second_moments(influence_model, optimizer)

    for index, (moment, parameter) in enumerate(
        zip(snapshot.parameter_second_moments, model.parameters()), start=1
    ):
        np.testing.assert_allclose(moment, np.full(tuple(parameter.shape), 0.5 * index ** 2))
        assert not moment.flags.writeable


def test_missing_or_unsupported_pytorch_optimizer_state_fails_without_mutation():
    model = torch.nn.Linear(2, 1)
    influence_model = BaseInfluenceModel(
        model, loss_function=torch.nn.MSELoss(reduction="none")
    )
    optimizer = torch.optim.Adam(model.parameters())

    with pytest.raises(RuntimeError, match="not initialized"):
        extract_optimizer_second_moments(influence_model, optimizer)
    assert len(optimizer.state) == 0

    with pytest.raises(TypeError, match="Adam or AdamW"):
        extract_optimizer_second_moments(
            influence_model, torch.optim.SGD(model.parameters(), lr=0.1)
        )


def test_amsgrad_snapshot_uses_raw_second_moment():
    model = torch.nn.Linear(1, 1, bias=False)
    influence_model = BaseInfluenceModel(
        model, loss_function=torch.nn.MSELoss(reduction="none")
    )
    parameter = next(model.parameters())
    optimizer = torch.optim.Adam([parameter], amsgrad=True)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["max_exp_avg_sq"].fill_(9.0)

    snapshot = extract_optimizer_second_moments(influence_model, optimizer)

    np.testing.assert_allclose(
        snapshot.parameter_second_moments[0],
        optimizer.state[parameter]["exp_avg_sq"].detach().numpy(),
    )
