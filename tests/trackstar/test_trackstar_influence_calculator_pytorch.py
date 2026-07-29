# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.common import BaseInfluenceModel
from deel.influenciae.trackstar import (
    FixedMix,
    ProjectionBlock,
    ProjectionTerm,
    TrackStarBuilder,
    TrackStarProjectionPlan,
)


pytestmark = pytest.mark.pytorch


def _projection_plan(wrapper):
    blocks = []
    for index, shape in enumerate(wrapper.parameter_layout.parameter_shapes):
        rows = shape[0] if shape else 1
        columns = int(np.prod(shape)) // rows if shape else 1
        term = ProjectionTerm(
            f"parameter_{index}",
            (index,),
            (shape,),
            (1, 1),
            output_axis=0,
            left_matrix=np.ones((1, rows)),
            right_matrix=np.ones((columns, 1)),
        )
        blocks.append(ProjectionBlock(f"block_{index}", (term,)))
    return TrackStarProjectionPlan.from_layout(wrapper.parameter_layout, blocks)


def test_pytorch_builder_fits_once_and_exact_search_matches_brute_force():
    torch.manual_seed(4)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=False, dtype=torch.float64),
        torch.nn.Linear(3, 1, bias=False, dtype=torch.float64),
    )
    wrapper = BaseInfluenceModel(model, loss_function=torch.nn.MSELoss(reduction="none"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model(torch.ones((2, 2), dtype=torch.float64)).square().sum().backward()
    optimizer.step()
    train = DataLoader(
        TensorDataset(torch.arange(12, dtype=torch.float64).reshape(6, 2) / 10, torch.zeros((6, 1))),
        batch_size=2,
    )
    evaluation = DataLoader(
        TensorDataset(torch.ones((3, 2), dtype=torch.float64), torch.zeros((3, 1))),
        batch_size=2,
    )
    builder = TrackStarBuilder.from_optimizer(
        wrapper,
        optimizer,
        _projection_plan(wrapper),
        optimizer_epsilon=1e-8,
        mix=FixedMix(0.25),
    )

    calculator = builder.fit(train, evaluation).build(max_score_block_elements=2)
    query_batch = next(iter(evaluation))
    result = calculator.search_batch(query_batch, 3)

    shards = list(calculator.train_store.iter_shards())
    train_vectors = np.concatenate([shard.vectors for shard in shards])
    train_ids = np.concatenate([shard.ids for shard in shards])
    query_vectors = calculator.represent_batch(query_batch)
    scores = query_vectors @ train_vectors.T
    expected = np.stack([np.lexsort((train_ids, -row))[:3] for row in scores])
    np.testing.assert_allclose(result.scores, np.take_along_axis(scores, expected, axis=1))
    np.testing.assert_array_equal(result.ids, train_ids[expected])
    np.testing.assert_allclose(np.linalg.norm(train_vectors, axis=1), 1.0)
    assert builder.state == "built"
    assert builder.train_gram.count == 6
    assert builder.evaluation_gram.count == 3
    with pytest.raises(RuntimeError, match="only be called once"):
        builder.fit(train, evaluation)


def test_builder_rejects_duplicate_custom_ids():
    model = torch.nn.Linear(1, 1, bias=False)
    wrapper = BaseInfluenceModel(model, loss_function=torch.nn.MSELoss(reduction="none"))
    optimizer = torch.optim.Adam(model.parameters())
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    dataset = [(torch.ones((2, 1)), torch.zeros((2, 1)))]
    builder = TrackStarBuilder.from_optimizer(
        wrapper, optimizer, _projection_plan(wrapper), optimizer_epsilon=1e-8
    )

    with pytest.raises(ValueError, match="unique"):
        builder.fit(
            dataset,
            dataset,
            train_id_extractor=lambda _batch, _next_id: np.array([7, 7]),
        )
