# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for FirstOrderInfluenceCalculator with PyTorch backend.
"""
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ConjugateGradientDescentIHVP, KfacIHVP, KfacIHVPFactory
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils.sorted_dict import ORDER
from ..utils_test import (
    assert_close,
    build_loader_torch,
    build_regression_tensors_torch,
    ground_truth_grads_hessian_last_layer_torch,
    make_linear_model_torch,
    set_seed_torch,
)


pytestmark = pytest.mark.pytorch


set_seed = set_seed_torch
make_linear_model = make_linear_model_torch
build_loader = build_loader_torch
build_regression_tensors = build_regression_tensors_torch
ground_truth_grads_hessian_last_layer = ground_truth_grads_hessian_last_layer_torch


def normalize_columns(v):
    """Normalize a matrix column-wise."""
    return v / torch.linalg.norm(v, dim=0, keepdim=True)


def extract_nested_influence_matrix(eval_inf_ds):
    """Extract a full (n_eval, n_train) matrix from nested influence dataset output."""
    influence_values = []
    for _, samples_inf_ds in eval_inf_ds:
        sample_values = []
        for _, inf_values in samples_inf_ds:
            sample_values.append(inf_values)
        influence_values.append(torch.cat(sample_values, dim=1))
    return torch.cat(influence_values, dim=0)


def extract_top_k(top_k_ds):
    """Extract concatenated top-k values/samples from top_k output dataset."""
    top_k_influences = []
    top_k_samples = []
    for _, influence_values, training_samples in top_k_ds:
        top_k_influences.append(influence_values)
        top_k_samples.append(training_samples)
    return torch.cat(top_k_influences, dim=0), torch.cat(top_k_samples, dim=0)


def build_ihvp_objects(influence_model, train_loader):
    """Create Exact and CGD IHVP calculators for parity checks."""
    return [
        (ExactIHVP(influence_model, train_loader), 5e-4),
        (ConjugateGradientDescentIHVP(influence_model, -1, train_loader, n_opt_iters=60), 1e-1),
    ]


def test_compute_influence_vector():
    """Test _compute_influence_vector against analytical ground truth."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=1)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads

    for ihvp_calculator, tolerance in build_ihvp_objects(influence_model, train_loader):
        for normalize in [True, False]:
            influence_calculator = FirstOrderInfluenceCalculator(
                influence_model,
                train_loader,
                ihvp_calculator,
                n_samples_for_hessian=25,
                shuffle_buffer_size=25,
                normalize=normalize,
            )

            inf_vectors = []
            for batch in train_loader:
                batch_inf = influence_calculator._compute_influence_vector(batch)
                assert batch_inf.shape == (5, 2)
                inf_vectors.append(batch_inf)
            inf_vectors = torch.cat(inf_vectors, dim=0)
            assert inf_vectors.shape == (25, 2)

            gt = normalize_columns(gt_inf_vec) if normalize else gt_inf_vec
            assert_close(inf_vectors, gt.T, epsilon=tolerance)


def test_first_order_accepts_ihvp_factory(tmp_path):
    """FirstOrderInfluenceCalculator should accept IHVP factories directly."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(20, seed=11)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    checkpoint_path = str(tmp_path / "kfac_first_order")
    factory = KfacIHVPFactory(damping=1e-3, factors_path=checkpoint_path)

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ihvp_calculator=factory,
    )

    assert isinstance(influence_calculator.ihvp_calculator, KfacIHVP)
    assert (tmp_path / "kfac_first_order" / "metadata.json").exists()


def test_compute_influence_vector_dataset_and_save_load():
    """Test dataset output of compute_influence_vector and save/load behavior."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=2)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads

    for ihvp_calculator, tolerance in build_ihvp_objects(influence_model, train_loader):
        for normalize in [True, False]:
            influence_calculator = FirstOrderInfluenceCalculator(
                influence_model,
                train_loader,
                ihvp_calculator,
                n_samples_for_hessian=25,
                shuffle_buffer_size=25,
                normalize=normalize,
            )

            inf_vect_ds = influence_calculator.compute_influence_vector(train_loader)
            gt = normalize_columns(gt_inf_vec) if normalize else gt_inf_vec

            all_vectors = []
            start_idx = 0
            for (batch_x, batch_y), batch_inf in inf_vect_ds:
                bs = batch_x.shape[0]
                assert_close(batch_x, inputs_train[start_idx:start_idx + bs], epsilon=1e-10)
                assert_close(batch_y, targets_train[start_idx:start_idx + bs], epsilon=1e-10)
                assert_close(batch_inf, gt.T[start_idx:start_idx + bs], epsilon=tolerance)
                all_vectors.append(batch_inf)
                start_idx += bs

            assert_close(torch.cat(all_vectors, dim=0), gt.T, epsilon=tolerance)

    # Save/load test on ExactIHVP path
    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "inf_vector_ds.pt")
        influence_calculator.compute_influence_vector(train_loader, save_influence_vector_ds_path=save_path)
        assert os.path.exists(save_path)

        loaded_inf_vect = influence_calculator._load_dataset(save_path)
        loaded_inf_vect = torch.stack(loaded_inf_vect, dim=0)
        assert loaded_inf_vect.shape == (25, 2)
        assert_close(loaded_inf_vect, gt_inf_vec.T, epsilon=5e-4)


def test_preprocess_sample_to_evaluate():
    """Test _preprocess_samples returns (batch_size, nb_params)."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=3)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    for batch in train_loader:
        preprocess = influence_calculator._preprocess_samples(batch)
        assert preprocess.shape == (5, influence_model.nb_params)


def test_compute_influence_value_from_influence_vector():
    """Test _estimate_influence_value_from_influence_vector against analytical values."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=4)
    inputs_test, targets_test = build_regression_tensors(25, seed=5)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_grads_test, _ = ground_truth_grads_hessian_last_layer(model, inputs_test, targets_test)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    influence_values = []
    for batch in test_loader:
        preproc = influence_calculator._preprocess_samples(batch)
        batch_values = influence_calculator._estimate_influence_value_from_influence_vector(preproc, gt_inf_vec.T)
        assert batch_values.shape == (5, 25)
        influence_values.append(batch_values)
    influence_values = torch.cat(influence_values, dim=0)
    assert influence_values.shape == (25, 25)

    gt_inf_values = gt_grads_test.T @ gt_inf_vec
    assert_close(influence_values, gt_inf_values, epsilon=5e-4)


def test_compute_pairwise_influence_value():
    """Test _compute_influence_value_from_batch against analytical self-influence."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=6)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train

    for ihvp_calculator, tolerance in build_ihvp_objects(influence_model, train_loader):
        for normalize in [True, False]:
            influence_calculator = FirstOrderInfluenceCalculator(
                influence_model,
                train_loader,
                ihvp_calculator,
                n_samples_for_hessian=25,
                shuffle_buffer_size=25,
                normalize=normalize,
            )

            gt = normalize_columns(gt_inf_vec) if normalize else gt_inf_vec
            gt_self = torch.sum(gt_grads_train.T * gt.T, dim=1, keepdim=True)

            influence_values = []
            for batch in train_loader:
                batch_values = influence_calculator._compute_influence_value_from_batch(batch)
                assert batch_values.shape == (5, 1)
                influence_values.append(batch_values)
            influence_values = torch.cat(influence_values, dim=0)
            assert influence_values.shape == (25, 1)

            assert_close(influence_values, gt_self, epsilon=tolerance)


@pytest.mark.parametrize("order", [ORDER.ASCENDING, ORDER.DESCENDING])
@pytest.mark.parametrize("normalize", [True, False])
def test_compute_top_k_from_training_dataset(order, normalize):
    """Test compute_top_k_from_training_dataset returns expected samples and values."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=7)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train
    gt = normalize_columns(gt_inf_vec) if normalize else gt_inf_vec
    gt_self = torch.sum(gt_grads_train.T * gt.T, dim=1)

    if order == ORDER.DESCENDING:
        expected_values, expected_indices = torch.topk(gt_self, k=5)
    else:
        expected_values, expected_indices = torch.topk(-gt_self, k=5)
        expected_values = -expected_values
    expected_samples = inputs_train[expected_indices]

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
        normalize=normalize,
    )

    top_k_samples, top_k_influences = influence_calculator.compute_top_k_from_training_dataset(
        train_loader,
        k=5,
        order=order,
    )
    assert top_k_samples.shape == (5, 1, 3)
    assert top_k_influences.shape == (5,)

    assert_close(top_k_influences, expected_values, epsilon=5e-4)
    assert_close(top_k_samples, expected_samples, epsilon=1e-8)


def test_compute_influence_values_dataset_and_compute_influence_values():
    """Test compute_influence_values dataset API and tensor API consistency."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=8)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train
    gt_self = torch.sum(gt_grads_train.T * gt_inf_vec.T, dim=1, keepdim=True)

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    inf_values_ds = influence_calculator.compute_influence_values(train_loader)
    expected_inf_values = []
    start_idx = 0
    for (batch_x, batch_y), batch_inf in inf_values_ds:
        bs = batch_x.shape[0]
        assert_close(batch_x, inputs_train[start_idx:start_idx + bs], epsilon=1e-10)
        assert_close(batch_y, targets_train[start_idx:start_idx + bs], epsilon=1e-10)
        assert_close(batch_inf, gt_self[start_idx:start_idx + bs], epsilon=5e-4)
        expected_inf_values.append(batch_inf)
        start_idx += bs
    expected_inf_values = torch.cat(expected_inf_values, dim=0)

    computed_inf_values = influence_calculator._compute_influence_values(train_loader)
    assert_close(computed_inf_values, expected_inf_values, epsilon=1e-4)


def test_compute_influence_values_from_tensor():
    """Test _estimate_individual_influence_values_from_batch against analytical matrix values."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=9)
    inputs_test, targets_test = build_regression_tensors(25, seed=10)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_grads_test, _ = ground_truth_grads_hessian_last_layer(model, inputs_test, targets_test)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train

    for ihvp_calculator, tolerance in build_ihvp_objects(influence_model, train_loader):
        for normalize in [True, False]:
            influence_calculator = FirstOrderInfluenceCalculator(
                influence_model,
                train_loader,
                ihvp_calculator,
                n_samples_for_hessian=25,
                shuffle_buffer_size=25,
                normalize=normalize,
            )

            gt = normalize_columns(gt_inf_vec) if normalize else gt_inf_vec
            gt_inf_values = gt_grads_test.T @ gt

            influence_values = []
            for test_batch in test_loader:
                local_values = []
                for train_batch in train_loader:
                    batch_values = influence_calculator._estimate_individual_influence_values_from_batch(
                        train_batch,
                        test_batch,
                    )
                    assert batch_values.shape == (5, 5)
                    local_values.append(batch_values)
                influence_values.append(torch.cat(local_values, dim=1))
            influence_values = torch.cat(influence_values, dim=0)

            assert influence_values.shape == (25, 25)
            assert_close(influence_values, gt_inf_values, epsilon=tolerance)


def test_compute_inf_values_with_inf_vect_dataset():
    """Test _estimate_inf_values_with_inf_vect_dataset with a manually built IHVP dataset."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=11)
    inputs_test, targets_test = build_regression_tensors(25, seed=12)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_grads_test, _ = ground_truth_grads_hessian_last_layer(model, inputs_test, targets_test)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train
    gt_inf_values = gt_grads_test.T @ gt_inf_vec

    # Build a PyTorch-style influence vector dataset: [((batch_x, batch_y), batch_inf_vector), ...]
    gt_ihvp_dataset = []
    start_idx = 0
    for batch_x, batch_y in train_loader:
        bs = batch_x.shape[0]
        batch_inf = gt_inf_vec.T[start_idx:start_idx + bs]
        gt_ihvp_dataset.append(((batch_x, batch_y), batch_inf))
        start_idx += bs

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    influence_values = []
    for samples_to_evaluate in test_loader:
        _, samples_inf_values_ds = influence_calculator._estimate_inf_values_with_inf_vect_dataset(
            gt_ihvp_dataset,
            samples_to_evaluate,
        )
        sample_values = []
        for _, inf_values in samples_inf_values_ds:
            assert inf_values.shape == (5, 5)
            sample_values.append(inf_values)
        sample_values = torch.cat(sample_values, dim=1)
        assert sample_values.shape == (5, 25)
        influence_values.append(sample_values)
    influence_values = torch.cat(influence_values, dim=0)

    assert influence_values.shape == (25, 25)
    assert_close(influence_values, gt_inf_values, epsilon=5e-4)


def test_compute_influence_values_for_dataset_to_evaluate_and_save_load():
    """Test estimate_influence_values_in_batches and its save/load paths on PyTorch."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=13)
    inputs_test, targets_test = build_regression_tensors(25, seed=14)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_grads_test, _ = ground_truth_grads_hessian_last_layer(model, inputs_test, targets_test)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train
    gt_inf_values = gt_grads_test.T @ gt_inf_vec

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    eval_inf_ds = influence_calculator.estimate_influence_values_in_batches(test_loader, train_loader)
    influence_values = extract_nested_influence_matrix(eval_inf_ds)
    assert influence_values.shape == (25, 25)
    assert_close(influence_values, gt_inf_values, epsilon=5e-4)

    with tempfile.TemporaryDirectory() as tmp_dir:
        inf_vect_path = os.path.join(tmp_dir, "influence_vector_ds.pt")
        inf_values_dir = os.path.join(tmp_dir, "influence_values_ds")

        influence_calculator.estimate_influence_values_in_batches(
            test_loader,
            train_loader,
            save_influence_vector_path=inf_vect_path,
            save_influence_value_path=inf_values_dir,
        )
        assert os.path.exists(inf_vect_path)
        assert os.path.isdir(inf_values_dir)

        saved_batches = sorted(os.listdir(inf_values_dir))
        assert len(saved_batches) == len(list(test_loader))

        loaded_influence_values = []
        for batch_file in saved_batches:
            batch_inf_ds = influence_calculator._load_dataset(os.path.join(inf_values_dir, batch_file))
            sample_values = []
            for _, inf_values in batch_inf_ds:
                sample_values.append(inf_values)
            loaded_influence_values.append(torch.cat(sample_values, dim=1))
        loaded_influence_values = torch.cat(loaded_influence_values, dim=0)
        assert_close(loaded_influence_values, gt_inf_values, epsilon=5e-4)

        loaded_inf_vect_ds = influence_calculator.estimate_influence_values_in_batches(
            test_loader,
            train_loader,
            load_influence_vector_path=inf_vect_path,
        )
        loaded_matrix = extract_nested_influence_matrix(loaded_inf_vect_ds)
        assert_close(loaded_matrix, gt_inf_values, epsilon=5e-4)


def _make_small_linear_case():
    """Build a tiny linear regression case for hook tests."""
    model = nn.Sequential(nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.7, -0.4]], dtype=torch.float32))
        model[0].bias.copy_(torch.tensor([0.2], dtype=torch.float32))

    inputs = torch.tensor(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.1], [0.5], [1.2], [1.6]], dtype=torch.float32)
    loader = build_loader(inputs, targets, batch_size=2)
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=nn.MSELoss(reduction="none"))
    calculator = FirstOrderInfluenceCalculator(
        influence_model,
        loader,
        ExactIHVP(influence_model, loader),
        n_samples_for_hessian=4,
        shuffle_buffer_size=4,
    )
    return loader, calculator


def test_evaluation_representation_provider_overrides_query_gradients():
    """Custom evaluation representations should override the default query Jacobian path."""
    set_seed(123)
    loader, calculator = _make_small_linear_case()

    def zero_provider(model, batch):
        return torch.zeros((batch[0].shape[0], model.nb_params), dtype=batch[0].dtype)

    inf_ds = calculator.estimate_influence_values_in_batches(
        loader,
        loader,
        evaluation_representation_provider=zero_provider,
    )
    influence_matrix = extract_nested_influence_matrix(inf_ds)
    assert torch.allclose(influence_matrix, torch.zeros_like(influence_matrix))


def test_training_payload_extractor_controls_top_k_payloads():
    """Top-k should return payload tensors produced by the custom extractor."""
    set_seed(123)
    loader, calculator = _make_small_linear_case()

    top_k_ds = calculator.top_k(
        loader,
        loader,
        k=1,
        training_payload_extractor=lambda batch: batch[0][:, 0].round().to(dtype=torch.int64),
    )

    payload_batches = []
    for _, _, training_payload in top_k_ds:
        payload_batches.append(training_payload)
    payloads = torch.cat(payload_batches, dim=0)

    assert payloads.dtype == torch.int64
    assert payloads.ndim == 2
    assert set(payloads.reshape(-1).tolist()).issubset({0, 1, 2, 3})


def test_default_influence_scores_keep_full_training_batches():
    """Influence score datasets should preserve the full training batch payload by default."""
    set_seed(123)
    loader, calculator = _make_small_linear_case()

    inf_ds = calculator.estimate_influence_values_in_batches(loader, loader)
    _, score_ds = next(iter(inf_ds))
    train_batch, influence_values = next(iter(score_ds))

    assert isinstance(train_batch, tuple)
    assert len(train_batch) == 2
    assert train_batch[0].shape == (2, 2)
    assert train_batch[1].shape == (2, 1)
    assert influence_values.shape == (2, 2)


@pytest.mark.parametrize("order", [ORDER.ASCENDING, ORDER.DESCENDING])
def test_top_k_dataset(order):
    """Test top_k dataset output against analytical top-k values/samples."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=15)
    inputs_test, targets_test = build_regression_tensors(25, seed=16)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_grads_test, _ = ground_truth_grads_hessian_last_layer(model, inputs_test, targets_test)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train
    gt_inf_values = gt_grads_test.T @ gt_inf_vec

    if order == ORDER.DESCENDING:
        gt_top_k = torch.topk(gt_inf_values, k=3, dim=1)
        gt_top_k_values = gt_top_k.values
    else:
        gt_top_k = torch.topk(-gt_inf_values, k=3, dim=1)
        gt_top_k_values = -gt_top_k.values
    gt_top_k_samples = inputs_train[gt_top_k.indices]

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    top_dataset_ds = influence_calculator.top_k(
        test_loader,
        train_loader,
        k=3,
        order=order,
        d_type=torch.float64,
    )
    top_k_influences, top_k_samples = extract_top_k(top_dataset_ds)
    assert top_k_influences.shape == (25, 3)
    assert top_k_samples.shape == (25, 3, 1, 3)

    assert_close(top_k_influences, gt_top_k_values, epsilon=5e-4)
    assert_close(top_k_samples, gt_top_k_samples, epsilon=1e-8)


def test_top_k_dataset_save_load():
    """Test top_k save/load paths for PyTorch backend."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=17)
    inputs_test, targets_test = build_regression_tensors(25, seed=18)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=5)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_grads_test, _ = ground_truth_grads_hessian_last_layer(model, inputs_test, targets_test)
    gt_inf_vec = torch.linalg.pinv(gt_hessian) @ gt_grads_train
    gt_inf_values = gt_grads_test.T @ gt_inf_vec

    gt_top_k = torch.topk(gt_inf_values, k=3, dim=1)
    gt_top_k_values = gt_top_k.values
    gt_top_k_samples = inputs_train[gt_top_k.indices]

    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        inf_vect_path = os.path.join(tmp_dir, "influence_vector_ds.pt")
        top_k_path = os.path.join(tmp_dir, "top_k_ds.pt")

        influence_calculator.top_k(
            test_loader,
            train_loader,
            k=3,
            save_influence_vector_ds_path=inf_vect_path,
            save_top_k_ds_path=top_k_path,
            d_type=torch.float64,
        )
        assert os.path.exists(inf_vect_path)
        assert os.path.exists(top_k_path)

        load_ds = influence_calculator._load_dataset(top_k_path)
        top_k_influences, top_k_samples = extract_top_k(load_ds)
        assert_close(top_k_influences, gt_top_k_values, epsilon=5e-4)
        assert_close(top_k_samples, gt_top_k_samples, epsilon=1e-8)

        other_load_ds = influence_calculator.top_k(
            test_loader,
            train_loader,
            k=3,
            load_influence_vector_ds_path=inf_vect_path,
            d_type=torch.float64,
        )
        top_k_influences, top_k_samples = extract_top_k(other_load_ds)
        assert_close(top_k_influences, gt_top_k_values, epsilon=5e-4)
        assert_close(top_k_samples, gt_top_k_samples, epsilon=1e-8)


def test_compute_influence_group():
    """Test compute_influence_vector_group against analytical group influence vector."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=19)
    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    group_loader = build_loader(inputs_train, targets_train, batch_size=25)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    reduced_grads = torch.sum(gt_grads_train, dim=1, keepdim=True)
    gt_group_inf = torch.linalg.pinv(gt_hessian) @ reduced_grads

    ihvp_objects = [
        (ExactIHVP(influence_model, train_loader), 5e-4),
        (ConjugateGradientDescentIHVP(influence_model, -1, train_loader, n_opt_iters=60), 1e-1),
    ]

    for ihvp_calculator, tolerance in ihvp_objects:
        influence_calculator = FirstOrderInfluenceCalculator(
            influence_model,
            train_loader,
            ihvp_calculator,
            n_samples_for_hessian=25,
            shuffle_buffer_size=25,
        )
        influence_group = influence_calculator.compute_influence_vector_group(group_loader)
        assert influence_group.shape == (1, 2)
        assert_close(influence_group, gt_group_inf.T, epsilon=tolerance)


def test_compute_influence_values_group():
    """Test estimate_influence_values_group against analytical group Cook's distance."""
    set_seed(0)
    model = make_linear_model()
    loss_fn = nn.MSELoss(reduction="none")

    inputs_train, targets_train = build_regression_tensors(25, seed=20)
    inputs_test, targets_test = build_regression_tensors(25, seed=21)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=5)
    group_train = build_loader(inputs_train, targets_train, batch_size=25)
    group_test = build_loader(inputs_test, targets_test, batch_size=25)

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
    gt_grads_train, gt_hessian = ground_truth_grads_hessian_last_layer(model, inputs_train, targets_train)
    gt_grads_test, _ = ground_truth_grads_hessian_last_layer(model, inputs_test, targets_test)

    reduced_train = torch.sum(gt_grads_train, dim=1, keepdim=True)
    reduced_test = torch.sum(gt_grads_test, dim=1, keepdim=True)
    gt_group_values = reduced_test.T @ (torch.linalg.pinv(gt_hessian) @ reduced_train)

    ihvp_objects = [
        (ExactIHVP(influence_model, train_loader), 5e-4),
        (ConjugateGradientDescentIHVP(influence_model, -1, train_loader, n_opt_iters=60), 1e-1),
    ]

    for ihvp_calculator, tolerance in ihvp_objects:
        influence_calculator = FirstOrderInfluenceCalculator(
            influence_model,
            train_loader,
            ihvp_calculator,
            n_samples_for_hessian=25,
            shuffle_buffer_size=25,
        )
        influence_values = influence_calculator.estimate_influence_values_group(group_train, group_test)
        assert influence_values.shape == (1, 1)
        assert_close(influence_values, gt_group_values, epsilon=tolerance)


def test_cnn_shapes():
    """Smoke test all main methods on a CNN-like PyTorch model."""
    set_seed(0)

    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=2, dtype=torch.float64),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 10, bias=True, dtype=torch.float64),
        nn.Linear(10, 10, bias=False, dtype=torch.float64),
    )
    nn.init.ones_(model[3].weight)
    nn.init.zeros_(model[3].bias)
    nn.init.ones_(model[4].weight)

    def mse_per_sample(predictions, targets):
        return torch.mean((predictions - targets) ** 2, dim=1)

    inputs_train = torch.randn((50, 3, 5, 5), dtype=torch.float64)
    targets_train_idx = torch.randint(0, 10, (50,))
    targets_train = torch.nn.functional.one_hot(targets_train_idx, num_classes=10).to(torch.float64)

    inputs_test = torch.randn((60, 3, 5, 5), dtype=torch.float64)
    targets_test_idx = torch.randint(0, 10, (60,))
    targets_test = torch.nn.functional.one_hot(targets_test_idx, num_classes=10).to(torch.float64)

    train_loader = build_loader(inputs_train, targets_train, batch_size=5)
    test_loader = build_loader(inputs_test, targets_test, batch_size=10)

    influence_model = InfluenceModel(model, loss_function=mse_per_sample)
    influence_calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_loader,
        ExactIHVP(influence_model, train_loader),
        n_samples_for_hessian=25,
        shuffle_buffer_size=25,
    )

    # compute_influence_values_from_tensor
    test_batch = next(iter(test_loader))
    train_batch = next(iter(train_loader))
    inf_val_from_tensor = influence_calculator._estimate_individual_influence_values_from_batch(
        train_samples=train_batch,
        samples_to_evaluate=test_batch,
    )
    assert inf_val_from_tensor.shape == (10, 5)

    # compute_influence_values_for_dataset_to_evaluate
    inf_val_dataset = influence_calculator.estimate_influence_values_in_batches(test_loader, train_loader)
    batch_samples, batched_associated_ds = next(iter(inf_val_dataset))
    assert batch_samples[0].shape == (10, 3, 5, 5)
    assert batch_samples[1].shape == (10, 10)
    (batch_x, batch_y), batch_inf = next(iter(batched_associated_ds))
    assert batch_x.shape == (5, 3, 5, 5)
    assert batch_y.shape == (5, 10)
    assert batch_inf.shape == (10, 5)

    # compute_influence_vector_dataset
    inf_vect_ds = influence_calculator.compute_influence_vector(train_loader)
    (batch_x, batch_y), inf_vect = next(iter(inf_vect_ds))
    assert batch_x.shape == (5, 3, 5, 5)
    assert batch_y.shape == (5, 10)
    assert inf_vect.shape == (5, influence_model.nb_params)

    # compute_influence_values_dataset and _compute_influence_values
    inf_values_dataset = influence_calculator.compute_influence_values(train_loader)
    (batch_x, batch_y), batch_inf = next(iter(inf_values_dataset))
    assert batch_x.shape == (5, 3, 5, 5)
    assert batch_y.shape == (5, 10)
    assert batch_inf.shape == (5, 1)

    inf_values = influence_calculator._compute_influence_values(train_loader)
    assert inf_values.shape == (50, 1)

    # compute_top_k_from_training_dataset and top_k_dataset
    top_k_train_samples, top_k_inf_val = influence_calculator.compute_top_k_from_training_dataset(train_loader, k=3)
    assert top_k_train_samples.shape == (3, 3, 5, 5)
    assert top_k_inf_val.shape == (3,)

    top_k_dataset = influence_calculator.top_k(test_loader, train_loader, k=3, d_type=torch.float64)
    (batch_eval_x, batch_eval_y), k_inf_val, k_training_samples = next(iter(top_k_dataset))
    assert batch_eval_x.shape == (10, 3, 5, 5)
    assert batch_eval_y.shape == (10, 10)
    assert k_inf_val.shape == (10, 3)
    assert k_training_samples.shape == (10, 3, 3, 5, 5)

    # Group methods
    train_group = build_loader(inputs_train, targets_train, batch_size=50)
    eval_group = build_loader(inputs_test[:50], targets_test[:50], batch_size=50)
    influence_group = influence_calculator.compute_influence_vector_group(train_group)
    assert influence_group.shape == (1, influence_model.nb_params)
    influence_group_values = influence_calculator.estimate_influence_values_group(train_group, eval_group)
    assert influence_group_values.shape == (1, 1)
