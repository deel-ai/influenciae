# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from types import SimpleNamespace

import numpy as np
import pytest

from deel.influenciae.utils.model_surgery import (
    _normalize_shape,
    _prepare_feature_dataset,
    compute_lje_alpha,
    create_surrogate_linear_model,
    perturb_head_single_sgd_step,
    split_batch_inputs_targets,
    train_surrogate_linear_model,
)


class _RecordingCreateLinearBackend:
    def __init__(self, feature_extractor, original_head, feature_shape, head_shape):
        self.feature_extractor = feature_extractor
        self.original_head = original_head
        self.feature_shape = feature_shape
        self.head_shape = head_shape
        self.linear_layer = object()
        self.reference_weight = np.arange(5, dtype=np.float64)
        self.created = None

    def get_layers(self, _model):
        return [self.linear_layer]

    @staticmethod
    def get_layer_io_features(_layer):
        return 5, 1

    def get_output_shape(self, model):
        if model is self.feature_extractor:
            return self.feature_shape
        if model is self.original_head:
            return self.head_shape
        raise AssertionError("Unexpected model passed to get_output_shape")

    def get_model_weights(self, model, layers=None):
        assert model is self.original_head
        assert layers == [self.linear_layer]
        return [self.reference_weight]

    def create_linear_model(self, **kwargs):
        self.created = kwargs
        return kwargs


class _DatasetBackend:
    def __init__(self, framework="unsupported", empty_batch_size=1):
        self.framework = framework
        self.empty_batch_size = empty_batch_size
        self.shuffle_calls = []
        self.take_calls = []

    def get_dataset_batch_size(self, dataset):
        if not dataset:
            return self.empty_batch_size
        return len(dataset[0][-1])

    def shuffle_dataset(self, dataset, buffer_size):
        self.shuffle_calls.append(buffer_size)
        return list(reversed(dataset))

    def take_dataset(self, dataset, n_batches):
        self.take_calls.append(n_batches)
        return list(dataset[:n_batches])

    @staticmethod
    def forward(model, inputs):
        return model(inputs)

    @staticmethod
    def concat(batches, axis=0):
        return np.concatenate(batches, axis=axis)

    @staticmethod
    def create_dataset_from_tensor_slices(tensors, batch_size):
        features, targets = tensors
        return [
            (features[idx:idx + batch_size], targets[idx:idx + batch_size])
            for idx in range(0, len(features), batch_size)
        ]

    @staticmethod
    def get_dataset_size(dataset):
        return sum(len(batch[-1]) for batch in dataset)

    @staticmethod
    def clone_model(model):
        return {"clone_of": model}


class _UnsupportedLJEBackend:
    framework = "unsupported"

    def __init__(self):
        self.weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    def get_model_weights(self, _model):
        return [self.weight]

    @staticmethod
    def forward(_model, inputs):
        return np.zeros((inputs.shape[0], 3), dtype=inputs.dtype)

    @staticmethod
    def normalize_binary_targets(targets, _logits):
        return targets

    @staticmethod
    def compute_jacobian(_model, _weights, _loss_function, inputs, _targets):
        return np.ones((inputs.shape[0], 2, 3), dtype=inputs.dtype)

    @staticmethod
    def reshape(value, shape):
        return np.reshape(value, shape)

    @staticmethod
    def tensor_shape(value):
        return value.shape

    @staticmethod
    def get_batch_size(value):
        return value.shape[0]

    @staticmethod
    def get_linear_weight_axes():
        return 0, 1

    @staticmethod
    def get_dtype(value):
        return value.dtype

    @staticmethod
    def cast(value, dtype):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def expand_dims(value, axis):
        return np.expand_dims(value, axis=axis)

    @staticmethod
    def ones_like(value):
        return np.ones_like(value)

    @staticmethod
    def repeat(value, repeats, axis):
        return np.repeat(value, repeats, axis=axis)


@pytest.mark.backend_agnostic
def test_split_batch_inputs_targets_supports_single_and_multi_input_batches():
    x_batch = np.array([[1.0], [2.0]], dtype=np.float32)
    y_batch = np.array([[0.0], [1.0]], dtype=np.float32)

    inputs, targets = split_batch_inputs_targets((x_batch, y_batch))
    assert inputs is x_batch
    assert targets is y_batch

    x_left = np.array([[1.0], [2.0]], dtype=np.float32)
    x_right = np.array([[10.0], [20.0]], dtype=np.float32)
    multi_inputs, multi_targets = split_batch_inputs_targets((x_left, x_right, y_batch))
    assert multi_inputs == (x_left, x_right)
    assert multi_targets is y_batch


@pytest.mark.backend_agnostic
def test_normalize_shape_and_surrogate_creation_cover_shape_and_dtype_fallbacks():
    assert _normalize_shape([(None, 4)]) == (None, 4)

    with pytest.raises(ValueError, match="non-empty shape list"):
        _normalize_shape([])

    feature_extractor = SimpleNamespace(compute_dtype=None, dtype="float64")
    original_head = object()
    backend = _RecordingCreateLinearBackend(
        feature_extractor=feature_extractor,
        original_head=original_head,
        feature_shape=[(None, None)],
        head_shape=[(None,)],
    )

    surrogate = create_surrogate_linear_model(
        backend=backend,
        feature_extractor=feature_extractor,
        original_head=original_head,
        lambda_regularization=0.25,
    )

    assert surrogate["input_shape"] == (5,)
    assert surrogate["out_features"] == 1
    assert surrogate["use_bias"] is False
    assert surrogate["l2_regularization"] == 0.25
    assert surrogate["dtype"] == "float64"
    assert np.array_equal(surrogate["reference_weight"], backend.reference_weight)


@pytest.mark.backend_agnostic
def test_prepare_feature_dataset_shuffles_takes_and_materializes_feature_batches():
    batch_one = (
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([[10.0], [20.0]], dtype=np.float32),
        np.array([[0.0], [1.0]], dtype=np.float32),
    )
    batch_two = (
        np.array([[3.0], [4.0]], dtype=np.float32),
        np.array([[30.0], [40.0]], dtype=np.float32),
        np.array([[2.0], [3.0]], dtype=np.float32),
    )
    dataset = [batch_one, batch_two]
    backend = _DatasetBackend()

    feature_dataset = _prepare_feature_dataset(
        backend=backend,
        feature_extractor=lambda inputs: inputs[0] + inputs[1],
        dataset=dataset,
        n_samples_for_hessian=3,
        shuffle_buffer_size=13,
    )

    assert backend.shuffle_calls == [13]
    assert backend.take_calls == [1]
    assert len(feature_dataset) == 1
    assert np.array_equal(feature_dataset[0][0], np.array([[33.0], [44.0]], dtype=np.float32))
    assert np.array_equal(feature_dataset[0][1], batch_two[-1])


@pytest.mark.backend_agnostic
def test_prepare_feature_dataset_raises_on_empty_collection():
    backend = _DatasetBackend(empty_batch_size=4)

    with pytest.raises(ValueError, match="Dataset used for Hessian estimation is empty"):
        _prepare_feature_dataset(
            backend=backend,
            feature_extractor=lambda inputs: inputs,
            dataset=[],
            n_samples_for_hessian=None,
            shuffle_buffer_size=5,
        )


@pytest.mark.backend_agnostic
def test_wrapper_functions_raise_for_unsupported_backends():
    dataset = [
        (
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[0.0], [1.0]], dtype=np.float32),
        )
    ]
    backend = _DatasetBackend()

    with pytest.raises(ValueError, match="Unsupported backend framework"):
        train_surrogate_linear_model(
            backend=backend,
            surrogate_model=object(),
            feature_extractor=lambda inputs: inputs,
            original_head=lambda inputs: inputs,
            train_set=dataset,
            loss_function=lambda _y_true, _y_pred: None,
            scaling_factor=0.1,
            epochs=1,
        )

    with pytest.raises(ValueError, match="Unsupported backend framework"):
        perturb_head_single_sgd_step(
            backend=backend,
            original_head="head",
            feature_extractor=lambda inputs: inputs,
            dataset=dataset,
            loss_function=lambda _y_true, _y_pred: None,
            n_samples_for_hessian=None,
            shuffle_buffer_size=5,
        )

    lje_backend = _UnsupportedLJEBackend()
    with pytest.raises(ValueError, match="Unsupported backend framework"):
        compute_lje_alpha(
            backend=lje_backend,
            perturbed_head=object(),
            ihvp_calculator=object(),
            loss_function=lambda _y_true, _y_pred: None,
            z_batch=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            y_batch=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
            epsilon=1e-5,
        )


@pytest.mark.tensorflow
def test_tensorflow_perturb_head_returns_early_without_trainable_variables():
    import tensorflow as tf

    from deel.influenciae.common import get_backend_for_model
    from deel.influenciae.utils._model_surgery_tf import perturb_head_single_sgd_step_tensorflow

    perturbed_head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Lambda(lambda inputs: inputs),
    ])
    _ = perturbed_head(tf.zeros((1, 2), dtype=tf.float32))
    backend = get_backend_for_model(perturbed_head)
    feature_dataset = tf.data.Dataset.from_tensor_slices((
        tf.ones((2, 2), dtype=tf.float32),
        tf.zeros((2, 2), dtype=tf.float32),
    )).batch(1)

    result = perturb_head_single_sgd_step_tensorflow(
        backend=backend,
        perturbed_head=perturbed_head,
        feature_extractor=SimpleNamespace(output_shape=(None, 2)),
        feature_dataset=feature_dataset,
        loss_function=lambda y_true, y_pred: tf.reduce_sum(tf.square(y_true - y_pred), axis=1),
    )

    assert result is perturbed_head


@pytest.mark.tensorflow
def test_tensorflow_perturb_head_raises_when_a_gradient_is_none():
    import tensorflow as tf

    from deel.influenciae.common import get_backend_for_model
    from deel.influenciae.utils._model_surgery_tf import perturb_head_single_sgd_step_tensorflow

    class UnusedWeightLayer(tf.keras.layers.Layer):
        def build(self, _input_shape):
            self.unused = self.add_weight(name="unused", shape=(1,), initializer="ones", trainable=True)

        def call(self, inputs):
            return inputs

    perturbed_head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        UnusedWeightLayer(),
    ])
    _ = perturbed_head(tf.zeros((1, 2), dtype=tf.float32))
    backend = get_backend_for_model(perturbed_head)
    feature_dataset = tf.data.Dataset.from_tensor_slices((
        tf.ones((2, 2), dtype=tf.float32),
        tf.zeros((2, 2), dtype=tf.float32),
    )).batch(1)

    with pytest.raises(ValueError, match="Gradient is None"):
        perturb_head_single_sgd_step_tensorflow(
            backend=backend,
            perturbed_head=perturbed_head,
            feature_extractor=SimpleNamespace(output_shape=(None, 2)),
            feature_dataset=feature_dataset,
            loss_function=lambda y_true, y_pred: tf.reduce_sum(tf.square(y_true - y_pred), axis=1),
        )


@pytest.mark.pytorch
def test_move_to_reference_device_pytorch_preserves_nested_structure():
    import torch

    from deel.influenciae.utils._model_surgery_pytorch import _move_to_reference_device

    reference_parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    nested = (
        torch.ones(2, dtype=torch.float32),
        [torch.zeros(3, dtype=torch.float32), (torch.full((1,), 2.0, dtype=torch.float32),)],
        "keep-me",
    )

    moved = _move_to_reference_device(nested, reference_parameter)

    assert isinstance(moved, tuple)
    assert isinstance(moved[1], list)
    assert isinstance(moved[1][1], tuple)
    assert moved[0].device == reference_parameter.device
    assert moved[1][0].device == reference_parameter.device
    assert moved[1][1][0].device == reference_parameter.device
    assert moved[2] == "keep-me"


@pytest.mark.pytorch
def test_pytorch_surrogate_training_skips_nan_loss_batches(monkeypatch):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from deel.influenciae.common import get_backend_for_model
    from deel.influenciae.utils import _model_surgery_pytorch

    class DummyOptimizer:
        last_instance = None

        def __init__(self, params, batches_per_epoch, scaling_factor):
            _ = list(params)
            self.batches_per_epoch = batches_per_epoch
            self.scaling_factor = scaling_factor
            self.zero_grad_calls = 0
            self.step_calls = 0
            DummyOptimizer.last_instance = self

        def zero_grad(self):
            self.zero_grad_calls += 1

        def step(self, **_kwargs):
            self.step_calls += 1

    class NaNHead(nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0], 1), float("nan"), dtype=inputs.dtype, device=inputs.device)

    monkeypatch.setattr(_model_surgery_pytorch, "BacktrackingLineSearchPyTorch", DummyOptimizer)

    surrogate_model = nn.Linear(2, 1, bias=False)
    backend = get_backend_for_model(surrogate_model)
    train_set = DataLoader(
        TensorDataset(torch.ones((4, 2), dtype=torch.float32), torch.zeros((4, 1), dtype=torch.float32)),
        batch_size=2,
        shuffle=False,
    )

    trained = _model_surgery_pytorch.train_surrogate_linear_model_pytorch(
        backend=backend,
        surrogate_model=surrogate_model,
        feature_extractor=nn.Identity(),
        original_head=NaNHead(),
        train_set=train_set,
        scaling_factor=0.1,
        epochs=1,
        batches_per_epoch=2,
    )

    assert trained is surrogate_model
    assert trained.training is False
    assert DummyOptimizer.last_instance.zero_grad_calls == 2
    assert DummyOptimizer.last_instance.step_calls == 0


@pytest.mark.pytorch
def test_pytorch_surrogate_training_skips_batches_with_none_first_gradient(monkeypatch):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from deel.influenciae.common import get_backend_for_model
    from deel.influenciae.utils import _model_surgery_pytorch

    class DummyOptimizer:
        last_instance = None

        def __init__(self, params, batches_per_epoch, scaling_factor):
            _ = list(params)
            self.batches_per_epoch = batches_per_epoch
            self.scaling_factor = scaling_factor
            self.zero_grad_calls = 0
            self.step_calls = 0
            DummyOptimizer.last_instance = self

        def zero_grad(self):
            self.zero_grad_calls += 1

        def step(self, **_kwargs):
            self.step_calls += 1

    class SurrogateWithUnusedFirstParameter(nn.Module):
        def __init__(self):
            super().__init__()
            self.unused = nn.Parameter(torch.ones((1,), dtype=torch.float32))
            self.weight = nn.Parameter(torch.ones((1, 2), dtype=torch.float32))
            self.losses = []

        def forward(self, inputs):
            return inputs @ self.weight.t()

    class ZeroHead(nn.Module):
        def forward(self, inputs):
            return torch.zeros((inputs.shape[0], 1), dtype=inputs.dtype, device=inputs.device)

    monkeypatch.setattr(_model_surgery_pytorch, "BacktrackingLineSearchPyTorch", DummyOptimizer)

    surrogate_model = SurrogateWithUnusedFirstParameter()
    backend = get_backend_for_model(surrogate_model)
    train_set = DataLoader(
        TensorDataset(torch.ones((4, 2), dtype=torch.float32), torch.zeros((4, 1), dtype=torch.float32)),
        batch_size=2,
        shuffle=False,
    )
    initial_weight = surrogate_model.weight.detach().clone()

    trained = _model_surgery_pytorch.train_surrogate_linear_model_pytorch(
        backend=backend,
        surrogate_model=surrogate_model,
        feature_extractor=nn.Identity(),
        original_head=ZeroHead(),
        train_set=train_set,
        scaling_factor=0.1,
        epochs=1,
        batches_per_epoch=2,
    )

    assert trained is surrogate_model
    assert trained.training is False
    assert DummyOptimizer.last_instance.zero_grad_calls == 4
    assert DummyOptimizer.last_instance.step_calls == 0
    assert torch.equal(trained.weight.detach(), initial_weight)
