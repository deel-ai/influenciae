# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for lazy PyTorch dataset wrappers.
"""
import importlib
import random

import pytest
import torch


pytestmark = pytest.mark.pytorch


@pytest.fixture
def lazy_module():
    """Import lazy dataset module."""
    return importlib.import_module("deel.influenciae.common.pytorch_lazy_dataset")


def test_ensure_reiterable_materializes_iterator(lazy_module):
    """One-pass iterators should be materialized and warned."""

    def dataset_iter():
        for idx in range(3):
            yield idx

    with pytest.warns(RuntimeWarning, match="one-pass iterator"):
        result = lazy_module.ensure_reiterable(dataset_iter(), context="test")

    assert result == [0, 1, 2]

def test_ensure_reiterable_keeps_reiterable_inputs(lazy_module):
    """Re-iterable datasets should be returned unchanged."""
    data = [1, 2, 3]
    assert lazy_module.ensure_reiterable(data) is data

def test_safe_len_and_is_iterator(lazy_module):
    """Iterator and len helpers should behave as expected."""
    iterator = iter([1, 2])

    assert lazy_module._is_iterator(iterator)
    assert not lazy_module._is_iterator([1, 2])
    assert lazy_module._safe_len([1, 2, 3]) == 3
    assert lazy_module._safe_len(iterator) is None


def test_mapped_dataset_is_lazy_and_reiterable(lazy_module):
    """MappedDataset should evaluate lazily and support multiple passes."""
    call_count = {"value": 0}

    def map_fn(value):
        call_count["value"] += 1
        return value * 2

    mapped = lazy_module.MappedDataset([1, 2, 3], map_fn)

    assert len(mapped) == 3
    assert call_count["value"] == 0

    assert list(mapped) == [2, 4, 6]
    assert call_count["value"] == 3

    assert list(mapped) == [2, 4, 6]
    assert call_count["value"] == 6

def test_mapped_dataset_materializes_one_pass_iterator(lazy_module):
    """MappedDataset should warn and materialize one-pass inputs."""

    def dataset_iter():
        for idx in range(2):
            yield idx

    with pytest.warns(RuntimeWarning, match="one-pass iterator"):
        mapped = lazy_module.MappedDataset(dataset_iter(), lambda value: value + 1)

    assert list(mapped) == [1, 2]
    assert list(mapped) == [1, 2]

def test_cached_dataset_is_stable(lazy_module):
    """CachedDataset should preserve items over repeated access."""
    cached = lazy_module.CachedDataset([10, 20, 30])

    assert list(cached) == [10, 20, 30]
    assert list(cached) == [10, 20, 30]
    assert cached[1] == 20


def test_zipped_dataset_values_and_length(lazy_module):
    """ZippedDataset should zip lazily and report min length."""
    zipped = lazy_module.ZippedDataset([1, 2, 3], ["a", "b"])

    assert len(zipped) == 2
    assert list(zipped) == [(1, "a"), (2, "b")]

def test_batched_dataset_tuple_collation_and_tail_batch(lazy_module):
    """BatchedDataset should collate tuple items and keep last partial batch."""
    samples = [
        (torch.tensor([1.0, 2.0]), torch.tensor([10.0])),
        (torch.tensor([3.0, 4.0]), torch.tensor([20.0])),
        (torch.tensor([5.0, 6.0]), torch.tensor([30.0])),
    ]

    batched = lazy_module.BatchedDataset(samples, batch_size=2)
    batches = list(batched)

    assert len(batched) == 2
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 2)
    assert batches[0][1].shape == (2, 1)
    assert batches[1][0].shape == (1, 2)
    assert batches[1][1].shape == (1, 1)
    assert torch.equal(batches[0][0][0], torch.tensor([1.0, 2.0]))

def test_batched_dataset_list_collation(lazy_module):
    """BatchedDataset should collate list samples preserving list structure."""
    samples = [
        [torch.tensor([1.0]), torch.tensor([2.0])],
        [torch.tensor([3.0]), torch.tensor([4.0])],
    ]

    batched = lazy_module.BatchedDataset(samples, batch_size=2)
    batch = list(batched)[0]

    assert isinstance(batch, list)
    assert len(batch) == 2
    assert batch[0].shape == (2, 1)
    assert batch[1].shape == (2, 1)

def test_batched_dataset_invalid_batch_size(lazy_module):
    """BatchedDataset should reject non-positive batch sizes."""
    with pytest.raises(ValueError):
        lazy_module.BatchedDataset([1, 2], batch_size=0)

def test_unbatched_dataset_tuple_and_tensor(lazy_module):
    """UnbatchedDataset should unroll tuple and tensor batches correctly."""
    tuple_batched = [
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[10.0], [20.0]]))
    ]
    unbatched_tuple = list(lazy_module.UnbatchedDataset(tuple_batched))

    assert len(unbatched_tuple) == 2
    assert torch.equal(unbatched_tuple[0][0], torch.tensor([1.0]))
    assert torch.equal(unbatched_tuple[1][1], torch.tensor([20.0]))

    tensor_batched = [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]
    unbatched_tensor = list(lazy_module.UnbatchedDataset(tensor_batched))

    assert len(unbatched_tensor) == 2
    assert torch.equal(unbatched_tensor[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(unbatched_tensor[1], torch.tensor([3.0, 4.0]))

def test_unbatched_dataset_passes_through_non_batched_items(lazy_module):
    """Items without a batch dimension should be passed through."""
    source = [torch.tensor(5.0), "x"]
    result = list(lazy_module.UnbatchedDataset(source))

    assert result[0].item() == 5.0
    assert result[1] == "x"

def test_taken_dataset_consumes_exact_count_and_caches(lazy_module):
    """TakenDataset should not over-consume and should replay cached values."""
    consumed = {"value": 0}

    def dataset_iter():
        for idx in range(6):
            consumed["value"] += 1
            yield idx

    taken = lazy_module.TakenDataset(dataset_iter(), count=3)
    assert consumed["value"] == 0

    first = list(taken)
    assert first == [0, 1, 2]
    assert consumed["value"] == 3

    second = list(taken)
    assert second == [0, 1, 2]
    assert consumed["value"] == 3

def test_taken_dataset_zero_count(lazy_module):
    """TakenDataset with zero count should consume nothing."""
    consumed = {"value": 0}

    def dataset_iter():
        for idx in range(5):
            consumed["value"] += 1
            yield idx

    taken = lazy_module.TakenDataset(dataset_iter(), count=0)
    assert list(taken) == []
    assert consumed["value"] == 0

def test_buffered_shuffle_dataset_invariants(lazy_module):
    """BufferedShuffleDataset should preserve length and items."""
    source = list(range(12))
    random.seed(123)

    shuffled = lazy_module.BufferedShuffleDataset(source, buffer_size=4)
    values = list(shuffled)

    assert len(shuffled) == len(source)
    assert len(values) == len(source)
    assert sorted(values) == source

def test_buffered_shuffle_dataset_materializes_one_pass_iterator(lazy_module):
    """BufferedShuffleDataset should warn and support re-iteration."""

    def dataset_iter():
        for idx in range(5):
            yield idx

    with pytest.warns(RuntimeWarning, match="one-pass iterator"):
        shuffled = lazy_module.BufferedShuffleDataset(dataset_iter(), buffer_size=2)

    first = list(shuffled)
    second = list(shuffled)
    assert sorted(first) == [0, 1, 2, 3, 4]
    assert sorted(second) == [0, 1, 2, 3, 4]

def test_buffered_shuffle_dataset_invalid_buffer(lazy_module):
    """BufferedShuffleDataset should reject invalid buffers."""
    with pytest.raises(ValueError):
        lazy_module.BufferedShuffleDataset([1, 2, 3], buffer_size=0)


def test_to_lazy_dataset_wraps_iterables(lazy_module):
    """to_lazy_dataset should expose fluent API for plain iterables."""
    dataset = lazy_module.to_lazy_dataset([1, 2, 3])

    assert isinstance(dataset, lazy_module.LazyDataset)
    assert list(dataset) == [1, 2, 3]

def test_fluent_pipeline_map_batch_unbatch_take_cache(lazy_module):
    """Fluent pipeline should preserve lazy + cache semantics."""
    samples = [
        (torch.tensor([1.0]), torch.tensor([10.0])),
        (torch.tensor([2.0]), torch.tensor([20.0])),
        (torch.tensor([3.0]), torch.tensor([30.0])),
        (torch.tensor([4.0]), torch.tensor([40.0])),
        (torch.tensor([5.0]), torch.tensor([50.0])),
    ]
    call_count = {"value": 0}

    def map_fn(sample):
        call_count["value"] += 1
        x, y = sample
        return x + 1.0, y * 2.0

    cached = (
        lazy_module.to_lazy_dataset(samples)
        .map(map_fn)
        .batch(2)
        .unbatch()
        .take(3)
        .cache()
    )
    calls_after_cache = call_count["value"]

    assert calls_after_cache >= 3
    assert len(cached) == 3

    first_pass = list(cached)
    second_pass = list(cached)

    assert call_count["value"] == calls_after_cache

    for item in [first_pass, second_pass]:
        assert len(item) == 3
        assert torch.equal(item[0][0], torch.tensor([2.0]))
        assert torch.equal(item[0][1], torch.tensor([20.0]))
        assert torch.equal(item[1][0], torch.tensor([3.0]))
        assert torch.equal(item[1][1], torch.tensor([40.0]))
        assert torch.equal(item[2][0], torch.tensor([4.0]))
        assert torch.equal(item[2][1], torch.tensor([60.0]))

def test_fluent_map_materializes_one_pass_iterator(lazy_module):
    """Fluent map should warn and materialize one-pass inputs."""

    def dataset_iter():
        for idx in range(2):
            yield idx

    with pytest.warns(RuntimeWarning, match="one-pass iterator"):
        mapped = lazy_module.to_lazy_dataset(dataset_iter()).map(lambda value: value + 1)

    assert list(mapped) == [1, 2]
    assert list(mapped) == [1, 2]

def test_adapter_len_does_not_consume_one_pass_iterator(lazy_module):
    """Asking len on one-pass adapter should not consume the source."""
    consumed = {"value": 0}

    def dataset_iter():
        for idx in range(4):
            consumed["value"] += 1
            yield idx

    dataset = lazy_module.to_lazy_dataset(dataset_iter())

    with pytest.raises(TypeError):
        len(dataset)

    assert consumed["value"] == 0
    assert list(dataset) == [0, 1, 2, 3]

def test_fluent_take_short_circuits_one_pass_iterator(lazy_module):
    """Fluent take should preserve short-circuit behavior on iterators."""
    consumed = {"value": 0}

    def dataset_iter():
        for idx in range(10):
            consumed["value"] += 1
            yield idx

    taken = lazy_module.to_lazy_dataset(dataset_iter()).take(3)
    assert consumed["value"] == 0

    assert list(taken) == [0, 1, 2]
    assert consumed["value"] == 3

    assert list(taken) == [0, 1, 2]
    assert consumed["value"] == 3

def test_batch_is_noop_when_already_batched(lazy_module):
    """Calling batch on an already batched dataset should be a no-op."""
    batched = lazy_module.BatchedDataset([1, 2, 3], batch_size=2)

    assert batched.batch(5) is batched
