# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests for the shared partitioning utilities.
These tests are backend-agnostic and do not require TensorFlow or PyTorch.
"""
import os
import numpy as np
import pytest

from deel.influenciae.utils.partitioning import (
    chunk_items,
    remove_path,
    atomic_write_directory,
    make_partition_dir,
    cleanup_partition_dir,
)


pytestmark = pytest.mark.backend_agnostic


# ---------------------------------------------------------------------------
# chunk_items
# ---------------------------------------------------------------------------

def test_chunk_items_even():
    result = chunk_items(list(range(6)), 2)
    assert result == [[0, 1], [2, 3], [4, 5]]


def test_chunk_items_uneven():
    result = chunk_items(list(range(5)), 2)
    assert result == [[0, 1], [2, 3], [4]]


def test_chunk_items_larger_than_list():
    result = chunk_items([1, 2, 3], 10)
    assert result == [[1, 2, 3]]


def test_chunk_items_empty():
    result = chunk_items([], 3)
    assert result == []


def test_chunk_items_size_one():
    result = chunk_items([10, 20, 30], 1)
    assert result == [[10], [20], [30]]


# ---------------------------------------------------------------------------
# remove_path
# ---------------------------------------------------------------------------

def test_remove_path_file(tmp_path):
    f = tmp_path / "dummy.txt"
    f.write_text("hello")
    assert f.exists()
    remove_path(str(f))
    assert not f.exists()


def test_remove_path_dir(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    (d / "file.txt").write_text("x")
    assert d.exists()
    remove_path(str(d))
    assert not d.exists()


def test_remove_path_nonexistent(tmp_path):
    # Should not raise
    remove_path(str(tmp_path / "no_such_file.txt"))


# ---------------------------------------------------------------------------
# atomic_write_directory
# ---------------------------------------------------------------------------

def test_atomic_write_directory_creates_target(tmp_path):
    target = str(tmp_path / "output")

    def writer(d):
        with open(os.path.join(d, "data.txt"), "w") as fh:
            fh.write("written")

    atomic_write_directory(target, writer)
    assert os.path.isdir(target)
    assert os.path.isfile(os.path.join(target, "data.txt"))


def test_atomic_write_directory_replaces_existing(tmp_path):
    target = str(tmp_path / "output")
    os.makedirs(target)
    old_file = os.path.join(target, "old.txt")
    with open(old_file, "w") as fh:
        fh.write("old")

    def writer(d):
        with open(os.path.join(d, "new.txt"), "w") as fh:
            fh.write("new")

    atomic_write_directory(target, writer)
    assert not os.path.exists(old_file)
    assert os.path.isfile(os.path.join(target, "new.txt"))


def test_atomic_write_directory_cleans_up_on_error(tmp_path):
    target = str(tmp_path / "output")

    def failing_writer(d):
        raise RuntimeError("intentional")

    with pytest.raises(RuntimeError):
        atomic_write_directory(target, failing_writer)

    # Target directory should not have been created
    assert not os.path.exists(target)
    # No leftover tmp dirs
    leftovers = [p for p in os.listdir(str(tmp_path)) if p.startswith(".tmp_partition_")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# make_partition_dir / cleanup_partition_dir
# ---------------------------------------------------------------------------

def test_make_partition_dir_default():
    d = make_partition_dir("myprefix")
    try:
        assert os.path.isdir(d)
        assert "myprefix" in os.path.basename(d)
    finally:
        remove_path(d)


def test_make_partition_dir_with_base(tmp_path):
    base = str(tmp_path / "base")
    d = make_partition_dir("myprefix", base_dir=base)
    assert os.path.isdir(d)
    assert d.startswith(base)


def test_cleanup_partition_dir_removes(tmp_path):
    d = make_partition_dir("test", base_dir=str(tmp_path))
    assert os.path.isdir(d)
    cleanup_partition_dir(d, keep_artifacts=False)
    assert not os.path.exists(d)


def test_cleanup_partition_dir_keeps_when_flag_set(tmp_path):
    d = make_partition_dir("test", base_dir=str(tmp_path))
    cleanup_partition_dir(d, keep_artifacts=True)
    assert os.path.isdir(d)
    remove_path(d)


def test_cleanup_partition_dir_none_is_noop():
    # Should not raise
    cleanup_partition_dir(None, keep_artifacts=False)
