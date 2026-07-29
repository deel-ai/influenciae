# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Streaming projected-curvature operations for backend-neutral TrackStar artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Tuple, Union

import numpy as np


class MixRule(Protocol):
    """Minimal protocol for selecting an eval-side curvature weight."""

    def alpha(self, train_gram: np.ndarray, eval_gram: np.ndarray) -> float:
        """Return a value in ``[0, 1]``."""


@dataclass(frozen=True)
class FixedMix:
    """Use a fixed eval-side weight for every projected block."""

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.value) <= 1.0:
            raise ValueError("FixedMix value must be in [0, 1].")

    def alpha(self, train_gram: np.ndarray, eval_gram: np.ndarray) -> float:
        """Return the configured fixed weight."""
        del train_gram, eval_gram
        return float(self.value)


@dataclass(frozen=True)
class AutoMix:
    """Balance train/eval contributions by their largest eigenvalues.

    The rule chooses ``alpha`` so the weighted spectral radii are equal:
    ``(1-alpha) * rho(train) == alpha * rho(eval)``. If both are zero,
    ``fallback`` is used.
    """

    fallback: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.fallback) <= 1.0:
            raise ValueError("AutoMix fallback must be in [0, 1].")

    def alpha(self, train_gram: np.ndarray, eval_gram: np.ndarray) -> float:
        """Return the spectral-radius balancing weight."""
        train_radius = max(float(np.linalg.eigvalsh(_symmetrize(train_gram))[-1]), 0.0)
        eval_radius = max(float(np.linalg.eigvalsh(_symmetrize(eval_gram))[-1]), 0.0)
        total = train_radius + eval_radius
        return float(self.fallback) if total == 0.0 else train_radius / total


@dataclass(frozen=True)
class GramSnapshot:
    """Serializable uncentered Gram sums and their sample count."""

    gram_sums: Tuple[np.ndarray, ...]
    count: int
    block_slices: Tuple[Tuple[str, int, int], ...]

    @property
    def grams(self) -> Tuple[np.ndarray, ...]:
        """Return per-block empirical uncentered Gram matrices."""
        if self.count <= 0:
            raise ValueError("Cannot average an empty Gram snapshot.")
        return tuple(gram_sum / self.count for gram_sum in self.gram_sums)


class ProjectedGramAccumulator:
    """Stream uncentered Gram sums without retaining projected rows."""

    def __init__(self, block_slices: Mapping[str, slice]):
        if not block_slices:
            raise ValueError("At least one projected block is required.")
        entries = []
        expected_start = 0
        for name, block_slice in block_slices.items():
            if block_slice.step not in (None, 1) or block_slice.start != expected_start:
                raise ValueError("Block slices must be contiguous, ordered, and have unit stride.")
            if block_slice.stop is None or block_slice.stop <= block_slice.start:
                raise ValueError("Every block slice must be non-empty.")
            entries.append((str(name), int(block_slice.start), int(block_slice.stop)))
            expected_start = int(block_slice.stop)
        self._block_slices = tuple(entries)
        self._gram_sums = [np.zeros((stop - start, stop - start), dtype=np.float64)
                           for _, start, stop in entries]
        self._count = 0

    @property
    def count(self) -> int:
        """Number of projected rows accumulated so far."""
        return self._count

    def update(self, projected_rows: np.ndarray) -> None:
        """Add one batch to every block Gram sum."""
        rows = np.asarray(projected_rows)
        width = self._block_slices[-1][2]
        if rows.ndim != 2 or rows.shape[1] != width:
            raise ValueError(f"Expected projected rows of shape (batch, {width}); got {rows.shape}.")
        if not np.all(np.isfinite(rows)):
            raise ValueError("Projected rows must be finite.")
        for gram_sum, (_, start, stop) in zip(self._gram_sums, self._block_slices):
            block = rows[:, start:stop].astype(np.float64, copy=False)
            gram_sum += block.T @ block
        self._count += int(rows.shape[0])

    def snapshot(self) -> GramSnapshot:
        """Copy the current sums into a portable immutable snapshot."""
        return GramSnapshot(tuple(gram.copy() for gram in self._gram_sums),
                            self._count, self._block_slices)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Curvature blocks must be square matrices.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Curvature blocks must be finite.")
    return 0.5 * (matrix + matrix.T)


def inverse_sqrt_psd(
    matrix: np.ndarray, *, rcond: float = 1e-12, negative_tolerance: float = 1e-10
) -> np.ndarray:
    """Return the Moore-Penrose inverse square root of a PSD matrix.

    Tiny negative eigenvalues from roundoff are clipped. Materially indefinite
    inputs are rejected; no ridge is added by default or implicitly.
    """
    if rcond < 0 or negative_tolerance < 0:
        raise ValueError("rcond and negative_tolerance must be non-negative.")
    symmetric = _symmetrize(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 2.220446049250313e-16)
    if float(eigenvalues[0]) < -negative_tolerance * scale:
        raise ValueError("Curvature block is not positive semidefinite.")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[-1])
    inverse_roots = np.zeros_like(eigenvalues)
    if largest > 0.0:
        retained = eigenvalues > rcond * largest
        inverse_roots[retained] = 1.0 / np.sqrt(eigenvalues[retained])
    return (eigenvectors * inverse_roots) @ eigenvectors.T


@dataclass(frozen=True)
class ProjectedCurvature:
    """Blockwise inverse-square-root curvature transform."""

    transforms: Tuple[np.ndarray, ...]
    block_slices: Tuple[Tuple[str, int, int], ...]
    alphas: Tuple[float, ...]

    def apply(self, projected_rows: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        """Transform each block and optionally unit-normalize complete rows."""
        rows = np.asarray(projected_rows)
        width = self.block_slices[-1][2]
        if rows.ndim != 2 or rows.shape[1] != width:
            raise ValueError(f"Expected projected rows of shape (batch, {width}); got {rows.shape}.")
        parts = []
        for transform, (_, start, stop) in zip(self.transforms, self.block_slices):
            parts.append(rows[:, start:stop] @ transform)
        result = np.concatenate(parts, axis=1)
        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            np.divide(result, norms, out=result, where=norms > 0.0)
        return result


def build_projected_curvature(
    train: GramSnapshot,
    evaluation: GramSnapshot,
    mix: Union[MixRule, float] = FixedMix(0.5),
    *,
    rcond: float = 1e-12,
    negative_tolerance: float = 1e-10,
) -> ProjectedCurvature:
    """Mix train/eval empirical Grams directly and build block transforms."""
    if train.block_slices != evaluation.block_slices:
        raise ValueError("Train and evaluation snapshots must have identical block slices.")
    if train.count <= 0 or evaluation.count <= 0:
        raise ValueError("Train and evaluation snapshots must both contain rows.")
    rule: MixRule = FixedMix(float(mix)) if isinstance(mix, (int, float)) else mix
    transforms = []
    alphas = []
    for train_gram, eval_gram in zip(train.grams, evaluation.grams):
        alpha = float(rule.alpha(train_gram, eval_gram))
        if not 0.0 <= alpha <= 1.0 or not np.isfinite(alpha):
            raise ValueError("Mix rules must return a finite alpha in [0, 1].")
        # Mix empirical Gram matrices themselves, not factors or square roots.
        mixed = (1.0 - alpha) * train_gram + alpha * eval_gram
        transforms.append(inverse_sqrt_psd(
            mixed, rcond=rcond, negative_tolerance=negative_tolerance
        ))
        alphas.append(alpha)
    return ProjectedCurvature(tuple(transforms), train.block_slices, tuple(alphas))
