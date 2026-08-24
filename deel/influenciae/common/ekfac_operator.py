# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Reusable application of an eigenvalue-corrected K-FAC inverse."""
from typing import Dict, List, Optional

from ..types import Tensor
from .backend import BaseBackend
from .kfac_factors import EKFACFactors, HEURISTIC_DAMPING_SCALE, LayerParameterMap
from .model_wrappers import BaseInfluenceModel


class EKFACInverseOperator:
    """Apply precomputed EK-FAC factors to row-batched flat vectors."""

    def __init__(
        self,
        model: BaseInfluenceModel,
        backend: BaseBackend,
        layer_map: LayerParameterMap,
        factors: EKFACFactors,
        damping: Optional[float] = 1e-4,
    ):
        self.model = model
        self.backend = backend
        self.layer_map = layer_map
        self.factors = factors
        self.damping = damping
        self.layer_damping: Dict[int, Tensor] = {}

        for info in layer_map.layers_info:
            idx = info.layer_idx
            if idx not in factors.Lambda_corrected:
                continue
            eigenvalues = factors.Lambda_corrected[idx]
            dtype = backend.get_dtype(eigenvalues)
            if damping is None:
                scale = backend.cast(backend.constant(HEURISTIC_DAMPING_SCALE), dtype)
                damping_tensor = scale * backend.reduce_mean(eigenvalues)
            else:
                damping_tensor = backend.cast(backend.constant(damping), dtype)
            self.layer_damping[idx] = damping_tensor

    def reshape_per_module(self, vectors: Tensor) -> Dict[int, Tensor]:
        """Reshape ``(R, P)`` vectors into framework-native module matrices."""
        rows = self._validate_vectors(vectors)
        per_module: Dict[int, Tensor] = {}
        for info in self.layer_map.layers_info:
            idx = info.layer_idx
            if idx not in self.layer_damping or idx not in self.factors.Q_A or idx not in self.factors.Q_G:
                continue
            q_a = self.factors.Q_A[idx]
            q_g = self.factors.Q_G[idx]
            n_out = int(self.backend.tensor_shape(q_g)[0])
            n_in_eff = int(self.backend.tensor_shape(q_a)[0])
            layer_slice = vectors[:, info.flat_start:info.flat_end]
            if self.backend.framework.value == "tensorflow":
                shape = (rows, n_in_eff, n_out)
            else:
                shape = (rows, n_out, n_in_eff)
            per_module[idx] = self.backend.reshape(layer_slice, shape)
        return per_module

    def apply(self, vectors: Tensor) -> Tensor:
        """Apply the inverse to row-batched flat vectors of shape ``(R, P)``."""
        rows = self._validate_vectors(vectors)
        backend = self.backend
        result = backend.zeros_like(vectors)

        for info in self.layer_map.layers_info:
            idx = info.layer_idx
            if (
                idx not in self.layer_damping
                or idx not in self.factors.Q_A
                or idx not in self.factors.Q_G
            ):
                continue

            q_a = self.factors.Q_A[idx]
            q_g = self.factors.Q_G[idx]
            lam_damped = self.factors.Lambda_corrected[idx] + self.layer_damping[idx]
            n_out = int(backend.tensor_shape(q_g)[0])
            n_in_eff = int(backend.tensor_shape(q_a)[0])
            q_g_t = backend.transpose(q_g)
            q_a_t = backend.transpose(q_a)
            layer_vectors = vectors[:, info.flat_start:info.flat_end]

            if backend.framework.value == "tensorflow":
                v_mat = backend.reshape(layer_vectors, (rows, n_in_eff, n_out))
                v_rotated = backend.matmul(backend.matmul(q_a_t, v_mat), q_g)
                lam_mat = backend.reshape(lam_damped, (n_out, n_in_eff))
                lam_native = backend.reshape(backend.transpose(lam_mat), (-1,))
                v_flat = backend.reshape(v_rotated, (rows, -1))
                v_scaled = v_flat / backend.expand_dims(lam_native, axis=0)
                v_scaled_mat = backend.reshape(v_scaled, (rows, n_in_eff, n_out))
                inverse_layer = backend.matmul(backend.matmul(q_a, v_scaled_mat), q_g_t)
            else:
                v_mat = backend.reshape(layer_vectors, (rows, n_out, n_in_eff))
                v_rotated = backend.matmul(backend.matmul(q_g_t, v_mat), q_a)
                v_flat = backend.reshape(v_rotated, (rows, -1))
                v_scaled = v_flat / backend.expand_dims(lam_damped, axis=0)
                v_scaled_mat = backend.reshape(v_scaled, (rows, n_out, n_in_eff))
                inverse_layer = backend.matmul(backend.matmul(q_g, v_scaled_mat), q_a_t)

            flat_layer = backend.reshape(inverse_layer, (rows, -1))
            result = self._write_layer_slice(
                result, flat_layer, info.flat_start, info.flat_end
            )

        return result

    def _validate_vectors(self, vectors: Tensor) -> int:
        shape = self.backend.tensor_shape(vectors)
        if len(shape) != 2 or int(shape[1]) != int(self.model.nb_params):
            raise ValueError(
                "EKFAC inverse vectors must have shape "
                f"(R, {int(self.model.nb_params)}); got {shape}."
            )
        return self.backend.get_batch_size(vectors)

    def _write_layer_slice(
        self, result: Tensor, values: Tensor, start: int, end: int
    ) -> Tensor:
        parts: List[Tensor] = []
        if start > 0:
            parts.append(result[:, :start])
        parts.append(values)
        if end < int(self.model.nb_params):
            parts.append(result[:, end:])
        return self.backend.concat(parts, axis=1)
