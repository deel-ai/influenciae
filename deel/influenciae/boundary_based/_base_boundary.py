# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Shared utilities for boundary-based influence calculators.
"""
from ..common import BaseBackend
from ..types import Tuple, Any


class _BaseBoundaryCalculatorMixin:
    """Internal helper mixin for common DeepFool boundary computations."""

    backend: BaseBackend

    def _delta_to_index(self, indexes_1: Any, indexes_2: Any, x: Any) -> Any:
        """
        Compute the difference between the logit of a given class and the other logits.

        Parameters
        ----------
        indexes_1
            The indices of other classes.
        indexes_2
            The indices of the predicted class.
        x
            The logits.

        Returns
        -------
        delta_x
            The difference between the logits.
        """
        x1 = self.backend.gather_along_axis(x, indexes_1, axis=1, batch_dims=1)
        x2 = self.backend.gather_along_axis(
            x,
            self.backend.expand_dims(indexes_2, axis=1),
            axis=1,
            batch_dims=1
        )

        x1_shape = self.backend.tensor_shape(x1)
        return x1 - self.backend.repeat(x2, x1_shape[1], axis=1)

    def _build_other_class_indices(self, y_pred_class: Any, y_shape: Tuple[Any, Any]) -> Any:
        """Build indices of classes different from the model-predicted class."""
        batch_size = y_shape[0]
        num_classes = y_shape[1]

        indexes_all = self.backend.tile(
            self.backend.expand_dims(self.backend.arange(0, num_classes), axis=0),
            (batch_size, 1)
        )
        indexes_class = self.backend.cast(
            self.backend.tile(
                self.backend.expand_dims(y_pred_class, axis=1),
                (1, num_classes)
            ),
            dtype=self.backend.int32_dtype()
        )

        mask = indexes_all != indexes_class
        return self.backend.reshape(
            self.backend.boolean_mask(indexes_all, mask),
            (-1, num_classes - 1)
        )

    def _compute_delta_y(self, indexes_other: Any, y_pred_class: Any, y: Any) -> Any:
        """Compute class-wise mean absolute logit deltas for DeepFool updates."""
        delta_y = self._delta_to_index(indexes_other, y_pred_class, y)
        return self.backend.abs(self.backend.reduce_mean(delta_y, axis=0))

    def _compute_step_condition(self, y: Any, y_pred_class: Any, eps: float) -> Any:
        """Compute whether optimization should continue for current logits."""
        y_computed = self.backend.argmax(y, axis=1)
        computation = self.backend.reduce_any(y_computed == y_pred_class)

        top_k_values, _ = self.backend.top_k(self.backend.squeeze(y, axis=0), k=2)
        enough_close = self.backend.abs(top_k_values[0] - top_k_values[1]) > eps

        return self.backend.logical_and(computation, enough_close)
