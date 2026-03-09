# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
K-FAC and EK-FAC Kronecker factor computation for Inverse Hessian Vector Products.

This module implements the Kronecker-Factored Approximate Curvature (K-FAC) and its
eigenvalue-corrected variant (EK-FAC) as described in:

- Martens & Grosse (2015), "Optimizing Neural Networks with Kronecker-Factored
  Approximate Curvature"
- Grosse et al. (2023), "Studying Large Language Model Generalization with Influence
  Functions" (arXiv:2308.03296)

K-FAC approximates the Fisher information matrix block-diagonally per layer, factoring
each block as a Kronecker product of input activation covariance (A) and output gradient
covariance (G).  EK-FAC refines this by eigendecomposing the factors and estimating
corrected diagonal eigenvalues.
"""
import warnings
from dataclasses import dataclass, field

from .backend import BaseBackend
from .model_wrappers import BaseInfluenceModel

from ..types import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerInfo:
    """Metadata for a single K-FAC-supported layer.

    Attributes
    ----------
    layer
        The framework-specific layer object.
    layer_idx
        Index of the layer in the model's layer list.
    weight_shape
        Shape of the weight tensor as reported by the framework.  In PyTorch
        this is ``(n_out, n_in)`` for Linear and ``(c_out, c_in, kH, kW)``
        for Conv2d.  In TensorFlow it is ``(n_in, n_out)`` for Dense and
        ``(kH, kW, c_in, c_out)`` for Conv2D.  Do **not** assume a fixed
        dimension ordering — use the Kronecker factor matrix shapes to derive
        ``n_out`` and ``n_in_eff`` instead.
    has_bias
        Whether the layer has a bias parameter.
    flat_start
        Start index of this layer's parameters in the flat gradient vector.
    flat_end
        End index (exclusive) of this layer's parameters in the flat gradient vector.
    """
    layer: Any
    layer_idx: int
    weight_shape: Tuple[int, ...]
    has_bias: bool
    flat_start: int
    flat_end: int


class LayerParameterMap:
    """Maps between flat gradient vectors and per-layer matrix representations.

    This mirrors the ``ForwardOverBackwardHVP._weight_slices`` pattern but
    adds layer-level bookkeeping needed by K-FAC: the map only includes
    *supported* (Linear / Conv2d) layers and records their position within the
    complete flat gradient vector.

    Parameters
    ----------
    model
        The influence model wrapper.
    backend
        The backend abstraction.
    target_layers
        If provided, restrict K-FAC to these layer indices only.
    """

    def __init__(
        self,
        model: BaseInfluenceModel,
        backend: BaseBackend,
        target_layers: Optional[List[int]] = None
    ):
        self.backend = backend
        self.layers_info: List[LayerInfo] = []

        all_layers = backend.get_layers(model.model)
        weights = model.weights

        def _iter_weight_keys(weight: Any) -> List[Tuple[str, Any]]:
            """Return matching keys for a weight object across backend wrappers.

            PyTorch parameters are stable Python objects, so ``id(param)`` is
            sufficient. TensorFlow/Keras can expose wrapped variable objects
            (e.g. KerasVariable vs. ResourceVariable), so we include both
            object-id and name-based keys from common wrapper attributes.
            """
            keys: List[Tuple[str, Any]] = []
            to_visit = [weight]

            normalize_weights = getattr(backend, "normalize_weights_to_watch", None)
            if callable(normalize_weights):
                try:
                    normalized = normalize_weights([weight])
                    if isinstance(normalized, (list, tuple)):
                        for normalized_weight in normalized:
                            to_visit.append(normalized_weight)
                    elif normalized is not None:
                        to_visit.append(normalized)
                except Exception:  # pylint: disable=broad-except
                    pass

            seen_ids = set()
            while to_visit:
                current = to_visit.pop()
                if current is None:
                    continue

                current_id = id(current)
                if current_id in seen_ids:
                    continue
                seen_ids.add(current_id)

                keys.append(("id", current_id))

                current_name = getattr(current, "name", None)
                if current_name is not None:
                    keys.append(("name", str(current_name)))

                for attr_name in ("_variable", "variable", "_value", "value", "handle"):
                    attr_value = getattr(current, attr_name, None)
                    if callable(attr_value):
                        try:
                            attr_value = attr_value()
                        except TypeError:
                            continue
                    if attr_value is not None:
                        to_visit.append(attr_value)

            return keys

        # Build a mapping from backend-stable parameter keys to their position
        # in the flat gradient vector (same logic as ForwardOverBackwardHVP).
        weight_key_to_flat: Dict[Tuple[str, Any], Tuple[int, int, Tuple[int, ...]]] = {}
        offset = 0
        for w in weights:
            shape = backend.tensor_shape(w)
            size = 1
            for d in shape:
                size *= int(d)
            flat_entry = (offset, offset + size, shape)
            for key in _iter_weight_keys(w):
                if key not in weight_key_to_flat:
                    weight_key_to_flat[key] = flat_entry
            offset += size

        for layer_idx, layer in enumerate(all_layers):
            if target_layers is not None and layer_idx not in target_layers:
                continue

            if not (backend.is_linear_layer(layer) or backend.is_conv2d_layer(layer)):
                if target_layers is not None:
                    warnings.warn(
                        f"Layer {layer_idx} ({type(layer).__name__}) is not a Linear/Dense "
                        f"or Conv2d layer and will be skipped by K-FAC.",
                        stacklevel=2,
                    )
                continue

            weight, bias = backend.get_layer_weight_and_bias(layer)
            weight_shape = backend.tensor_shape(weight)

            # Find the weight in the flat vector
            flat_entry = None
            for key in _iter_weight_keys(weight):
                if key in weight_key_to_flat:
                    flat_entry = weight_key_to_flat[key]
                    break

            if flat_entry is None:
                # Weight not tracked (frozen, outside watched range, or wrapper mismatch)
                continue

            flat_start = flat_entry[0]
            flat_end = flat_entry[1]

            has_bias = bias is not None
            if has_bias:
                for key in _iter_weight_keys(bias):
                    if key in weight_key_to_flat:
                        # Bias immediately follows weight in the flat vector
                        flat_end = weight_key_to_flat[key][1]
                        break

            self.layers_info.append(LayerInfo(
                layer=layer,
                layer_idx=layer_idx,
                weight_shape=weight_shape,
                has_bias=has_bias,
                flat_start=flat_start,
                flat_end=flat_end,
            ))

        if not self.layers_info:
            warnings.warn(
                "No supported layers (Linear/Dense, Conv2d) found for K-FAC. "
                "The IHVP will fall back to a zero approximation.",
                stacklevel=2,
            )

    @property
    def n_supported_layers(self) -> int:
        return len(self.layers_info)


# ---------------------------------------------------------------------------
# K-FAC factor computation
# ---------------------------------------------------------------------------

class KroneckerFactors:
    """Compute and store Kronecker factors A_l and G_l for each supported layer.

    For a fully-connected layer with weight W of shape ``(n_out, n_in)``:

    * ``A_l = E[a_l a_l^T]``  — input activation covariance ``(n_in, n_in)``
      (or ``(n_in+1, n_in+1)`` when bias is present, with the homogeneous "1"
      appended to activations).
    * ``G_l = E[g_l g_l^T]``  — output gradient covariance ``(n_out, n_out)``.

    For Conv2d the activations are unfolded (im2col) before computing A.

    Parameters
    ----------
    model
        The influence model wrapper.
    train_dataset
        Batched training dataset for estimating the factors.
    backend
        The backend abstraction.
    layer_map
        Pre-computed ``LayerParameterMap``.
    """

    def __init__(
        self,
        model: BaseInfluenceModel,
        train_dataset: Any,
        backend: BaseBackend,
        layer_map: LayerParameterMap,
    ):
        self.backend = backend
        self.model = model
        self.layer_map = layer_map
        self.A: Dict[int, Any] = {}  # layer_idx -> A factor
        self.G: Dict[int, Any] = {}  # layer_idx -> G factor

        self._compute_factors(train_dataset)

    # ------------------------------------------------------------------
    # Factor computation
    # ------------------------------------------------------------------

    def _compute_factors(self, train_dataset: Any) -> None:
        """Estimate A and G from *train_dataset* using hooks."""
        backend = self.backend
        model = self.model

        def _to_int(value: Any) -> int:
            """Convert backend scalar values to Python int."""
            try:
                return int(value)
            except TypeError:
                return int(value.numpy())

        # Storage for running sums
        a_sums: Dict[int, Any] = {}
        g_sums: Dict[int, Any] = {}
        n_rows_per_layer: Dict[int, int] = {}

        # --- Register hooks -----------------------------------------------
        handles = []
        activations: Dict[int, Any] = {}
        grad_outputs: Dict[int, Any] = {}

        for info in self.layer_map.layers_info:
            idx = info.layer_idx

            def _fwd_hook(layer, inp, out, _idx=idx):
                # inp is a tuple; take the first element (the actual input tensor)
                a = inp if not isinstance(inp, tuple) else inp[0]
                activations[_idx] = a

            def _bwd_hook(layer, grad_inp, grad_out, _idx=idx):
                g = grad_out if not isinstance(grad_out, tuple) else grad_out[0]
                grad_outputs[_idx] = g

            handles.append(backend.register_forward_hook(info.layer, _fwd_hook))
            handles.append(backend.register_backward_hook(info.layer, _bwd_hook))

        # --- Iterate over dataset -----------------------------------------
        try:
            for batch in train_dataset:
                if isinstance(batch, (list, tuple)):
                    batch_tuple = tuple(batch)
                else:
                    batch_tuple = (batch,)

                # Use model's preprocessing
                model_inp, y_true, sample_weight = model.process_batch_for_loss_fn(batch_tuple)

                # Forward pass
                activations.clear()
                grad_outputs.clear()

                # We need gradients w.r.t. outputs to trigger backward hooks.
                # Compute the per-sample loss and backprop.
                self._forward_backward(model, model_inp, y_true, sample_weight,
                                       activations, grad_outputs)

                # Accumulate factors
                for info in self.layer_map.layers_info:
                    idx = info.layer_idx
                    if idx not in activations or idx not in grad_outputs:
                        continue

                    a = activations[idx]
                    g = grad_outputs[idx]

                    a_mat, g_mat = self._prepare_activation_gradient(info, a, g)

                    a_rows = _to_int(backend.get_batch_size(a_mat))
                    g_rows = _to_int(backend.get_batch_size(g_mat))
                    if a_rows != g_rows:
                        raise ValueError(
                            f"Activation/gradient row mismatch for layer {idx}: "
                            f"{a_rows} vs {g_rows}."
                        )

                    n_rows_per_layer[idx] = n_rows_per_layer.get(idx, 0) + a_rows

                    if idx not in a_sums:
                        a_sums[idx] = backend.matmul(backend.transpose(a_mat), a_mat)
                        g_sums[idx] = backend.matmul(backend.transpose(g_mat), g_mat)
                    else:
                        a_sums[idx] = a_sums[idx] + backend.matmul(backend.transpose(a_mat), a_mat)
                        g_sums[idx] = g_sums[idx] + backend.matmul(backend.transpose(g_mat), g_mat)
        finally:
            # --- Cleanup hooks --------------------------------------------
            for h in handles:
                backend.remove_hook(h)

        # --- Normalize and store ------------------------------------------
        for info in self.layer_map.layers_info:
            idx = info.layer_idx
            if idx not in a_sums:
                continue

            n_rows = n_rows_per_layer.get(idx, 0)
            if n_rows <= 0:
                continue

            a_norm = backend.cast(
                backend.constant(float(n_rows)),
                backend.get_dtype(a_sums[idx])
            )
            g_norm = backend.cast(
                backend.constant(float(n_rows)),
                backend.get_dtype(g_sums[idx])
            )
            self.A[idx] = a_sums[idx] / a_norm
            self.G[idx] = g_sums[idx] / g_norm

    def _forward_backward(
        self,
        model: BaseInfluenceModel,
        model_inp: Any,
        y_true: Any,
        sample_weight: Optional[Any],
        activations: Dict[int, Any],
        grad_outputs: Dict[int, Any],
    ) -> None:
        """Run forward + backward to populate hook-captured activations and gradients.

        In PyTorch we can just do a normal forward/backward pass and the hooks
        fire automatically.  In TensorFlow we need a GradientTape approach;
        the forward hook wrapper on ``layer.call`` fires during the tape-traced
        forward pass, and we manually invoke backward hooks with the per-layer
        output gradients obtained from the tape.
        """
        backend = self.backend

        if backend.framework.value == "pytorch":
            self._forward_backward_pytorch(model, model_inp, y_true, sample_weight)
        else:
            self._forward_backward_tensorflow(
                model, model_inp, y_true, sample_weight, grad_outputs
            )

    def _forward_backward_pytorch(
        self,
        model: BaseInfluenceModel,
        model_inp: Any,
        y_true: Any,
        sample_weight: Optional[Any],
    ) -> None:
        """PyTorch: standard forward + backward; hooks fire automatically."""
        import torch

        predictions = model.model(model_inp)
        loss = model.loss_function(predictions, y_true)
        if sample_weight is not None:
            loss = loss * sample_weight
        total_loss = loss.sum()
        total_loss.backward()

        # Zero grad after capture (we don't need optimizer updates)
        model.model.zero_grad()

    def _forward_backward_tensorflow(
        self,
        model: BaseInfluenceModel,
        model_inp: Any,
        y_true: Any,
        sample_weight: Optional[Any],
        grad_outputs: Dict[int, Any],
    ) -> None:
        """TensorFlow: use GradientTape + per-layer gradient for backward hooks.

        Strategy
        --------
        1. Create a persistent ``GradientTape``.
        2. Wrap each supported layer's ``call`` to both record intermediate
           outputs and call ``tape.watch`` on them *immediately*, so the tape
           traces operations downstream of each layer output.
        3. Run the full forward pass and compute the loss inside the tape
           context.
        4. After the tape context, use the tape to obtain per-layer output
           gradients and manually fire the backward hooks stored on each layer.
        5. In ``finally``, restore the original ``call`` for every layer to
           prevent stacking wrappers across batches.
        """
        import tensorflow as tf

        # Collect intermediate outputs so we can compute per-layer gradients
        layer_outputs: Dict[int, Any] = {}

        # Save original calls — will be restored in `finally`.
        original_calls: Dict[int, Callable] = {}

        try:
            with tf.GradientTape(persistent=True) as tape:
                # Install wrappers *inside* the tape context so `tape` is
                # available to the closure for `tape.watch`.
                for info in self.layer_map.layers_info:
                    idx = info.layer_idx
                    _orig = info.layer.call
                    original_calls[idx] = _orig

                    def _record_and_watch(*a, _idx=idx, _orig_fn=_orig, _tape=tape, **kw):
                        out = _orig_fn(*a, **kw)
                        layer_outputs[_idx] = out
                        _tape.watch(out)
                        return out

                    info.layer.call = _record_and_watch

                # Forward pass — the wrapped ``call`` fires, populates
                # *layer_outputs*, and calls ``tape.watch`` on each output.
                # The forward hooks installed by the caller also fire and
                # populate *activations*.
                predictions = model.model(model_inp, training=False)

                loss = model.loss_function(y_true, predictions)
                if sample_weight is not None:
                    loss = loss * sample_weight
                batch_size = tf.shape(predictions)[0]
                loss = tf.reshape(loss, (batch_size, -1))
                loss = tf.reduce_mean(loss, axis=1)
                total_loss = tf.reduce_sum(loss)

            # Compute gradient of total_loss w.r.t. each layer output and
            # manually invoke backward hooks.
            for info in self.layer_map.layers_info:
                idx = info.layer_idx
                if idx not in layer_outputs:
                    continue
                try:
                    g = tape.gradient(total_loss, layer_outputs[idx])
                    if g is not None:
                        grad_outputs[idx] = g
                        # Fire backward hooks
                        hooks = getattr(info.layer, '_kfac_backward_hooks', [])
                        for hook in hooks:
                            hook(info.layer, None, (g,))
                except Exception:  # pylint: disable=broad-except
                    pass
            del tape
        finally:
            # Restore original calls to prevent stacking wrappers
            for info in self.layer_map.layers_info:
                idx = info.layer_idx
                if idx in original_calls:
                    info.layer.call = original_calls[idx]

    def _prepare_activation_gradient(
        self, info: LayerInfo, activation: Any, grad_output: Any
    ) -> Tuple[Any, Any]:
        """Reshape activation and gradient for factor accumulation.

        For Linear/Dense layers:
        - activation: ``(batch, n_in)`` -> append 1 column if bias -> ``(batch, n_in+bias)``
        - grad_output: ``(batch, n_out)``

        For Conv2d layers:
        - activation: ``(batch, C_in, H, W)`` -> unfold -> ``(batch*H'*W', C_in*kH*kW+bias)``
        - grad_output: ``(batch, C_out, H', W')`` -> ``(batch*H'*W', C_out)``

        Returns ``(a_mat, g_mat)`` where rows correspond to samples.
        """
        backend = self.backend

        if backend.is_conv2d_layer(info.layer):
            return self._prepare_conv2d(info, activation, grad_output)

        # Linear / Dense
        a = activation
        g = grad_output

        # Ensure 2-D: (batch, features)
        if backend.tensor_ndim(a) > 2:
            shape = backend.tensor_shape(a)
            a = backend.reshape(a, (shape[0], -1))

        if backend.tensor_ndim(g) > 2:
            shape = backend.tensor_shape(g)
            g = backend.reshape(g, (shape[0], -1))
        elif backend.tensor_ndim(g) == 1:
            g = backend.expand_dims(g, axis=0)

        if info.has_bias:
            # Append column of ones for the bias term
            batch_size = backend.get_batch_size(a)
            ones = backend.cast(
                backend.ones((batch_size, 1)),
                backend.get_dtype(a)
            )
            a = backend.concat([a, ones], axis=1)

        return a, g

    def _prepare_conv2d(
        self, info: LayerInfo, activation: Any, grad_output: Any
    ) -> Tuple[Any, Any]:
        """Unfold Conv2d activations (im2col) and reshape gradients."""
        backend = self.backend

        # For PyTorch: activation is (B, C_in, H, W), grad_output is (B, C_out, H', W')
        # For TF/Keras: activation is (B, H, W, C_in), grad_output is (B, H', W', C_out)
        if backend.framework.value == "pytorch":
            return self._prepare_conv2d_pytorch(info, activation, grad_output)
        return self._prepare_conv2d_tensorflow(info, activation, grad_output)

    def _prepare_conv2d_pytorch(
        self, info: LayerInfo, activation: Any, grad_output: Any
    ) -> Tuple[Any, Any]:
        """PyTorch Conv2d unfolding."""
        import torch
        import torch.nn.functional as F

        layer = info.layer
        # Unfold using the layer's kernel_size, stride, padding, dilation
        a_unfold = F.unfold(
            activation,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
                if isinstance(layer.padding, int) else (0, 0),
            dilation=layer.dilation,
        )
        # a_unfold: (B, C_in*kH*kW, L) where L = H'*W'
        # Transpose to (B, L, C_in*kH*kW)
        a_unfold = a_unfold.permute(0, 2, 1)
        b, l, cin_k = a_unfold.shape

        if info.has_bias:
            ones = torch.ones(b, l, 1, device=a_unfold.device, dtype=a_unfold.dtype)
            a_unfold = torch.cat([a_unfold, ones], dim=2)

        # Reshape to (B*L, cin_k+bias)
        a_mat = a_unfold.reshape(b * l, -1)

        # grad_output: (B, C_out, H', W') -> (B, H'*W', C_out) -> (B*L, C_out)
        g = grad_output.permute(0, 2, 3, 1).reshape(b * l, -1)

        return a_mat, g

    def _prepare_conv2d_tensorflow(
        self, info: LayerInfo, activation: Any, grad_output: Any
    ) -> Tuple[Any, Any]:
        """TensorFlow Conv2d unfolding via extract_patches."""
        import tensorflow as tf

        layer = info.layer
        kernel_size = layer.kernel_size
        strides = layer.strides
        # Keras stores padding as string ('valid' or 'same')
        padding = layer.padding.upper() if isinstance(layer.padding, str) else 'VALID'

        # activation: (B, H, W, C_in)
        # tf.image.extract_patches -> (B, H', W', kH*kW*C_in)
        a_patches = tf.image.extract_patches(
            activation,
            sizes=[1, kernel_size[0], kernel_size[1], 1],
            strides=[1, strides[0], strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=padding,
        )
        shape = tf.shape(a_patches)
        b, hp, wp, patch_dim = shape[0], shape[1], shape[2], shape[3]
        l = hp * wp

        a_mat = tf.reshape(a_patches, [b * l, patch_dim])
        if info.has_bias:
            ones = tf.ones([b * l, 1], dtype=a_mat.dtype)
            a_mat = tf.concat([a_mat, ones], axis=1)

        # grad_output: (B, H', W', C_out) -> (B*L, C_out)
        g_shape = tf.shape(grad_output)
        g = tf.reshape(grad_output, [b * l, g_shape[-1]])

        return a_mat, g


# ---------------------------------------------------------------------------
# EK-FAC factor computation
# ---------------------------------------------------------------------------

class EKFACFactors(KroneckerFactors):
    """Extends :class:`KroneckerFactors` with eigendecomposition and corrected eigenvalues.

    After computing the base Kronecker factors ``A_l`` and ``G_l``, this class:

    1. Eigendecomposes each: ``A_l = Q_A Lambda_A Q_A^T``,  ``G_l = Q_G Lambda_G Q_G^T``.
    2. Estimates *corrected* diagonal eigenvalues from a second pass over training
       data (or a subset), following the EK-FAC procedure in Grosse et al. (2023).

    Parameters
    ----------
    model
        The influence model wrapper.
    train_dataset
        Batched training dataset.
    backend
        The backend abstraction.
    layer_map
        Pre-computed ``LayerParameterMap``.
    n_ekfac_samples
        Number of samples (batches are consumed until this many samples are seen)
        used for corrected eigenvalue estimation.  ``None`` means use all data.
    """

    def __init__(
        self,
        model: BaseInfluenceModel,
        train_dataset: Any,
        backend: BaseBackend,
        layer_map: LayerParameterMap,
        n_ekfac_samples: Optional[int] = None,
    ):
        # Compute base K-FAC factors (A, G)
        super().__init__(model, train_dataset, backend, layer_map)

        # Eigendecompose A and G
        self.Q_A: Dict[int, Any] = {}
        self.Lambda_A: Dict[int, Any] = {}
        self.Q_G: Dict[int, Any] = {}
        self.Lambda_G: Dict[int, Any] = {}

        for info in layer_map.layers_info:
            idx = info.layer_idx
            if idx in self.A:
                lam_a, q_a = backend.eigh(self.A[idx])
                lam_g, q_g = backend.eigh(self.G[idx])
                self.Q_A[idx] = q_a
                self.Lambda_A[idx] = lam_a
                self.Q_G[idx] = q_g
                self.Lambda_G[idx] = lam_g

        # Corrected eigenvalues
        self.Lambda_corrected: Dict[int, Any] = {}
        self._estimate_corrected_eigenvalues(train_dataset, n_ekfac_samples)

    def _estimate_corrected_eigenvalues(
        self, train_dataset: Any, n_ekfac_samples: Optional[int]
    ) -> None:
        """Estimate the corrected diagonal eigenvalues for EK-FAC.

        For each layer l, the corrected eigenvalue matrix ``Λ_corrected`` is a
        flattened vector of shape ``(n_out * n_in_eff,)`` where ``n_in_eff``
        includes bias.  Each entry ``(i,j)`` is estimated as:

            E[ (Q_G^T g)_i^2  *  (Q_A^T a)_j^2 ]

        i.e. the expected squared Kronecker-rotated per-sample quantities.
        """
        backend = self.backend
        model = self.model
        layer_map = self.layer_map

        def _to_int(value: Any) -> int:
            """Convert backend scalar values to Python int."""
            try:
                return int(value)
            except TypeError:
                return int(value.numpy())

        # Storage for running sums: layer_idx -> (n_out, n_in_eff) accumulator
        corrected_sums: Dict[int, Any] = {}
        n_rows_per_layer: Dict[int, int] = {}
        n_samples = 0

        # Register hooks (same as factor computation)
        handles = []
        activations: Dict[int, Any] = {}
        grad_outputs_captured: Dict[int, Any] = {}

        for info in layer_map.layers_info:
            idx = info.layer_idx

            def _fwd_hook(layer, inp, out, _idx=idx):
                a = inp if not isinstance(inp, tuple) else inp[0]
                activations[_idx] = a

            def _bwd_hook(layer, grad_inp, grad_out, _idx=idx):
                g = grad_out if not isinstance(grad_out, tuple) else grad_out[0]
                grad_outputs_captured[_idx] = g

            handles.append(backend.register_forward_hook(info.layer, _fwd_hook))
            handles.append(backend.register_backward_hook(info.layer, _bwd_hook))

        try:
            for batch in train_dataset:
                if isinstance(batch, (list, tuple)):
                    batch_tuple = tuple(batch)
                else:
                    batch_tuple = (batch,)

                model_inp, y_true, sample_weight = model.process_batch_for_loss_fn(batch_tuple)
                batch_size = _to_int(backend.get_batch_size(model_inp))
                n_samples += batch_size

                activations.clear()
                grad_outputs_captured.clear()

                self._forward_backward(
                    model, model_inp, y_true, sample_weight,
                    activations, grad_outputs_captured
                )

                for info in layer_map.layers_info:
                    idx = info.layer_idx
                    if idx not in activations or idx not in grad_outputs_captured:
                        continue
                    if idx not in self.Q_A:
                        continue

                    a = activations[idx]
                    g = grad_outputs_captured[idx]
                    a_mat, g_mat = self._prepare_activation_gradient(info, a, g)

                    a_rows = _to_int(backend.get_batch_size(a_mat))
                    g_rows = _to_int(backend.get_batch_size(g_mat))
                    if a_rows != g_rows:
                        raise ValueError(
                            f"Activation/gradient row mismatch for layer {idx}: "
                            f"{a_rows} vs {g_rows}."
                        )

                    n_rows_per_layer[idx] = n_rows_per_layer.get(idx, 0) + a_rows

                    # Rotate into eigenbasis: (batch, n_in_eff) @ Q_A -> (batch, n_in_eff)
                    a_rot = backend.matmul(a_mat, self.Q_A[idx])
                    # (batch, n_out) @ Q_G -> (batch, n_out)
                    g_rot = backend.matmul(g_mat, self.Q_G[idx])

                    # Corrected eigenvalue: E[ g_rot_i^2 * a_rot_j^2 ]
                    # g_rot^2: (batch, n_out), a_rot^2: (batch, n_in_eff)
                    # Outer per sample then average: (batch, n_out, 1) * (batch, 1, n_in_eff)
                    # Sum across batch -> (n_out, n_in_eff)
                    g_sq = backend.multiply(g_rot, g_rot)  # (batch, n_out)
                    a_sq = backend.multiply(a_rot, a_rot)  # (batch, n_in_eff)

                    # (batch, n_out, 1) * (batch, 1, n_in_eff) -> (batch, n_out, n_in_eff)
                    g_sq_exp = backend.expand_dims(g_sq, axis=2)
                    a_sq_exp = backend.expand_dims(a_sq, axis=1)
                    batch_corrected = backend.multiply(g_sq_exp, a_sq_exp)

                    # Sum across batch dimension
                    batch_sum = backend.reduce_sum(batch_corrected, axis=0)
                    if idx not in corrected_sums:
                        corrected_sums[idx] = batch_sum
                    else:
                        corrected_sums[idx] = corrected_sums[idx] + batch_sum

                if n_ekfac_samples is not None and n_samples >= n_ekfac_samples:
                    break
        finally:
            for h in handles:
                backend.remove_hook(h)

        # Normalize and flatten
        for info in layer_map.layers_info:
            idx = info.layer_idx
            if idx not in corrected_sums:
                continue

            n_rows = n_rows_per_layer.get(idx, 0)
            if n_rows <= 0:
                continue

            n_tensor = backend.cast(
                backend.constant(float(n_rows)),
                backend.get_dtype(corrected_sums[idx])
            )
            # (n_out, n_in_eff) -> flatten to (n_out * n_in_eff,)
            corrected = corrected_sums[idx] / n_tensor
            self.Lambda_corrected[idx] = backend.reshape(corrected, (-1,))
