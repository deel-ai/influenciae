# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing the representer point theorem for kernels for estimating the
influence of training data-points, as per:
https://arxiv.org/abs/1811.09720

Supports both TensorFlow and PyTorch models through the backend abstraction layer.
"""
from .base_representer_point import BaseRepresenterPoint
from ..common import Framework
from ..types import Tuple, Callable, Union, Any, Optional


class RepresenterPointL2(BaseRepresenterPoint):
    """
    A class implementing a method to compute the influence of training points through
    the representer point theorem for kernels.

    It builds a kernel that approximates the model's last layer such that for a training
    point x_i and a test point x_t with label y_t:

    y_t = sum_i k(alpha_i, x_i, x_t)

    Supports both TensorFlow and PyTorch models through the backend abstraction layer.

    Disclaimer: This method only works on classification problems!

    Parameters
    ----------
    model
        A model that has already been trained (TensorFlow or PyTorch).
    train_set
        A batched dataset with the points with which the model was trained.
    loss_function
        The loss function with which the model was trained. This loss function MUST NOT be reduced.
    lambda_regularization
        The coefficient for the regularization of the surrogate last layer that needs
        to be trained for this method.
    scaling_factor
        A float with the scaling factor for the SGD backtracking line-search optimizer
        for fitting the surrogate linear model.
    epochs
        An integer for the amount of epochs to fit the linear model.
    layer_index
        Layer index of the logits (-1 for last layer by default).
    """

    def __init__(
            self,
            model: Any,
            train_set: Any,
            loss_function: Union[Callable, Any],
            lambda_regularization: float,
            scaling_factor: float = 0.1,
            epochs: int = 100,
            layer_index: int = -1,
    ):
        super().__init__(model, train_set, loss_function, layer_index)
        self.n_train = self.backend.get_dataset_size(train_set)
        self.train_set = train_set
        self.lambda_regularization = lambda_regularization
        self.scaling_factor = scaling_factor
        self.epochs = epochs
        self.linear_layer: Optional[Any] = None
        self._train_last_layer(self.epochs)

    def _train_last_layer(self, epochs: int):
        """
        Trains an L2-regularized surrogate linear model to predict like the model on the
        training dataset. The optimization is done using a Backtracking Line-Search
        algorithm with the Armijo condition and SGD as the optimizer as it was done
        in the original implementation.

        Parameters
        ----------
        epochs
            An integer with the amount of epochs to train the surrogate model.
        """
        if self.backend.framework == Framework.TENSORFLOW:
            self._train_last_layer_tensorflow(epochs)
        else:
            self._train_last_layer_pytorch(epochs)

    def _train_last_layer_tensorflow(self, epochs: int):
        """TensorFlow-specific training using BacktrackingLineSearch optimizer."""
        import tensorflow as tf
        from tensorflow.keras.losses import MeanSquaredError
        from ..utils import BacktrackingLineSearch

        self.linear_layer = self._create_surrogate_model_tensorflow()
        optimizer = BacktrackingLineSearch(
            batches_per_epoch=int(self.n_train / self.backend.get_dataset_batch_size(self.train_set)),
            scaling_factor=self.scaling_factor
        )
        mse_loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        self.linear_layer.compile(optimizer=optimizer, loss=mse_loss)
        assert self.linear_layer is not None  # Type narrowing for mypy
        for _ in range(epochs):
            for x_batch, _ in self.train_set:
                loss, grads, z_batch, y_target = self._learn_step_last_layer_tensorflow(x_batch, mse_loss)
                optimizer.step(self.linear_layer, loss, z_batch, y_target, grads)

        self.linear_layer.compile(optimizer=optimizer, loss=self.loss_function)

    def _train_last_layer_pytorch(self, epochs: int):
        """PyTorch-specific training using BacktrackingLineSearchPyTorch optimizer."""
        import torch
        import torch.nn as nn
        from ..utils import BacktrackingLineSearchPyTorch

        device = next(self.model.parameters()).device
        self.linear_layer = self._create_surrogate_model_pytorch().to(device)
        assert self.linear_layer is not None  # Type narrowing for mypy
        mse_loss = nn.MSELoss(reduction="mean")

        # Create the backtracking line search optimizer
        optimizer = BacktrackingLineSearchPyTorch(
            params=self.linear_layer.parameters(),
            batches_per_epoch=int(self.n_train / len(next(iter(self.train_set))[0])),
            scaling_factor=self.scaling_factor
        )

        W = self.linear_layer.weight  # only parameter (bias=False)

        self.linear_layer.train()
        for _ in range(epochs):
            for batch in self.train_set:
                x_batch = batch[0].to(device)

                # Feature maps + teacher logits (no grad)
                with torch.no_grad():
                    z_batch = self.feature_extractor(x_batch)
                    y_target = self.original_head(z_batch)

                # Forward + loss (with L2 reg like Keras L2: lambda * sum(W^2))
                optimizer.zero_grad()
                logits = self.linear_layer(z_batch)
                mse = mse_loss(logits, y_target)
                reg = self.lambda_regularization * (W.pow(2).sum())
                loss = mse + reg

                # If already non-finite, skip update (defensive)
                if not torch.isfinite(loss):
                    continue

                loss.backward()
                g = W.grad
                if g is None or not torch.isfinite(g).all():
                    optimizer.zero_grad()
                    continue

                # Clip to avoid rare spikes
                torch.nn.utils.clip_grad_norm_([W], max_norm=10.0)

                # Collect gradients for the optimizer
                gradients = [p.grad.clone() for p in self.linear_layer.parameters() if p.grad is not None]

                # Define closure for loss re-evaluation
                def closure():
                    assert self.linear_layer is not None  # Already checked earlier
                    with torch.no_grad():
                        logits_new = self.linear_layer(z_batch)
                        mse_new = mse_loss(logits_new, y_target)
                        reg_new = self.lambda_regularization * (W.pow(2).sum())
                        return mse_new + reg_new

                # Perform backtracking line search step
                optimizer.step(
                    model=self.linear_layer,
                    current_loss=loss.detach(),
                    x_inputs=z_batch,
                    labels=y_target,
                    gradients=gradients,
                    closure=closure
                )

        self.linear_layer.eval()

    def _learn_step_last_layer_tensorflow(self, x_batch: Any, mse_loss: Any) -> Tuple[Any, Any, Any, Any]:
        """
        TensorFlow-specific learning step for the surrogate linear model.

        Parameters
        ----------
        x_batch
            A training sample wrt to which we wish to compute the gradients.
        mse_loss
            A callable that computes the MSE loss.

        Returns
        -------
        loss, gradients, z_batch, y_target
            Tuple with loss value, gradients, latent space, and target predictions.
        """
        import tensorflow as tf

        assert self.linear_layer is not None  # Initialized in __init__
        z_batch = self.feature_extractor(x_batch)
        y_target = self.model.layers[-1](z_batch)
        with tf.GradientTape() as tape:
            logits = self.linear_layer(z_batch, training=True)
            loss = mse_loss(y_target, logits)
        gradients = tape.gradient(loss, self.linear_layer.trainable_weights)
        return loss, gradients, z_batch, y_target

    def _create_surrogate_model_tensorflow(self) -> Any:
        """
        Creates an L2-regularized linear model for TensorFlow.

        Returns
        -------
        surrogate_model
            A TensorFlow L2-regularized linear model.
        """
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.regularizers import L2

        inputs = Input(shape=self.feature_extractor.output_shape[1:], dtype=self.model.output.dtype)
        last_layer = Dense(
            self.model.output_shape[-1],
            use_bias=False,
            kernel_regularizer=L2(self.lambda_regularization),
            dtype=self.model.output.dtype
        )
        outputs = last_layer(inputs)
        surrogate_model = Model(inputs=inputs, outputs=outputs)
        surrogate_model.layers[-1].trainable = True
        surrogate_model.compile(loss=self.model.compiled_loss)

        return surrogate_model

    def _create_surrogate_model_pytorch(self) -> Any:
        """
        Creates an L2-regularized linear model for PyTorch.

        Returns
        -------
        surrogate_model
            A PyTorch L2-regularized linear model.
        """
        import torch.nn as nn

        # Get input and output dimensions from the feature extractor and model
        children = list(self.model.children())
        last_layer = children[-1]

        if hasattr(last_layer, 'in_features') and hasattr(last_layer, 'out_features'):
            in_features = last_layer.in_features
            out_features = last_layer.out_features
        else:
            raise ValueError("Could not determine input/output dimensions for surrogate model")

        # Create a simple linear layer without bias
        surrogate_model = nn.Linear(in_features, out_features, bias=False)

        # Move to same device as the original model
        device = next(self.model.parameters()).device
        surrogate_model = surrogate_model.to(device)

        return surrogate_model

    def _create_surrogate_model(self) -> Any:
        """
        Creates an L2-regularized linear model to use as surrogate with the
        right input and output shapes.

        Returns
        -------
        surrogate_model
            An L2-regularized linear model.
        """
        if self.backend.framework == Framework.TENSORFLOW:
            return self._create_surrogate_model_tensorflow()
        else:
            return self._create_surrogate_model_pytorch()

    def _compute_alpha(self, z_batch: Any, y_batch: Any) -> Any:
        """
        Computes the alpha factor for the kernel approximation. This element gives a notion of
        the resistance that each training data-point towards minimizing the norm of the linear
        layer's weight matrix. This is essentially this method's notion of influence score.

        Parameters
        ----------
        z_batch
            A training sample wrt to which we wish to compute the gradients.
        y_batch
            Label associated to the training sample.

        Returns
        -------
        alpha
            The alpha coefficients representing the influence score.
        """
        if self.backend.framework == Framework.TENSORFLOW:
            return self._compute_alpha_tensorflow(z_batch, y_batch)
        else:
            return self._compute_alpha_pytorch(z_batch, y_batch)

    def _compute_alpha_tensorflow(self, z_batch: Any, y_batch: Any) -> Any:
        """TensorFlow-specific alpha computation."""
        import tensorflow as tf

        assert self.linear_layer is not None  # Initialized in __init__
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(self.linear_layer.weights)
            logits = self.linear_layer(z_batch)
            loss = self.linear_layer.compiled_loss(y_batch, logits)
        alpha = tape.jacobian(loss, self.linear_layer.weights)[0]
        alpha = tf.divide(
            alpha,
            -2. * self.lambda_regularization * tf.cast(self.n_train, alpha.dtype) + tf.constant(1e-5, dtype=alpha.dtype)
        )

        # Now, divide each of the alpha_i by their feature maps
        alpha = tf.multiply(
            alpha,
            tf.repeat(
                tf.expand_dims(
                    tf.divide(tf.ones_like(z_batch), z_batch + tf.constant(1e-5, dtype=alpha.dtype)),
                    axis=-1),
                alpha.shape[-1], axis=-1
            )
        )
        alpha = tf.reduce_sum(alpha, axis=1)

        return alpha

    def _compute_alpha_pytorch(self, z_batch: Any, y_batch: Any) -> Any:
        """PyTorch-specific alpha computation (stable + matches TF intent)."""
        import torch

        assert self.linear_layer is not None  # Initialized in __init__
        device = z_batch.device
        dtype = z_batch.dtype
        y_batch = y_batch.to(device=device)

        W = self.linear_layer.weight  # (out_features, in_features)

        # Forward
        logits = self.linear_layer(z_batch)

        # Per-sample loss; make sure each sample is a scalar
        loss = self.loss_function(logits, y_batch)
        # BCEWithLogitsLoss(reduction='none') may return (B,1); ensure (B,)
        loss = loss.view(loss.shape[0], -1).sum(dim=1)

        # Per-sample gradients wrt W: shape (B, out, in)
        grads = []
        for i in range(loss.shape[0]):
            gi = torch.autograd.grad(loss[i], W, retain_graph=True, create_graph=False)[0]
            grads.append(gi)
        alpha = torch.stack(grads, dim=0)

        # Scale by (-2 * lambda * n_train + eps) in *same dtype*
        eps = torch.tensor(1e-5, device=device, dtype=dtype)
        denom = (-2.0 * self.lambda_regularization * float(self.n_train))
        denom = torch.tensor(denom, device=device, dtype=dtype) + eps
        alpha = alpha / denom

        # Divide by feature maps (TF does 1/(z + eps)); avoid pathological tiny denominators
        z_denom = z_batch + eps
        z_denom = torch.where(z_denom.abs() < eps, eps * torch.ones_like(z_denom), z_denom)
        z_inv = 1.0 / z_denom  # (B, in)

        # alpha: (B, out, in) * (B, 1, in) -> sum over in -> (B, out)
        alpha = (alpha * z_inv.unsqueeze(1)).sum(dim=2)

        return alpha

    def predict_with_kernel(self, samples_to_evaluate: Tuple[Any, ...]) -> Any:
        """
        Uses the learned kernel to approximate the model's predictions on a group of samples.

        Parameters
        ----------
        samples_to_evaluate
            A single batch of tensors with the samples for which we wish to approximate the model's
            predictions.

        Returns
        -------
        predictions
            A tensor with an approximation of the model's predictions.
        """
        if self.backend.framework == Framework.TENSORFLOW:
            return self._predict_with_kernel_tensorflow(samples_to_evaluate)
        else:
            return self._predict_with_kernel_pytorch(samples_to_evaluate)

    def _predict_with_kernel_tensorflow(self, samples_to_evaluate: Tuple[Any, ...]) -> Any:
        """TensorFlow-specific kernel prediction."""
        import tensorflow as tf

        influence_vectors = self.compute_influence_vector(self.train_set)
        _, dataset_influence = self._estimate_inf_values_with_inf_vect_dataset(influence_vectors, samples_to_evaluate)
        dataset_influence = dataset_influence.map(lambda x, v: v)
        dataset_iterator = iter(dataset_influence)

        def body_fun(i, value):
            v = next(dataset_iterator)
            i = i + 1
            value = tf.cast(value, v.dtype) + tf.reduce_sum(v, axis=1)
            return i, value

        _, predictions = tf.while_loop(
            lambda i, value: i < dataset_influence.cardinality(),
            body_fun,
            [tf.constant(0, dtype=tf.int64),
             tf.zeros((tf.shape(samples_to_evaluate[-1])[0],), dtype=tf.float32)]
        )

        return predictions

    def _predict_with_kernel_pytorch(self, samples_to_evaluate: Tuple[Any, ...]) -> Any:
        """PyTorch-specific kernel prediction."""
        import torch

        influence_vectors = self.compute_influence_vector(self.train_set)
        _, dataset_influence = self._estimate_inf_values_with_inf_vect_dataset(influence_vectors, samples_to_evaluate)

        predictions = None
        for _, v in dataset_influence:
            batch_pred = torch.sum(v, dim=1)
            if predictions is None:
                predictions = batch_pred
            else:
                predictions = predictions + batch_pred

        return predictions
