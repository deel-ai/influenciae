# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a technique based on the representer point theorem for kernels,
but using a local jacobian expansion, as per
https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf

Supports both TensorFlow and PyTorch models through the backend abstraction layer.
"""
import copy
from typing import Any, Optional, Union

from .._optional_imports import import_optional_attr, import_optional_module
from .base_representer_point import BaseRepresenterPoint
from ..common import InfluenceModel, InverseHessianVectorProductFactory, Framework
from ..types import DatasetLike


class RepresenterPointLJE(BaseRepresenterPoint):
    """
    Representer Point Selection via Local Jacobian Expansion for Post-hoc Classifier Explanation of Deep Neural
    Networks and Ensemble Models
    https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf

    Supports both TensorFlow and PyTorch models through the backend abstraction layer.

    Disclaimer: This technique requires the last layer of the model to be a Dense/Linear layer with no bias.

    Parameters
    ----------
    influence_model
        The model implementing the InfluenceModel interface (TensorFlow or PyTorch).
    dataset
        A batched dataset with the points with which the model was trained.
    ihvp_calculator_factory
        An InverseHessianVectorProductFactory for creating new instances of the InverseHessianVectorProduct
        class.
    n_samples_for_hessian
        An integer for the amount of samples from the training dataset that will be used for the computation of the
        hessian matrix.
        If None, the whole dataset will be used.
    target_layer
        Either a string or an integer identifying the layer on which to compute the influence-related quantities.
    shuffle_buffer_size
        An integer with the buffer size for the training set's shuffle operation (TensorFlow only).
    epsilon
        An epsilon value to prevent division by zero.
    """
    def __init__(
            self,
            influence_model: InfluenceModel,
            dataset: DatasetLike,
            ihvp_calculator_factory: InverseHessianVectorProductFactory,
            n_samples_for_hessian: Optional[int] = None,
            target_layer: Union[int, str] = -1,
            shuffle_buffer_size: int = 10000,
            epsilon: float = 1e-5
    ):
        super().__init__(influence_model.model, dataset, influence_model.loss_function)
        self.epsilon = epsilon

        if self.backend.framework == Framework.TENSORFLOW:
            self._init_tensorflow(influence_model, dataset, ihvp_calculator_factory,
                                  n_samples_for_hessian, target_layer, shuffle_buffer_size)
        else:
            self._init_pytorch(influence_model, dataset, ihvp_calculator_factory,
                               n_samples_for_hessian, target_layer)

    def _init_tensorflow(
            self,
            influence_model: InfluenceModel,
            dataset: DatasetLike,
            ihvp_calculator_factory: InverseHessianVectorProductFactory,
            n_samples_for_hessian: Optional[int],
            _target_layer: Union[int, str],
            shuffle_buffer_size: int
    ):
        """TensorFlow-specific initialization."""
        tf = import_optional_module("tensorflow", extra="tensorflow")

        self.epsilon_tensor = tf.constant(self.epsilon, dtype=tf.float32)

        # In the paper, the authors explain that in practice, they use a single step of SGD to compute the
        # perturbed model's weights. We will do the same here.
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
        target_layer_shape = self.feature_extractor.output_shape
        perturbed_head = tf.keras.models.clone_model(self.original_head)
        perturbed_head.set_weights(self.original_head.get_weights())
        perturbed_head.build(target_layer_shape)
        perturbed_head.compile(optimizer=optimizer, loss=influence_model.loss_function)

        # Get a dataset to compute the SGD step
        batch_size = self.backend.get_dataset_batch_size(dataset)
        if n_samples_for_hessian is None:
            dataset_to_estimate_hessian = dataset
        else:
            n_batches_for_hessian = max(n_samples_for_hessian // batch_size, 1)
            shuffled_dataset = self.backend.shuffle_dataset(dataset, shuffle_buffer_size)
            dataset_to_estimate_hessian = self.backend.take_dataset(shuffled_dataset, n_batches_for_hessian)
        f_array, y_array = None, None
        for x, y in dataset_to_estimate_hessian:
            f = self.feature_extractor(x)
            f_array = f if f_array is None else tf.concat([f_array, f], axis=0)
            y_array = y if y_array is None else tf.concat([y_array, y], axis=0)
        dataset_to_estimate_hessian = tf.data.Dataset.from_tensor_slices((f_array, y_array)).batch(batch_size)

        # Accumulate the gradients for the whole dataset and then update
        trainable_vars = perturbed_head.trainable_variables
        accum_vars = [tf.Variable(tf.zeros_like(t_var), trainable=False) for t_var in trainable_vars]
        total_samples = 0
        for x, y in dataset_to_estimate_hessian:
            with tf.GradientTape() as tape:
                y_pred = perturbed_head(x)
                loss = self.loss_function(y, y_pred)
                loss = self._ensure_per_sample_loss_tensorflow(loss, tf)
                loss = -tf.reduce_mean(loss)
            gradients = tape.gradient(loss, trainable_vars)

            batch_size_tensor = tf.shape(x)[0]
            batch_size = int(batch_size_tensor.numpy())
            total_samples += batch_size

            for i, grad in enumerate(gradients):
                if grad is None:
                    raise ValueError("Gradient is None while computing perturbed-head update for RPS-LJE")
                accum_vars[i].assign_add(grad * tf.cast(batch_size, grad.dtype))

        if total_samples == 0:
            raise ValueError("Dataset used for Hessian estimation is empty")

        mean_grads = [accum_var / tf.cast(total_samples, accum_var.dtype) for accum_var in accum_vars]
        optimizer.apply_gradients(zip(mean_grads, trainable_vars))

        # Keep the perturbed head
        self.perturbed_head = perturbed_head

        # Create the new model with the perturbed weights to compute the hessian matrix
        model = InfluenceModel(
            self.perturbed_head,
            start_layer=None,
            loss_function=influence_model.loss_function
        )
        self.ihvp_calculator = ihvp_calculator_factory.build(model, dataset_to_estimate_hessian)

    def _init_pytorch(
            self,
            influence_model: InfluenceModel,
            dataset: DatasetLike,
            ihvp_calculator_factory: InverseHessianVectorProductFactory,
            n_samples_for_hessian: Optional[int],
            _target_layer: Union[int, str]
    ):
        """PyTorch-specific initialization."""
        torch = import_optional_module("torch", extra="pytorch")
        data_loader_cls = import_optional_attr("torch.utils.data", "DataLoader", extra="pytorch")
        tensor_dataset_cls = import_optional_attr("torch.utils.data", "TensorDataset", extra="pytorch")

        device = next(influence_model.model.parameters()).device

        # Clone the original head for perturbation
        perturbed_head = copy.deepcopy(self.original_head)
        perturbed_head.to(device)
        perturbed_head.train()

        # Use a single step of SGD to compute the perturbed model's weights
        optimizer = torch.optim.SGD(perturbed_head.parameters(), lr=1e-4)

        # Get a dataset to compute the SGD step
        f_list, y_list = [], []
        n_samples_seen = 0
        with torch.no_grad():
            for batch in dataset:
                x = batch[0].to(device)
                y = batch[-1].to(device)
                f = self.feature_extractor(x)
                f_list.append(f)
                y_list.append(y)
                n_samples_seen += x.shape[0]
                if n_samples_for_hessian is not None and n_samples_seen >= n_samples_for_hessian:
                    break

        f_array = torch.cat(f_list, dim=0)
        y_array = torch.cat(y_list, dim=0)

        # Get the batch size from the original dataset
        batch_size = self.backend.get_dataset_batch_size(dataset)
        dataset_to_estimate_hessian = data_loader_cls(
            tensor_dataset_cls(f_array, y_array),
            batch_size=batch_size,
            shuffle=False
        )

        # Accumulate the gradients for the whole dataset and then update
        optimizer.zero_grad()
        dtype = next(perturbed_head.parameters()).dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_samples = 0
        for f_batch, y_batch in dataset_to_estimate_hessian:
            f_batch = f_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = perturbed_head(f_batch)

            per_sample_loss = influence_model.loss_function(y_pred, y_batch)
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.view(per_sample_loss.shape[0], -1).sum(dim=1)
            elif per_sample_loss.dim() == 0:
                raise ValueError("Loss function must return per-sample losses (reduction='none')")

            total_loss = total_loss - per_sample_loss.sum()
            total_samples += int(f_batch.shape[0])

        if total_samples == 0:
            raise ValueError("Dataset used for Hessian estimation is empty")

        (total_loss / float(total_samples)).backward()
        optimizer.step()

        # Set perturbed head to eval mode
        perturbed_head.eval()
        self.perturbed_head = perturbed_head

        # Create the new model with the perturbed weights to compute the hessian matrix
        model = InfluenceModel(
            self.perturbed_head,
            0,  # Start layer for the head model
            loss_function=influence_model.loss_function
        )
        self.ihvp_calculator = ihvp_calculator_factory.build(model, dataset_to_estimate_hessian)

    def _compute_alpha(self, z_batch: Any, y_batch: Any) -> Any:
        """
        Computes the alpha vector for the Local Jacobian Expansion approximation.

        Parameters
        ----------
        z_batch
            A tensor with the perturbed model's predictions.
        y_batch
            A tensor with the ground truth labels.

        Returns
        -------
        A tensor with the alpha vector for the Local Jacobian Expansion approximation.
        """
        if self.backend.framework == Framework.TENSORFLOW:
            return self._compute_alpha_tensorflow(z_batch, y_batch)
        return self._compute_alpha_pytorch(z_batch, y_batch)

    def _compute_alpha_tensorflow(self, z_batch: Any, y_batch: Any) -> Any:
        """TensorFlow-specific alpha computation."""
        tf = import_optional_module("tensorflow", extra="tensorflow")

        # First, we compute the second term, which contains the Hessian vector product
        weights = self.backend.normalize_weights_to_watch(list(self.perturbed_head.trainable_weights))
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(weights)
            logits = self.perturbed_head(z_batch)
            loss = self.loss_function(y_batch, logits)
            loss = self._ensure_per_sample_loss_tensorflow(loss, tf)
        grads = tape.jacobian(loss, weights)[0]
        grads = tf.multiply(
            grads,
            tf.repeat(
                tf.expand_dims(
                    tf.divide(tf.ones_like(z_batch),
                              tf.cast(tf.shape(z_batch)[0], z_batch.dtype) * z_batch +
                              tf.cast(self.epsilon, z_batch.dtype)),
                    axis=-1),
                grads.shape[-1], axis=-1
            )
        )
        second_term = tf.map_fn(
            lambda v: self.ihvp_calculator._compute_ihvp_single_batch(  # pylint: disable=protected-access
                tf.expand_dims(v, axis=0),
                use_gradient=False
            ),
            grads
        )
        second_term = tf.reduce_sum(tf.reshape(second_term, tf.shape(grads)), axis=1)

        # Second, we compute the first term, which contains the weights
        first_term = tf.concat(weights, axis=0)
        first_term = tf.multiply(
            first_term,
            tf.repeat(
                tf.expand_dims(
                    tf.divide(tf.ones_like(z_batch),
                              tf.cast(tf.shape(z_batch)[0], z_batch.dtype) * z_batch +
                              tf.cast(self.epsilon, z_batch.dtype)),
                    axis=-1),
                first_term.shape[-1], axis=-1
            )
        )
        first_term = tf.reduce_sum(first_term, axis=1)

        return first_term - second_term  # alpha is first term minus second term

    def _compute_alpha_pytorch(self, z_batch: Any, y_batch: Any) -> Any:
        """PyTorch-specific alpha computation."""
        torch = import_optional_module("torch", extra="pytorch")

        device = z_batch.device
        dtype = z_batch.dtype
        y_batch = y_batch.to(device=device)
        batch_size = z_batch.shape[0]

        # Get the weights from the perturbed head
        weights_list = [p for p in self.perturbed_head.parameters() if p.requires_grad]

        # First, we compute the second term, which contains the Hessian vector product
        logits = self.perturbed_head(z_batch)
        loss = self.loss_function(logits, y_batch)
        # Ensure loss is per-sample
        if loss.dim() > 1:
            loss = loss.view(batch_size, -1).sum(dim=1)
        elif loss.dim() == 0:
            raise ValueError("Loss function must return per-sample losses (reduction='none')")

        # Compute per-sample gradients
        grads_list = []
        for i in range(batch_size):
            grad_i = torch.autograd.grad(loss[i], weights_list, retain_graph=True, create_graph=False)
            grad_flat = torch.cat([g.view(-1) for g in grad_i])
            grads_list.append(grad_flat)
        grads = torch.stack(grads_list, dim=0)  # (batch_size, num_params)

        # Reshape grads to match weight shape (batch_size, in_features, out_features)
        # Assuming a single linear layer weight of shape (out_features, in_features)
        weight_shape = weights_list[0].shape  # (out_features, in_features)
        out_features, in_features = weight_shape
        grads_reshaped = grads.view(batch_size, out_features, in_features).permute(0, 2, 1)  # (batch, in, out)

        # Divide by feature maps (z_batch): (batch, in)
        eps = torch.tensor(self.epsilon, device=device, dtype=dtype)
        divisor = batch_size * z_batch + eps  # (batch, in)
        divisor_expanded = divisor.unsqueeze(-1)  # (batch, in, 1)
        grads_divided = grads_reshaped / divisor_expanded  # (batch, in, out)

        # Compute IHVP for each sample
        second_term_list = []
        for i in range(batch_size):
            # Flatten and expand dims for IHVP computation
            grad_flat = grads_divided[i].permute(1, 0).reshape(1, -1)  # (1, out*in)
            ihvp_result = self.ihvp_calculator._compute_ihvp_single_batch(  # pylint: disable=protected-access
                (grad_flat,),
                use_gradient=False
            )
            # ihvp_result shape: (num_params, 1) -> flatten
            second_term_list.append(ihvp_result.squeeze())
        second_term = torch.stack(second_term_list, dim=0)  # (batch, num_params)

        # Reshape second_term back to (batch, in, out) and sum over in_features
        second_term_reshaped = second_term.view(batch_size, out_features, in_features).permute(0, 2, 1)
        second_term_summed = second_term_reshaped.sum(dim=1)  # (batch, out)

        # Compute the first term: weights divided by feature maps
        # weights_list[0] is (out_features, in_features), transpose to (in, out)
        weights_transposed = weights_list[0].T  # (in, out)
        weights_expanded = weights_transposed.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, in, out)
        first_term = weights_expanded / divisor_expanded  # (batch, in, out)
        first_term_summed = first_term.sum(dim=1)  # (batch, out)

        return first_term_summed - second_term_summed  # alpha is first term minus second term
