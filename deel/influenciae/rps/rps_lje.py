# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a technique based on the representer point theorem for kernels,
but using a local jacobian expansion, as per
https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf
"""
import tensorflow as tf

from .base_representer_point import BaseRepresenterPoint
from ..common import InfluenceModel, InverseHessianVectorProductFactory
from ..types import Union, Optional


class RepresenterPointLJE(BaseRepresenterPoint):
    """
    Representer Point Selection via Local Jacobian Expansion for Post-hoc Classifier Explanation of Deep Neural
    Networks and Ensemble Models
    https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf

    Disclaimer: This technique requires the last layer of the model to be a Dense layer with no bias.

    Parameters
    ----------
    influence_model
        The TF2.X model implementing the InfluenceModel interface.
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
        An integer with the buffer size for the training set's shuffle operation.
    epsilon
        An epsilon value to prevent division by zero.
    """
    def __init__(
            self,
            influence_model: InfluenceModel,
            dataset: tf.data.Dataset,
            ihvp_calculator_factory: InverseHessianVectorProductFactory,
            n_samples_for_hessian: Optional[int] = None,
            target_layer: Union[int, str] = -1,
            shuffle_buffer_size: int = 10000,
            epsilon: float = 1e-5
    ):
        super().__init__(influence_model.model, dataset, influence_model.loss_function)
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)

        # In the paper, the authors explain that in practice, they use a single step of SGD to compute the
        # perturbed model's weights. We will do the same here.
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
        target_layer_shape = influence_model.model.layers[target_layer].input.type_spec.shape
        perturbed_head = tf.keras.models.clone_model(self.original_head)
        perturbed_head.set_weights(self.original_head.get_weights())
        perturbed_head.build(target_layer_shape)
        perturbed_head.compile(optimizer=optimizer, loss=influence_model.loss_function)

        # Get a dataset to compute the SGD step
        if n_samples_for_hessian is None:
            dataset_to_estimate_hessian = dataset
        else:
            n_batches_for_hessian = max(n_samples_for_hessian // dataset._batch_size, 1)
            dataset_to_estimate_hessian = dataset.shuffle(shuffle_buffer_size).take(n_batches_for_hessian)
        f_array, y_array = None, None
        for x, y in dataset_to_estimate_hessian:
            f = self.feature_extractor(x)
            f_array = f if f_array is None else tf.concat([f_array, f], axis=0)
            y_array = y if y_array is None else tf.concat([y_array, y], axis=0)
        dataset_to_estimate_hessian = tf.data.Dataset.from_tensor_slices((f_array, y_array)).batch(dataset._batch_size)

        # Accumulate the gradients for the whole dataset and then update
        trainable_vars = perturbed_head.trainable_variables
        accum_vars = [tf.Variable(tf.zeros_like(t_var.read_value()), trainable=False)
                      for t_var in trainable_vars]
        for x, y in dataset_to_estimate_hessian:
            with tf.GradientTape() as tape:
                y_pred = perturbed_head(x)
                loss = -perturbed_head.loss(y, y_pred)
            gradients = tape.gradient(loss, trainable_vars)
            _ = [accum_vars[i].assign_add(grad) for i, grad in enumerate(gradients)]
        optimizer.apply_gradients(zip(accum_vars, trainable_vars))

        # Keep the perturbed head
        self.perturbed_head = perturbed_head

        # Create the new model with the perturbed weights to compute the hessian matrix
        model = InfluenceModel(
            self.perturbed_head,
            1,  # layer 0 is InputLayer
            loss_function=influence_model.loss_function
        )
        self.ihvp_calculator = ihvp_calculator_factory.build(model, dataset_to_estimate_hessian)

    def _compute_alpha(self, z_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
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
        # First, we compute the second term, which contains the Hessian vector product
        weights = self.perturbed_head.trainable_weights
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(weights)
            logits = self.perturbed_head(z_batch)
            loss = self.perturbed_head.compiled_loss(y_batch, logits)
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
        first_term = tf.concat(list(weights), axis=0)
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
