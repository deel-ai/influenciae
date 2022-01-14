import tensorflow as tf

from influenciae.influence.influence_calculator import BaseInfluenceCalculator
from influenciae.common.tf_operations import is_dataset_batched


class FirstOrderInfluenceCalculator(BaseInfluenceCalculator):
    """
    A class implementing the necessary methods to compute the different influence quantities using the first-order
    approximation presented by Koh et al. in
    https://proceedings.neurips.cc/paper/2019/file/a78482ce76496fcf49085f2190e675b4-Paper.pdf

    - Influence function vectors (the actual value of the weights when removing the datapoints in question)
    - Influence values/Cook's distance (a measure of reliance of the model on the individual datapoints)
    - Group's influence function vectors (the actual value of weights when removing an entire group of datapoints)
    - Group's influence values (a measure of reliance of the model on an entire group of points)
    """
    def compute_influence(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Computes the influence function vector (an estimation of the value of the weights when removing the
        datapoints) of each point individually.

        Args:
            dataset: tf.data.Dataset
                A batched TF dataset containing the points of which we wish to compute the influence of removal.

        Returns:
            influence_vectors: tf.Tensor
                A tensor containing one vector per input point
        """
        if not is_dataset_batched(dataset):
            raise ValueError("The dataset must be batched before performing this operation.")
        influence_vectors = self.ihvp_calculator.compute_ihvp(dataset)
        influence_vectors = tf.transpose(influence_vectors)

        return influence_vectors

    def compute_influence_values(
            self,
            dataset_train: tf.data.Dataset,
            dataset_to_evaluate: tf.data.Dataset
    ) -> tf.Tensor:
        """
        Computes Cook's distance of each datapoint provided individually, giving measure of the influence
        that each point carries on the model's weights.

        In particular, as defined by Koh et al., these influence values are computed as followed:

        influence_value = \nabla_{\theta} \ell(\theta, dataset_to_evaluate).T @ inv_hessian @ \ell(\theta, dataset_train)

        The dataset_train contains the points we will be removing and dataset_to_evaluate, those with respect to whom
        we will be measuring the influence. As we will be performing the same operation in batches, we consider that
        each point from one dataset corresponds to one from the other. As such, both datasets must contain the same
        amount of points

        Args:
            dataset_train: tf.data.Dataset
                A batched TF dataset containing the points we wish to remove
            dataset_to_evaluate: tf.data.Dataset
                A batched TF dataset containing the points with respect to whom we wish to measure the influence of
                removing the training points

        Returns:
            influence_values: tf.Tensor
                A tensor containing one influence value per pair of input values (one coming from each dataset)
        """
        if not is_dataset_batched(dataset_train) or not is_dataset_batched(dataset_to_evaluate):
            raise ValueError("Both datasets must be batched before performing this operation.")
        if dataset_train.cardinality().numpy() * dataset_train._batch_size != dataset_to_evaluate.cardinality().numpy() * dataset_to_evaluate._batch_size:
            raise ValueError("The amount of points in the train and evaluation groups must match.")
        grads = self.model.batch_jacobian(dataset_to_evaluate)
        grads = tf.reshape(grads, (dataset_train.cardinality().numpy() * dataset_train._batch_size, -1))
        ihvp = self.ihvp_calculator.compute_ihvp(dataset_train)
        influence_values = tf.reduce_sum(tf.math.multiply(grads, tf.transpose(ihvp)), axis=1, keepdims=True)  # performs the batched matmul

        return influence_values

    def compute_influence_group(
            self,
            group: tf.data.Dataset
    ) -> tf.Tensor:
        """
        Computes the influence function vector (an estimation of the value of the weights when removing the
        datapoints) of the whole group.

        Args:
            group: tf.data.Dataset
                A batched TF dataset containing the group of points of which we wish to compute the influence
                of removal

        Returns:
            influence_group: tf.Tensor
                A tensor containing one vector for the whole group
        """
        if not is_dataset_batched(group):
            raise ValueError("The dataset must be batched before performing this operation.")
        ihvp = self.ihvp_calculator.compute_ihvp(group)
        reduced_ihvp = tf.reduce_sum(ihvp, axis=1)
        influence_group = tf.reshape(reduced_ihvp, (1, -1))

        return influence_group

    def compute_influence_values_group(
            self,
            group_train: tf.data.Dataset,
            group_to_evaluate: tf.data.Dataset
    ) -> tf.Tensor:
        """
        Computes Cook's distance of the whole group of datapoints provided, giving measure of the influence
        that the group carries on the model's weights.

        In particular, as defined by Koh et al., these influence values are computed as followed:

        influence_value = \nabla_{\theta} \ell(\theta, dataset_to_evaluate).T @ inv_hessian @ \ell(\theta, dataset_train)

        The dataset_train contains the points we will be removing and dataset_to_evaluate, those with respect to whom
        we will be measuring the influence. As we will be performing the same operation in batches, we consider that
        each point from one dataset corresponds to one from the other. As such, both datasets must contain the same
        amount of points

        Args:
            group_train: tf.data.Dataset
                A batched TF dataset containing the group of points we wish to remove
            group_to_evaluate: tf.data.Dataset
                A batched TF dataset containing the group of points with respect to whom we wish to measure the
                influence of removing the training points

        Returns:
            influence_values_group: tf.Tensor
                A tensor containing one influence value for the whole group
        """
        if not is_dataset_batched(group_train) or not is_dataset_batched(group_to_evaluate):
            raise ValueError("Both datasets must be batched before performing this operation.")
        if group_train.cardinality().numpy() * group_train._batch_size != group_to_evaluate.cardinality().numpy() * group_to_evaluate._batch_size:
            raise ValueError("The amount of points in the train and evaluation groups must match.")
        reduced_grads = tf.reduce_sum(tf.reshape(self.model.batch_jacobian(group_to_evaluate),
                                                 (group_train.cardinality().numpy() * group_train._batch_size, -1)),
                                      axis=0, keepdims=True)
        reduced_ihvp = tf.reduce_sum(self.ihvp_calculator.compute_ihvp(group_train), axis=1, keepdims=True)
        influence_values_group = tf.matmul(reduced_grads, reduced_ihvp)

        return influence_values_group
