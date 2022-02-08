"""
Inverse Hessian Vector Product (ihvp) module
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras import Model

from ..common import InfluenceModel, is_dataset_batched

from ..types import Optional
from ..common import assert_batched_dataset


class InverseHessianVectorProduct(ABC):
    def __init__(self, model: InfluenceModel, train_dataset: tf.data.Dataset):
        """
        An interface for classes that perform hessian-vector products.

        Parameters
        ----------
        model
            A TF model following the InfluenceModel interface whose weights we wish to use for the calculation of
            these (inverse)-hessian-vector products.
        train_dataset
            A batched TF dataset containing the training dataset's point we wish to employ for the estimation of
            the hessian matrix.
        """
        assert_batched_dataset(train_dataset)

        self.model = model
        self.train_set = train_dataset

    @abstractmethod
    def compute_ihvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        raise NotImplementedError()


class ExactIHVP(InverseHessianVectorProduct):
    def __init__(self, model: InfluenceModel, train_dataset: tf.data.Dataset):
        """
        A class that performs the 'exact' computation of the inverse-hessian-vector product.
        As such, it will calculate the hessian of the provided model's loss wrt its parameters,
        compute its Moore-Penrose pseudo-inverse (for numerical stability) and the multiply it
        by the gradients.

        To speed up the algorithm, the hessian matrix is calculated once at instantiation.

        For models with a considerable amount of weights, this implementation may be infeasible
        due to its O(n^2) complexity for the hessian, plus the O(n^3) for its inversion.
        If its memory consumption is too high, you should consider using the CGD approximation.

        Parameters
        ----------
        model
            The TF2.X model implementing the InfluenceModel interface.
        train_dataset
            The TF dataset, already batched and containing only the samples we wish to use for
            the computation of the hessian matrix.
        """
        super(ExactIHVP, self).__init__(model, train_dataset)

        self.inv_hessian = self._compute_inv_hessian(self.train_set)
        self.hessian = None

    def _compute_inv_hessian(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Compute the (pseudo)-inverse of the hessian matrix wrt to the model's parameters using
        backward-mode AD.

        Disclaimer: this implementation trades memory usage for speed, so it can be quite
        memory intensive, especially when dealing with big models.

        Parameters
        ----------
        dataset
            A TF dataset containing the whole or part of the training dataset for the
            computation of the inverse of the mean hessian matrix.

        Returns
        ----------
        inv_hessian
            A tf.Tensor with the resulting inverse hessian matrix
        """
        weights = self.model.weights
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_hess:
            tape_hess.watch(weights)
            grads = self.model.batch_gradient(dataset) if dataset._batch_size == 1 \
                else self.model.batch_jacobian(dataset)

        hess = tf.squeeze(tape_hess.jacobian(grads, weights))
        hessian = tf.reduce_mean(tf.reshape(hess, (-1, int(tf.reduce_prod(weights.shape)), int(tf.reduce_prod(weights.shape)))), axis=0)

        return tf.linalg.pinv(hessian)

    def compute_ihvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points using the exact
        formulation.

        Parameters
        ----------
        group
            A TF dataset containing the group of points of which we wish to compute the
            inverse-hessian-vector product.
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing one rank-1 tensor per input point
        """
        assert_batched_dataset(group)

        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian(group), (-1, self.inv_hessian.shape[0]))
            ihvp = tf.matmul(self.inv_hessian, grads, transpose_b=True)
        else:
            ihvp = tf.concat([tf.matmul(self.inv_hessian, vector, transpose_b=True) for vector in group], axis=0)

        return ihvp

    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points using the exact formulation

        Args:
        Parameters
        ----------
        group
            A TF dataset containing the group of points of which we wish to compute the
            hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        assert_batched_dataset(group)

        if self.hessian is None:
            self.hessian = tf.linalg.pinv(self.inv_hessian)

        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian(group), (-1, self.inv_hessian.shape[0]))
            hvp = tf.matmul(self.hessian, grads, transpose_b=True)
        else:
            hvp = tf.concat([tf.matmul(self.hessian, vector, transpose_b=True) for vector in group], axis=0)

        return hvp
