# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Influence calculator module implementing the approximations in ["Scaling up Influence
Functions"](https://arxiv.org/pdf/2112.03052.pdf). The Arnoldi algorithm effectively
reduces the dimension of the problem of computing IHVPs and allows for the calculation
of influence values on big neural network models.
"""
import tensorflow as tf

from ..common import InfluenceModel, BaseInfluenceCalculator, ForwardOverBackwardHVP
from ..types import Tuple


class ArnoldiInfluenceCalculator(BaseInfluenceCalculator):
    """
    A class implementing an influence score based on reducing the dimension of the problem
    of computing IHVPs through the Arnoldi algorithm as per https://arxiv.org/pdf/2112.03052.pdf
    This allows this calculator to be used on models with a considerable amount of weights
    in a time-efficient manner. The influence score being calculated is theoretically the
    same as the rest of the calculators in the `influence` sub-package.

    Notes
    -----
    This influence calculator applies several approximations to allow it to scale in the amount of model weights
    and in compute time. In [https://arxiv.org/pdf/2112.03052.pdf], the authors compute the influence for
    transformer models with tens of millions of parameters.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    train_dataset
        A batched TF dataset with the points with which the model was trained.
    subspace_dim
        The dimension of the Krylov subspace for the Arnoldi algorithm.
    force_hermitian
        A boolean indicating if we should force the projected matrix to be hermitian before the eigenvalue computation.
    k_largest_eig_vals
        An integer for the amount of top eigenvalues to keep for the influence estimations.
    dtype
        Numeric type for the Krylov basis (tf.float32 by default).
    """
    def __init__(
            self,
            model: InfluenceModel,
            train_dataset: tf.data.Dataset,
            subspace_dim: int,
            force_hermitian: bool,
            k_largest_eig_vals: int,
            dtype: tf.dtypes = tf.float32
    ):
        self.subspace_dim = subspace_dim
        self.force_hermitian = force_hermitian
        self.k_largest_eig_vals = k_largest_eig_vals
        self.model = model
        self.hvp_calculator = ForwardOverBackwardHVP(model, train_dataset)
        self.dtype = dtype

        self.eig_vals, self.G = self.arnoldi(self.model.nb_params)

    def arnoldi(self, dim: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Builds the projection of the inverse of the hessian on the Krylov subspaces.

        Parameters
        ----------
        dim
            The dimension of the basis

        Returns
        -------
        eig_vals
            The eigen values of the projection
        G
            The projection matrix
        """
        v = tf.random.normal((dim,), dtype=self.dtype)
        A, W = self._build_orthogonal_basis(v)
        eig_vals, G = self._distill(A, W)

        return eig_vals, G

    @tf.function
    def __build_orthogonal_basis_iter(
            self,
            W: tf.Tensor,
            A: tf.Tensor,
            index: int
    ) -> Tuple[tf.Tensor, tf.Tensor, int]:
        """
        Builds the new vector of the Krylov's basis and computes the projection of the hessian for this vector.

        Parameters
        ----------
        W
            The (index+1) first vectors of the Krylov's basis
        A
            The "index" first vectors of the projection of the hessian on the Krylov's basis
        index
            The current index of the Krylov's basis

        Returns
        -------
        W
            The (index+2) first vectors of the Krylov's basis
        A
            The (index+1) first vectors of the projection of the hessian on the Krylov's basis
        index
            The current index of the Krylov's basis
        """
        w_next = self.hvp_calculator(W[index])
        w_next = tf.squeeze(w_next, axis=1)
        size = index + 1
        A_next_line = tf.reduce_sum(W[:size] * tf.repeat(tf.expand_dims(w_next, axis=0), size, axis=0), axis=1)
        WA_product = tf.reduce_sum(W[:size] * tf.repeat(tf.expand_dims(A_next_line, axis=1), tf.shape(W)[1], axis=1),
                                   axis=0)
        w_next = w_next - WA_product

        w_next_norm = tf.norm(w_next)

        padding_size = tf.shape(A)[1] - tf.shape(A_next_line)[0] - 1
        A_next_line = tf.concat(
            [A_next_line, tf.expand_dims(w_next_norm, axis=0), tf.zeros((padding_size,), self.dtype)],
            axis=0)

        W = tf.concat([W[:size], tf.expand_dims(w_next, axis=0) / w_next_norm,
                       tf.zeros((tf.shape(W)[0] - size - 1, tf.shape(W)[1]), self.dtype)], axis=0)

        A = tf.concat(
            [A[:index], tf.expand_dims(A_next_line, axis=0),
             tf.zeros((tf.shape(A)[0] - index - 1, tf.shape(A)[1]), self.dtype)],
            axis=0)

        return W, A, index + 1

    def _build_orthogonal_basis(self, v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build orthonormal basis for the Krylov subspaces with the first vector of the basis v.
        Project the hessian on the Krylov subspaces.

        Parameters
        ----------
        v
            The first vector of the Krylov basis

        Returns
        -------
        A
            The projection of the hessian on the Krylov basis
        W
            The Krylov basis
        """
        W0 = tf.concat(
            [tf.expand_dims(v / tf.norm(v), axis=0), tf.zeros((self.subspace_dim, tf.shape(v)[0]), self.dtype)],
            axis=0)
        A0 = tf.zeros((self.subspace_dim, self.subspace_dim + 1), dtype=v.dtype)

        W, A, _ = tf.while_loop(lambda W, A, index: index < self.subspace_dim, self.__build_orthogonal_basis_iter,
                                [W0, A0, tf.constant(0, dtype=tf.int32)],
                                parallel_iterations=1)

        return A, W

    def _distill(self, A: tf.Tensor, W: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Inverse the projection by performing the following operations:

            Hessian = W^T * A * W

            A = V^T * I * V
            G^T = W^T * V^T
            G = V W

            => Hessian = G^T * I * G

        Parameters
        ----------
        A
            The projection of the hessian on the Krylov basis
        W
            The Krylov basis

        Returns
        -------
        eig_vals
            The eigen values of the projection
        G
            The projection matrix
        """
        A = A[:, :-1]
        W = W[:-1, :]

        if self.force_hermitian:
            maindiag = tf.linalg.diag_part(A, k=0)
            superdiag = tf.linalg.diag_part(A, k=1)
            subdiag = tf.linalg.diag_part(A, k=-1)

            superdiag = (superdiag + subdiag) / 2.0

            with tf.device('cpu'):
                eig_vals, eig_vectors = tf.linalg.eigh_tridiagonal(maindiag, superdiag, eigvals_only=False)
        else:
            eig_vals, eig_vectors = tf.linalg.eig(A)

        _, idx = tf.math.top_k(- tf.abs(eig_vals), k=self.k_largest_eig_vals)
        eig_vals = tf.gather(eig_vals, idx, axis=-1)
        eig_vectors = tf.gather(eig_vectors, idx, axis=-1)

        G = tf.matmul(eig_vectors, tf.cast(W, dtype=eig_vectors.dtype), transpose_a=True)

        return eig_vals, G

    def _compute_influence_vector(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Compute an equivalent of the influence vector for a sample of training points.

        Parameters
        ----------
        train_samples
            A tensor with a group of training samples of which we wish to compute the influence.

        Returns
        -------
        influence_vectors
            A tensor with the influence for each sample.
        """
        g_train = self.model.batch_jacobian_tensor(train_samples)

        influence_vectors = tf.matmul(tf.cast(g_train, dtype=self.G.dtype), self.G, transpose_b=True) / self.eig_vals

        return influence_vectors

    def _preprocess_samples(self, samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Pre-process a sample to facilitate evaluation afterwards. In this case, it amounts to transforming
        it into it's "influence vector".

        Parameters
        ----------
        samples
            A tensor with the group of samples we wish to evaluate.

        Returns
        -------
        evaluate_vect
            A tensor with the pre-processed samples.
        """
        g_sample = self.model.batch_jacobian_tensor(samples)

        evaluate_vect = tf.matmul(tf.cast(g_sample, self.G.dtype), self.G, transpose_b=True)

        return evaluate_vect

    def _estimate_influence_value_from_influence_vector(
            self,
            preproc_test_sample: tf.Tensor,
            influence_vector: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the influence score of a (pre-processed) sample and an "influence vector" from a training
        data-point

        Parameters
        ----------
        preproc_test_sample
            A tensor with a (pre-processed) test sample
        influence_vector
            A tensor with an "influence vector" calculated using a training point

        Returns
        -------
        influence_values
            A tensor with the influence scores
        """
        influence_values = tf.matmul(preproc_test_sample, tf.transpose(influence_vector))
        influence_values = tf.math.real(influence_values)

        return influence_values

    def _compute_influence_value_from_batch(self, train_samples: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Compute the influence score for a training sample

        Parameters
        ----------
        train_samples
            Training sample

        Returns
        -------
        The influence score
        """
        g_train = self.model.batch_jacobian_tensor(train_samples)
        influence_vectors = tf.matmul(tf.cast(g_train, dtype=self.G.dtype), self.G, transpose_b=True)

        influence_values = tf.reduce_sum(influence_vectors * influence_vectors / self.eig_vals, axis=1, keepdims=True)
        influence_values = tf.math.real(influence_values)

        return influence_values

    def _estimate_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[tf.Tensor, ...],
            samples_to_evaluate: Tuple[tf.Tensor, ...]
    ) -> tf.Tensor:
        """
        Estimate the (individual) influence scores of a single batch of samples with respect to
        a batch of samples belonging to the model's training dataset.

        Parameters
        ----------
        train_samples
            A single batch of training samples (and their target values).
        samples_to_evaluate
            A single batch of samples of which we wish to compute the influence of removing the training
            samples.

        Returns
        -------
        A tensor containing the individual influence scores.
        """
        return self._estimate_influence_value_from_influence_vector(
            self._preprocess_samples(samples_to_evaluate),
            self._compute_influence_vector(train_samples)
        )
