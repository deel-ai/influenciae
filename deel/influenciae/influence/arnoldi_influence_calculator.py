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
from typing import Tuple

from ..common import InfluenceModel, BaseInfluenceCalculator, ForwardOverBackwardHVP
from ..common.backend import BaseBackend
from ..types import DType, DatasetLike, Tensor


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
        The model implementing the InfluenceModel interface (TensorFlow or PyTorch).
    train_dataset
        A batched dataset with the points with which the model was trained.
    subspace_dim
        The dimension of the Krylov subspace for the Arnoldi algorithm.
    force_hermitian
        A boolean indicating if we should force the projected matrix to be hermitian before the eigenvalue computation.
    k_largest_eig_vals
        An integer for the amount of top eigenvalues to keep for the influence estimations.
    dtype
        Numeric type for the Krylov basis (float32 by default).
    """
    def __init__(
            self,
            model: InfluenceModel,
            train_dataset: DatasetLike,
            subspace_dim: int,
            force_hermitian: bool,
            k_largest_eig_vals: int,
            dtype: DType = None
    ):
        self.subspace_dim = subspace_dim
        self.force_hermitian = force_hermitian
        self.k_largest_eig_vals = k_largest_eig_vals
        self.model = model
        self.backend: BaseBackend = model.backend
        self.hvp_calculator = ForwardOverBackwardHVP(model, train_dataset)

        # Set default dtype based on backend
        if dtype is None:
            self.dtype = self.backend.float32_dtype()
        else:
            self.dtype = dtype

        self.eig_vals, self.G = self.arnoldi(self.model.nb_params)

    def arnoldi(self, dim: int) -> Tuple[Tensor, Tensor]:
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
        v = self.backend.random_normal((dim,), dtype=self.dtype)
        A, W = self._build_orthogonal_basis(v)
        eig_vals, G = self._distill(A, W)

        return eig_vals, G

    def _build_orthogonal_basis_iter(
            self,
            W: Tensor,
            A: Tensor,
            index: int
    ) -> Tuple[Tensor, Tensor, int]:
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
        w_next = self.backend.squeeze(w_next, axis=1)
        size = index + 1

        # Compute A_next_line = dot product of W[:size] with w_next
        W_slice = W[:size]
        w_next_expanded = self.backend.expand_dims(w_next, axis=0)
        w_next_repeated = self.backend.repeat(w_next_expanded, size, axis=0)
        A_next_line = self.backend.reduce_sum(W_slice * w_next_repeated, axis=1)

        # Compute WA_product
        A_next_line_expanded = self.backend.expand_dims(A_next_line, axis=1)
        W_shape = self.backend.tensor_shape(W)
        A_next_line_repeated = self.backend.repeat(A_next_line_expanded, W_shape[1], axis=1)
        WA_product = self.backend.reduce_sum(W_slice * A_next_line_repeated, axis=0)
        w_next = w_next - WA_product

        w_next_norm = self.backend.norm(w_next)

        # Build padded A_next_line
        A_shape = self.backend.tensor_shape(A)
        padding_size = A_shape[1] - size - 1
        w_next_norm_expanded = self.backend.expand_dims(w_next_norm, axis=0)
        zeros_padding = self.backend.zeros((padding_size,), dtype=self.dtype)
        A_next_line = self.backend.concat([A_next_line, w_next_norm_expanded, zeros_padding], axis=0)

        # Update W
        w_next_normalized = self.backend.expand_dims(w_next / w_next_norm, axis=0)
        zeros_W = self.backend.zeros((W_shape[0] - size - 1, W_shape[1]), dtype=self.dtype)
        W = self.backend.concat([W[:size], w_next_normalized, zeros_W], axis=0)

        # Update A
        A_next_line_expanded = self.backend.expand_dims(A_next_line, axis=0)
        zeros_A = self.backend.zeros((A_shape[0] - index - 1, A_shape[1]), dtype=self.dtype)
        A = self.backend.concat([A[:index], A_next_line_expanded, zeros_A], axis=0)

        return W, A, index + 1

    def _build_orthogonal_basis(self, v: Tensor) -> Tuple[Tensor, Tensor]:
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
        v_shape = self.backend.tensor_shape(v)
        v_normalized = self.backend.expand_dims(v / self.backend.norm(v), axis=0)
        zeros_W = self.backend.zeros((self.subspace_dim, v_shape[0]), dtype=self.dtype)
        W0 = self.backend.concat([v_normalized, zeros_W], axis=0)
        A0 = self.backend.zeros((self.subspace_dim, self.subspace_dim + 1), dtype=self.dtype)

        # Use backend's while_loop for efficiency (especially for TensorFlow graph compilation)
        def cond_fn(_W, _A, index):
            return index < self.subspace_dim

        def body_fn(W, A, index):
            return self._build_orthogonal_basis_iter(W, A, index)

        W, A, _ = self.backend.while_loop(
            cond_fn,
            body_fn,
            [W0, A0, 0],
            maximum_iterations=self.subspace_dim
        )

        return A, W

    def _distill(self, A: Tensor, W: Tensor) -> Tuple[Tensor, Tensor]:
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
            maindiag = self.backend.diag_part(A, k=0)
            superdiag = self.backend.diag_part(A, k=1)
            subdiag = self.backend.diag_part(A, k=-1)

            superdiag = (superdiag + subdiag) / 2.0

            eig_vals, eig_vectors = self.backend.eigh_tridiagonal(maindiag, superdiag, eigvals_only=False)
        else:
            eig_vals, eig_vectors = self.backend.eig(A)

        # Get top k eigenvalues by smallest absolute value
        neg_abs_eig_vals = -self.backend.abs(eig_vals)
        _, idx = self.backend.top_k(neg_abs_eig_vals, k=self.k_largest_eig_vals)
        eig_vals = self.backend.gather_along_axis(eig_vals, idx, axis=-1)
        eig_vectors = self.backend.gather_along_axis(eig_vectors, idx, axis=-1)

        G = self.backend.matmul(
            self.backend.transpose(eig_vectors),
            self.backend.cast(W, dtype=self.backend.get_dtype(eig_vectors))
        )

        return eig_vals, G

    def _compute_influence_vector(self, train_samples: Tuple[Tensor, ...]) -> Tensor:
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

        influence_vectors = self.backend.matmul(
            self.backend.cast(g_train, dtype=self.backend.get_dtype(self.G)),
            self.backend.transpose(self.G)
        ) / self.eig_vals

        return influence_vectors

    def _preprocess_samples(self, samples: Tuple[Tensor, ...]) -> Tensor:
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

        evaluate_vect = self.backend.matmul(
            self.backend.cast(g_sample, self.backend.get_dtype(self.G)),
            self.backend.transpose(self.G)
        )

        return evaluate_vect

    def _estimate_influence_value_from_influence_vector(
            self,
            preproc_test_sample: Tensor,
            influence_vector: Tensor
    ) -> Tensor:
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
        influence_values = self.backend.matmul(preproc_test_sample, self.backend.transpose(influence_vector))
        influence_values = self.backend.real(influence_values)

        return influence_values

    def _compute_influence_value_from_batch(self, train_samples: Tuple[Tensor, ...]) -> Tensor:
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
        influence_vectors = self.backend.matmul(
            self.backend.cast(g_train, dtype=self.backend.get_dtype(self.G)),
            self.backend.transpose(self.G)
        )

        influence_values = self.backend.reduce_sum(
            influence_vectors * influence_vectors / self.eig_vals,
            axis=1,
            keepdims=True
        )
        influence_values = self.backend.real(influence_values)

        return influence_values

    def _estimate_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[Tensor, ...],
            samples_to_evaluate: Tuple[Tensor, ...]
    ) -> Tensor:
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
