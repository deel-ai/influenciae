# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Conjugate Gradients solver based on jax.scipy's implementation and wikipedia
https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html
https://en.wikipedia.org/wiki/Conjugate_gradient_method

BiCGSTAB (Biconjugate Gradient Stabilized) solver based also on jax.scipy's implementation and wikipedia
https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.bicgstab.html
https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB
"""
import tensorflow as tf

from ..types import Callable, Optional


def _identity(x): # pylint: disable=C0116
    return x


def conjugate_gradients_solve(
        operator: Callable,
        b: tf.Tensor,
        x0: Optional[tf.Tensor] = None,
        *,
        maxiter: int,
        tol: float = 1e-3,
        atol: float = 1e-5,
        M: Callable = _identity
):
    """
    A simple Conjugate Gradients solver based on jax.scipy

    Parameters
    ----------
    operator: Callable
        The operator that calculates the linear map A(x). It is assumed to be hermitian and positive definite
    b: tf.Tensor
        The right hand side of the linear system, represented by a single vector.
    x0: Optional
        A tensor with the same shape as b and the output that servers as a first guess for the solution
    maxiter: int
        The maximum amount of iterations
    tol: float
        Tolerance for convergence. norm(residual) <= max(tol * norm(b), atol)
    atol: float
        Tolerance for convergence. norm(residual) <= max(tol * norm(b), atol)
    M: Callable
        A preconditioner approximating the inverse of A.

    Returns
    -------
    x_final: tf.Tensor
        A tensor with the solution found by the solver
    """
    if x0 is None:
        x0 = tf.zeros_like(b)

    bs = tf.reduce_sum(tf.matmul(b, b, transpose_a=True))
    atol2 = tf.reduce_max([tf.cast(tf.square(tol), dtype=bs.dtype) * bs, tf.cast(tf.square(atol), dtype=bs.dtype)])

    def cond_fun(_, r, gamma, __, k):
        rs = gamma if M is _identity else tf.reduce_sum(tf.matmul(r, r, transpose_a=True))
        cond1 = tf.greater(rs, atol2)
        cond2 = tf.greater(maxiter, k)
        return tf.logical_and(cond1, cond2)

    def body_fun(x, r, gamma, p, k):
        Ap = operator(p)
        alpha = gamma / (tf.reduce_sum(tf.matmul(p, Ap, transpose_a=True)))
        x_ = x + alpha * p
        r_ = r - alpha * Ap
        z_ = M(r_)
        gamma_ = tf.reduce_sum(tf.matmul(r_, z_, transpose_a=True))
        beta_ = gamma_ / gamma
        p_ = z_ + beta_ * p

        return x_, r_, gamma_, p_, k + 1

    r0 = b - operator(x0)
    p0 = z0 = M(r0)
    gamma0 = tf.reduce_sum(tf.matmul(r0, z0, transpose_a=True))
    initial_value = [x0, r0, gamma0, p0, tf.constant(0, dtype=tf.int32)]

    val = tf.while_loop(
        cond=cond_fun,
        body=body_fun,
        loop_vars=initial_value
    )

    x_final, *_ = val

    return x_final


def biconjugate_gradient_stabilized_solve(
        operator: Callable,
        b: tf.Tensor,
        x0: Optional[tf.Tensor] = None,
        *,
        maxiter: int,
        tol: float = 1e-5,
        atol: float = 1e-6,
        M: Callable = _identity):
    """
    A simple BiCGSTAB solver based on jax.scipy

    Parameters
    ----------
    operator: Callable
        The operator that calculates the linear map A(x). It is assumed to be hermitian and positive definite
    b: tf.Tensor
        The right hand side of the linear system, represented by a single vector.
    x0: Optional
        A tensor with the same shape as b and the output that servers as a first guess for the solution
    maxiter: int
        The maximum amount of iterations
    tol: float
        Tolerance for convergence. norm(residual) <= max(tol * norm(b), atol)
    atol: float
        Tolerance for convergence. norm(residual) <= max(tol * norm(b), atol)
    M: Callable
        A preconditioner approximating the inverse of A.

    Returns
    -------
    x_final: tf.Tensor
        A tensor with the solution found by the solver
    """
    if x0 is None:
        x0 = tf.zeros_like(b)

    bs = tf.reduce_sum(tf.matmul(b, b, transpose_a=True))
    atol2 = tf.reduce_max([tf.square(tol) * bs, tf.square(atol)])

    def cond_fun(value):
        _, r, *_, k = value
        rs = tf.reduce_sum(tf.matmul(r, r, transpose_a=True))
        return (rs > atol2) & (k < maxiter) & (k >= 0)

    def body_fun(value):
        x, r, rhat, alpha, omega, rho, p, q, k = value
        rho_ = tf.reduce_sum(tf.matmul(rhat, r, transpose_a=True))
        beta = rho_ / rho * alpha / omega
        p_ = r + beta * (p - omega * q)
        phat = M(p_)
        q_ = operator(phat)
        alpha_ = rho_ / tf.reduce_sum(tf.matmul(rhat, q_, transpose_a=True))
        s = r - alpha_ * q_
        exit_early = tf.reduce_sum(tf.matmul(s, s, transpose_a=True)) < atol2
        shat = M(s)
        t = operator(shat)
        omega_ = tf.reduce_sum(tf.matmul(t, s, transpose_a=True)) / tf.reduce_sum(tf.matmul(t, t, transpose_a=True))
        x_ = x + alpha_ * phat if exit_early else x + alpha_ * phat + omega_ * shat
        r_ = s if exit_early else s - omega_ * t
        k_ = tf.where((omega_ == 0) | (alpha_ == 0), -11, k + 1)
        k_ = tf.where((rho_ == 0), -10, k_)
        return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_

    r0 = b - operator(x0)
    rho0 = alpha0 = omega0 = 1.
    initial_value = (x0, r0, r0, alpha0, omega0, rho0, r0, r0, 0)

    # Perform the minimization until convergence
    val = initial_value
    while cond_fun(val):
        val = body_fun(val)
    x_final, *_ = val

    return x_final
