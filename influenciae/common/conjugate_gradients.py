"""
Conjugate Gradients solver based on jax.scipy's implementation and wikipedia
https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html
https://en.wikipedia.org/wiki/Conjugate_gradient_method
"""
import tensorflow as tf

from ..types import Callable, Optional


def _identity(x): return x


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
    atol2 = tf.reduce_max([tf.square(tol) * bs, tf.square(atol)])

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma if M is _identity else tf.reduce_sum(tf.matmul(r, r, transpose_a=True))
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = operator(p)
        alpha = gamma / tf.reduce_sum(tf.matmul(p, Ap, transpose_a=True))
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
    initial_value = (x0, r0, gamma0, p0, 0)

    # Perform the minimization until convergence
    val = initial_value
    while cond_fun(val):
        val = body_fun(val)
    x_final, *_ = val

    return x_final
