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
import numpy as np

from ..common.backend import BaseBackend, get_backend_for_tensor, Framework
from ..types import Callable, Optional, Any


def _identity(x):  # pylint: disable=C0116
    return x


def conjugate_gradients_solve(
    operator: Callable[[Any], Any],
    b: Any,
    x0: Optional[Any] = None,
    maxiter: int = 100,
    tol: float = 1e-10,
    eps: float = 1e-12,
    backend: Optional[BaseBackend] = None,
) -> Any:
    """
    Solve Ax = b using Conjugate Gradients where operator(x) returns Ax.

    This function is backend-agnostic and supports both TensorFlow and PyTorch tensors,
    as well as NumPy arrays.

    Parameters
    ----------
    operator
        Function implementing A @ x.
    b
        Right-hand side vector (typically shape (n, 1) or (n,)).
    x0
        Initial guess. If None, uses zeros_like(b).
    maxiter
        Maximum number of CG iterations.
    tol
        Stop when ||r|| <= tol.
    eps
        Small number to avoid division by zero.
    backend
        Optional backend instance. If None, will be auto-detected from tensor type.

    Returns
    -------
    x
        Approximate solution with the same type as b.
    """
    # Handle NumPy arrays separately (no backend needed)
    if isinstance(b, np.ndarray):
        return _conjugate_gradients_numpy(operator, b, x0, maxiter, tol, eps)

    # Auto-detect backend if not provided
    if backend is None:
        backend = get_backend_for_tensor(b)

    if backend.framework == Framework.TENSORFLOW:
        return _conjugate_gradients_tensorflow(operator, b, x0, maxiter, tol, eps, backend)

    # Initialize solution
    x = backend.zeros_like(b) if x0 is None else x0

    # Compute initial residual: r = b - Ax
    r = b - operator(x)
    p = backend.copy(r)

    # Compute initial squared residual norm
    rs_old = backend.reduce_sum(backend.multiply(r, r))

    for _ in range(maxiter):
        # Compute A @ p
        Ap = operator(p)

        # Compute step size: alpha = r^T r / (p^T A p)
        pAp = backend.reduce_sum(backend.multiply(p, Ap))
        denom = backend.maximum(pAp, eps)
        alpha = rs_old / denom

        # Update solution: x = x + alpha * p
        x = x + alpha * p

        # Update residual: r = r - alpha * Ap
        r = r - alpha * Ap

        # Compute new squared residual norm
        rs_new = backend.reduce_sum(backend.multiply(r, r))

        # Check convergence
        residual_norm = backend.sqrt(rs_new)
        # Convert to Python scalar for comparison if needed
        if hasattr(residual_norm, 'item'):
            residual_norm = residual_norm.item()
        elif hasattr(residual_norm, 'numpy'):
            residual_norm = float(residual_norm.numpy())

        if residual_norm <= tol:
            break

        # Update search direction: p = r + (rs_new / rs_old) * p
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def _conjugate_gradients_tensorflow(
    operator: Callable[[Any], Any],
    b: Any,
    x0: Optional[Any],
    maxiter: int,
    tol: float,
    eps: float,
    backend: BaseBackend,
) -> Any:
    """
    TF-safe Conjugate Gradients: uses backend.while_loop so it can run under
    tf.function / tf.data.Dataset.map / tf.map_fn without AutoGraph issues.
    """
    dtype = backend.get_dtype(b)
    tol_t = backend.constant(tol, dtype=dtype)
    eps_t = backend.constant(eps, dtype=dtype)

    # Use a TF scalar for maxiter to avoid mixed Python/Tensor comparisons in graph mode
    maxiter_t = backend.constant(maxiter, dtype=backend.int32_dtype())

    x = backend.zeros_like(b) if x0 is None else x0
    r = b - operator(x)
    p = backend.copy(r)
    rs = backend.reduce_sum(backend.multiply(r, r))

    k0 = backend.constant(0, dtype=backend.int32_dtype())

    def cond_fn(k, x, r, p, rs):
        # Continue while k < maxiter and ||r|| > tol
        return backend.logical_and(k < maxiter_t, backend.sqrt(rs) > tol_t)

    def body_fn(k, x, r, p, rs):
        Ap = operator(p)
        pAp = backend.reduce_sum(backend.multiply(p, Ap))
        denom = backend.maximum(pAp, eps_t)
        alpha = rs / denom

        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = backend.reduce_sum(backend.multiply(r, r))

        # Avoid rs==0 division edge case
        rs_safe = backend.maximum(rs, eps_t)
        beta = rs_new / rs_safe
        p = r + beta * p

        return [k + 1, x, r, p, rs_new]

    k, x, r, p, rs = backend.while_loop(
        cond_fn=cond_fn,
        body_fn=body_fn,
        loop_vars=[k0, x, r, p, rs],
        maximum_iterations=None,
    )

    return x


def _conjugate_gradients_numpy(
    operator: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    maxiter: int = 100,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    NumPy implementation of Conjugate Gradients solver.

    Parameters
    ----------
    operator
        Function implementing A @ x.
    b
        Right-hand side vector.
    x0
        Initial guess. If None, uses zeros_like(b).
    maxiter
        Maximum number of CG iterations.
    tol
        Stop when ||r|| <= tol.
    eps
        Small number to avoid division by zero.

    Returns
    -------
    x
        Approximate solution.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()

    r = b - operator(x)
    p = r.copy()
    rs_old = float(np.sum(r * r))

    for _ in range(maxiter):
        Ap = operator(p)
        denom = max(float(np.sum(p * Ap)), eps)
        alpha = rs_old / denom

        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = float(np.sum(r * r))
        if np.sqrt(rs_new) <= tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def biconjugate_gradient_stabilized_solve(
        operator: Callable,
        b: Any,
        x0: Optional[Any] = None,
        *,
        maxiter: int,
        tol: float = 1e-5,
        atol: float = 1e-6,
        M: Callable = _identity,
        backend: Optional[BaseBackend] = None):
    """
    A BiCGSTAB (Biconjugate Gradient Stabilized) solver.

    This function is backend-agnostic and supports both TensorFlow and PyTorch tensors.

    Parameters
    ----------
    operator
        The operator that calculates the linear map A(x). It is assumed to be hermitian and positive definite.
    b
        The right hand side of the linear system, represented by a single vector.
    x0
        A tensor with the same shape as b and the output that serves as a first guess for the solution.
    maxiter
        The maximum amount of iterations.
    tol
        Tolerance for convergence. norm(residual) <= max(tol * norm(b), atol)
    atol
        Tolerance for convergence. norm(residual) <= max(tol * norm(b), atol)
    M
        A preconditioner approximating the inverse of A.
    backend
        Optional backend instance. If None, will be auto-detected from tensor type.

    Returns
    -------
    x_final
        A tensor with the solution found by the solver.
    """
    # Handle NumPy arrays separately
    if isinstance(b, np.ndarray):
        return _bicgstab_numpy(operator, b, x0, maxiter=maxiter, tol=tol, atol=atol, M=M)

    # Auto-detect backend if not provided
    if backend is None:
        backend = get_backend_for_tensor(b)

    if x0 is None:
        x0 = backend.zeros_like(b)

    # Helper function to compute dot product (x^T @ y)
    def dot(x, y):
        return backend.reduce_sum(backend.multiply(x, y))

    bs = dot(b, b)
    atol2 = backend.maximum(tol * tol * bs, atol * atol)

    def check_convergence(r, k):
        rs = dot(r, r)
        # Convert to Python scalar for comparison
        if hasattr(rs, 'item'):
            rs_val = rs.item()
            atol2_val = atol2.item() if hasattr(atol2, 'item') else float(atol2)
        elif hasattr(rs, 'numpy'):
            rs_val = float(rs.numpy())
            atol2_val = float(atol2.numpy()) if hasattr(atol2, 'numpy') else float(atol2)
        else:
            rs_val = float(rs)
            atol2_val = float(atol2)
        return (rs_val > atol2_val) and (k < maxiter) and (k >= 0)

    r0 = b - operator(x0)
    rho = alpha = omega = 1.0
    x = x0
    r = backend.copy(r0)
    rhat = backend.copy(r0)
    p = backend.copy(r0)
    q = backend.copy(r0)
    k = 0

    while check_convergence(r, k):
        rho_ = dot(rhat, r)

        # Handle potential division by zero
        if hasattr(rho_, 'item'):
            rho_val = rho_.item()
        elif hasattr(rho_, 'numpy'):
            rho_val = float(rho_.numpy())
        else:
            rho_val = float(rho_)

        if rho_val == 0:
            break

        beta = (rho_ / rho) * (alpha / omega)
        p = r + beta * (p - omega * q)
        phat = M(p)
        q = operator(phat)

        rhat_q = dot(rhat, q)
        alpha = rho_ / rhat_q

        s = r - alpha * q

        # Check for early exit
        ss = dot(s, s)
        if hasattr(ss, 'item'):
            ss_val = ss.item()
            atol2_val = atol2.item() if hasattr(atol2, 'item') else float(atol2)
        elif hasattr(ss, 'numpy'):
            ss_val = float(ss.numpy())
            atol2_val = float(atol2.numpy()) if hasattr(atol2, 'numpy') else float(atol2)
        else:
            ss_val = float(ss)
            atol2_val = float(atol2)

        if ss_val < atol2_val:
            x = x + alpha * phat
            break

        shat = M(s)
        t = operator(shat)

        ts = dot(t, s)
        tt = dot(t, t)
        omega = ts / tt

        # Check for breakdown
        if hasattr(omega, 'item'):
            omega_val = omega.item()
        elif hasattr(omega, 'numpy'):
            omega_val = float(omega.numpy())
        else:
            omega_val = float(omega)

        if hasattr(alpha, 'item'):
            alpha_val = alpha.item()
        elif hasattr(alpha, 'numpy'):
            alpha_val = float(alpha.numpy())
        else:
            alpha_val = float(alpha)

        if omega_val == 0 or alpha_val == 0:
            break

        x = x + alpha * phat + omega * shat
        r = s - omega * t
        rho = rho_
        k += 1

    return x


def _bicgstab_numpy(
        operator: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        *,
        maxiter: int,
        tol: float = 1e-5,
        atol: float = 1e-6,
        M: Callable = _identity) -> np.ndarray:
    """
    NumPy implementation of BiCGSTAB solver.

    Parameters
    ----------
    operator
        The operator that calculates the linear map A(x).
    b
        The right hand side of the linear system.
    x0
        Initial guess. If None, uses zeros_like(b).
    maxiter
        Maximum number of iterations.
    tol
        Tolerance for convergence.
    atol
        Absolute tolerance for convergence.
    M
        A preconditioner approximating the inverse of A.

    Returns
    -------
    x
        Approximate solution.
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    def dot(x, y):
        return float(np.sum(x * y))

    bs = dot(b, b)
    atol2 = max(tol * tol * bs, atol * atol)

    r0 = b - operator(x0)
    rho = alpha = omega = 1.0
    x = x0.copy()
    r = r0.copy()
    rhat = r0.copy()
    p = r0.copy()
    q = r0.copy()
    k = 0

    while dot(r, r) > atol2 and k < maxiter and k >= 0:
        rho_ = dot(rhat, r)

        if rho_ == 0:
            break

        beta = (rho_ / rho) * (alpha / omega)
        p = r + beta * (p - omega * q)
        phat = M(p)
        q = operator(phat)

        alpha = rho_ / dot(rhat, q)
        s = r - alpha * q

        if dot(s, s) < atol2:
            x = x + alpha * phat
            break

        shat = M(s)
        t = operator(shat)

        omega = dot(t, s) / dot(t, t)

        if omega == 0 or alpha == 0:
            break

        x = x + alpha * phat + omega * shat
        r = s - omega * t
        rho = rho_
        k += 1

    return x
