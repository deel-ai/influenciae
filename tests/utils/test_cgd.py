# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import pytest

from deel.influenciae.utils.conjugate_gradients import conjugate_gradients_solve
from ..utils_test import almost_equal


def test_conjugate_gradients_solve_tensorflow():
    """Test conjugate gradients solve with TensorFlow tensors."""
    import tensorflow as tf

    # Create a random invertible symmetric matrix
    matrix_operator = tf.random.uniform((3, 3))
    diagonal = tf.reduce_sum(tf.abs(matrix_operator), axis=1)
    matrix_operator = tf.matmul(tf.transpose(tf.linalg.set_diag(matrix_operator, diagonal)),
                                tf.linalg.set_diag(matrix_operator, diagonal))
    operator = lambda x: tf.matmul(matrix_operator, x)

    # Define the linear problem Ax=b
    b = tf.transpose(tf.convert_to_tensor([[1, 0, 2]], dtype=tf.float32))
    actual_solution = tf.matmul(tf.linalg.inv(matrix_operator), b)
    cgd_solution = conjugate_gradients_solve(operator, b, None, maxiter=20)
    almost_equal(actual_solution, cgd_solution, epsilon=1e-4)


def test_conjugate_gradients_solve_tensorflow_batched_rhs():
    """Test conjugate gradients solve with batched RHS (TensorFlow)."""
    import tensorflow as tf

    matrix_operator = tf.random.uniform((3, 3))
    diagonal = tf.reduce_sum(tf.abs(matrix_operator), axis=1)
    matrix_operator = tf.matmul(tf.transpose(tf.linalg.set_diag(matrix_operator, diagonal)),
                                tf.linalg.set_diag(matrix_operator, diagonal))
    operator = lambda x: tf.matmul(matrix_operator, x)

    b = tf.convert_to_tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=tf.float32)
    actual_solution = tf.matmul(tf.linalg.inv(matrix_operator), b)
    cgd_solution = conjugate_gradients_solve(operator, b, None, maxiter=20)

    almost_equal(actual_solution, cgd_solution, epsilon=1e-4)


def test_conjugate_gradients_solve_numpy():
    """Test conjugate gradients solve with NumPy arrays."""
    # Create a random invertible symmetric positive definite matrix
    np.random.seed(42)
    A = np.random.rand(3, 3).astype(np.float32)
    A = A + A.T  # Make symmetric
    A = A + 3 * np.eye(3, dtype=np.float32)  # Make diagonally dominant (positive definite)

    b = np.array([[1], [0], [2]], dtype=np.float32)

    operator = lambda x: A @ x
    actual_solution = np.linalg.solve(A, b)
    cgd_solution = conjugate_gradients_solve(operator, b, None, maxiter=20)

    assert np.allclose(actual_solution, cgd_solution, atol=1e-4), \
        f"NumPy solution mismatch: expected {actual_solution.flatten()}, got {cgd_solution.flatten()}"


def test_conjugate_gradients_solve_pytorch():
    """Test conjugate gradients solve with PyTorch tensors."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    # Create a random invertible symmetric positive definite matrix
    torch.manual_seed(42)
    A = torch.rand(3, 3, dtype=torch.float32)
    A = A + A.T  # Make symmetric
    A = A + 3 * torch.eye(3, dtype=torch.float32)  # Make diagonally dominant (positive definite)

    b = torch.tensor([[1.0], [0.0], [2.0]], dtype=torch.float32)

    operator = lambda x: torch.matmul(A, x)
    actual_solution = torch.linalg.solve(A, b)
    cgd_solution = conjugate_gradients_solve(operator, b, None, maxiter=20)

    assert torch.allclose(actual_solution, cgd_solution, atol=1e-4), \
        f"PyTorch solution mismatch: expected {actual_solution.flatten()}, got {cgd_solution.flatten()}"


def test_conjugate_gradients_solve_pytorch_batched_rhs():
    """Test conjugate gradients solve with batched RHS (PyTorch)."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    torch.manual_seed(7)
    A = torch.rand(3, 3, dtype=torch.float32)
    A = A + A.T
    A = A + 3 * torch.eye(3, dtype=torch.float32)

    b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    operator = lambda x: torch.matmul(A, x)
    actual_solution = torch.linalg.solve(A, b)
    cgd_solution = conjugate_gradients_solve(operator, b, None, maxiter=20)

    assert torch.allclose(actual_solution, cgd_solution, atol=1e-4), \
        f"PyTorch solution mismatch: expected {actual_solution.flatten()}, got {cgd_solution.flatten()}"


def test_conjugate_gradients_solve_with_explicit_backend():
    """Test that explicit backend parameter works correctly."""
    import tensorflow as tf
    from deel.influenciae.common import get_backend, Framework

    backend = get_backend(Framework.TENSORFLOW)

    # Create a simple SPD matrix
    matrix = tf.constant([[4.0, 1.0], [1.0, 3.0]], dtype=tf.float32)
    b = tf.constant([[1.0], [2.0]], dtype=tf.float32)

    operator = lambda x: tf.matmul(matrix, x)
    actual_solution = tf.matmul(tf.linalg.inv(matrix), b)
    cgd_solution = conjugate_gradients_solve(operator, b, None, maxiter=20, backend=backend)

    almost_equal(actual_solution, cgd_solution, epsilon=1e-4)
