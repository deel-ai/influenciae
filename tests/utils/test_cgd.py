# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf

from deel.influenciae.utils.conjugate_gradients import conjugate_gradients_solve
from ..utils_test import almost_equal


def test_conjugate_gradients_solve():
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
