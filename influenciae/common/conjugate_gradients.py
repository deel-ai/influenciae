import tensorflow as tf


def _identity(x): return x


def conjugate_gradients_solve(operator, b, x0=None, *, maxiter, tol=1e-3, atol=1e-5, M=_identity):
    if x0 is None:
        x0 = tf.zeros_like(b)

    bs = tf.reduce_sum(tf.matmul(b, b, transpose_a=True))
    atol2 = tf.reduce_max([tf.square(tol) * bs, tf.square(atol)])

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma if M is _identity else tf.reduce_sum(tf.matmul(r, r, transpose_b=True))
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
