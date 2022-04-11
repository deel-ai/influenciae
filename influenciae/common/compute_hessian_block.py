"""
Sometimes calculation of the Hessian might be tedious. This scripts
aims to provide some easy way to parallelize computation of this
matrix
"""

import tensorflow as tf

from .model_wrappers import InfluenceModel
from .tf_operations import assert_batched_dataset

def compute_hessian_block(
    model: InfluenceModel,
    partial_trainingset: tf.data.Dataset,
    parallel_iter
):
    """
    If you have a large amount of parameters and/or your datapoints are numerically
    huge (e.g inputs are a significant quantity of large image) it might be challenging
    to compute the stochastic hessian of all this points in one raw. You might want to
    split the operation into different process and retrieve the stochastic estimation
    afterwards. This function is here to help you: it will return the sum of the hessian
    of each input in the partial_trainingset and the number of hessian. It will then be
    easy to retrieve the stochastic hessian of the entire training set.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    partial_trainingset
        A slice of the dataset over which you want to get the hessian. It should be a
        batch dataset with a batch size of 1
    parallel_iter
        The number of parallel iteration to compute the jacobian of the gradients of
        a single input wrt to the weights of your model. Should be a natural divider
        of your gradient vector size

    Returns
    -------
    hess_sum
        The sum of the hessians of each element in the partial training set
    nb_input
        The number of inputs in the partial training set
    """
    assert_batched_dataset(partial_trainingset)
    batch_size = partial_trainingset._batch_size # pylint: disable=W0212
    assert batch_size == 1 , \
        "This function is used for supposedly hard hessian, therefore batch \
        your dataset with a batch size of 1"

    hess_sum = None
    nb_input = 0

    for single_x, single_y in partial_trainingset:
        nb_input += 1
        current_hess = compute_hard_hessian(model, single_x, single_y, parallel_iter)
        hess_sum = current_hess if hess_sum is None else hess_sum + current_hess

    return hess_sum, nb_input

@tf.function
def compute_hard_hessian(model, single_x, single_y, parallel_iter):
    """
    This function compute the hessian matrix of the model's loss wrt its parameters
    on single_x. It is used when the computation of such a matrix is computationnally
    challenging. For example, in semantic segmentation, to compute the hessian matrix
    of a single image (only wrt the last Fully Convolutional Layer) requires to get the
    jacobian of the gradients of the non reduced loss vector wrt to the weights of the
    model. Such a gradient have a size of (nb_class, embedded_width, embedded_height)
    and for only one single input! Therefore, this method use some specific configuration
    to allow this computation.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    single_x
        A single input
    single_y
        The corresponding label
    parallel_iter
        The number to control how many iterations are dispatched in parallel

    Returns
    -------
    hess
        The hessian matrix of the model's loss on single_x wrt its parameters
    """
    weights = model.weights

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_hess:
        tape_hess.watch(weights)
        grads = InfluenceModel._gradient(model, weights, model.loss_function, single_x, single_y) # pylint: disable=W0212

    hess = tf.squeeze(
        tape_hess.jacobian(
            grads, weights, parallel_iterations=parallel_iter,
            experimental_use_pfor=False
            )
        )
    hess = tf.reshape(hess, (-1, int(tf.reduce_prod(weights.shape)), int(tf.reduce_prod(weights.shape))))
    return hess
