# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Sometimes, calculation of the Hessian can be tedious. These scripts
aims to provide some easy way to parallelize computation of this
matrix
"""

import tensorflow as tf

from .model_wrappers import InfluenceModel
from .tf_operations import is_dataset_batched

def compute_hessian_block(
    model: InfluenceModel,
    partial_trainingset: tf.data.Dataset,
    parallel_iter: int
):
    """
    If you have a large amount of parameters and/or your datapoints are numerically
    hard to handle (e.g. inputs are a considerable amount of high-resolution images),
    it might be challenging to estimate the mean hessian of all these points directly.
    You might want to split the operation into different processes and retrieve the estimation
    afterwards.
    This function is here to help you: it will return the sum of the hessian of each input in
    the partial_trainingset and the amount of points considered. It will then be easy to
    retrieve the estimation of the mean hessian over the entire training set.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    partial_trainingset
        A slice of the dataset over which you want to get the hessian. It should be a
        batched dataset with a batch size of 1
    parallel_iter
        The number of parallel iterations to compute the jacobian of the gradients of
        a single input wrt to the weights of your model. Should be a natural divisor
        of your gradient vector size

    Returns
    -------
    hess_sum
        The sum of the hessians of each element in the partial training set
    nb_input
        The number of inputs in the partial training set
    """
    assert is_dataset_batched(partial_trainingset) == 1, \
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
    This function computes the hessian matrix of the model's loss wrt its parameters
    on single_x. It is used when the computation of such a matrix is computationally
    challenging. For example, in semantic segmentation, to compute the hessian matrix
    of a single image (only wrt the last Fully Convolutional Layer) requires to get the
    jacobian of the gradients of the non reduced loss vector wrt to the weights of the
    model. Such a gradient has a size of (nb_class, embedded_width, embedded_height)
    and for only one single input! Therefore, this method uses a specific configuration
    that facilitates this computation.

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
    gradients = []

    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_hess:
        tape_hess.watch(weights)
        grads = InfluenceModel._gradient(model, weights, model.loss_function, single_x, single_y) # pylint: disable=W0212
        gradients.append(grads)
        gradients = tf.concat(grads, axis=0)

    hess = tape_hess.jacobian(
            gradients, weights,
            parallel_iterations=parallel_iter,
            experimental_use_pfor=False
            )

    hess = [tf.reshape(h, shape=(model.nb_params, -1)) for h in hess]

    hessian = tf.concat(hess, axis=-1)

    return hessian
