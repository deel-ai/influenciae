# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import (Reduction, MeanSquaredError)

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP

from ..utils_test import almost_equal, jacobian_ground_truth, hessian_ground_truth


def test_compute_ihvp_single_batch():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    stochastic_hessian = tf.reduce_mean(hessian_list, axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(stochastic_hessian)
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)

    ## ExactIHVP
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))

    # test the compute_ihvp_single_batch
    ihvp_list = []
    for batch in train_set.batch(1):
        batch_ihvp = ihvp_calculator._compute_ihvp_single_batch(batch)
        assert batch_ihvp.shape == (2,1)
        ihvp_list.append(batch_ihvp)
    ihvp_batch = tf.concat(ihvp_list, axis=1)
    assert almost_equal(ihvp_batch, ground_truth_ihvp, epsilon=1e-2)

    ## ConjugateGradientIHVP
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, extractor_layer=-1, train_dataset=train_set.batch(5))

    # test the compute_ihvp_single_batch
    ihvp_list = []
    for batch in train_set.batch(1):
        batch_ihvp = ihvp_calculator._compute_ihvp_single_batch(batch)
        assert batch_ihvp.shape == (2,1)
        ihvp_list.append(batch_ihvp)
    ihvp_batch = tf.concat(ihvp_list, axis=1)
    assert almost_equal(ihvp_batch, ground_truth_ihvp, epsilon=1e-2)

    ## LissaIHVP
    ihvp_calculator = LissaIHVP(influence_model, extractor_layer=-1, train_dataset=train_set.batch(5),
                                scale=4., damping=1e-4, n_opt_iters=200)

    # test the compute_ihvp_single_batch
    ihvp_list = []
    for batch in train_set.batch(1):
        batch_ihvp = ihvp_calculator._compute_ihvp_single_batch(batch)
        assert batch_ihvp.shape == (2, 1)
        ihvp_list.append(batch_ihvp)
    ihvp_batch = tf.concat(ihvp_list, axis=1)
    assert almost_equal(ihvp_batch, ground_truth_ihvp, epsilon=1e-1)


def test_compute_hvp_single_batch():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the HVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_hessian = tf.reduce_mean(hessian_list, axis=0)
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_hvp = tf.matmul(ground_truth_hessian, ground_truth_grads)

    ## ExactIHVP
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))

    # test the compute_hvp_single_batch
    ihvp_calculator.hessian = tf.linalg.pinv(ihvp_calculator.inv_hessian)
    hvp_list = []
    for batch in train_set.batch(1):
        batch_hvp = ihvp_calculator._compute_hvp_single_batch(batch)
        assert batch_hvp.shape == (2,1)
        hvp_list.append(batch_hvp)
    hvp_batch = tf.concat(hvp_list, axis=1)
    assert almost_equal(hvp_batch, ground_truth_hvp, epsilon=1e-2)

    ## ConjugateGradientIHVP
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, extractor_layer=-1, train_dataset=train_set.batch(5))

    # test the compute_hvp_single_batch
    hvp_list = []
    for batch in train_set.batch(1):
        batch_hvp = ihvp_calculator._compute_hvp_single_batch(batch)
        assert batch_hvp.shape == (2,1)
        hvp_list.append(batch_hvp)
    hvp_batch = tf.concat(hvp_list, axis=1)
    assert almost_equal(hvp_batch, ground_truth_hvp, epsilon=1e-2)

    ## LissaIHVP
    ihvp_calculator = LissaIHVP(influence_model, extractor_layer=-1, train_dataset=train_set.batch(5),
                                scale=4., damping=1e-4, n_opt_iters=200)

    # test the compute_hvp_single_batch
    hvp_list = []
    for batch in train_set.batch(1):
        batch_hvp = ihvp_calculator._compute_hvp_single_batch(batch)
        assert batch_hvp.shape == (2, 1)
        hvp_list.append(batch_hvp)
    hvp_batch = tf.concat(hvp_list, axis=1)
    assert almost_equal(hvp_batch, ground_truth_hvp, epsilon=1e-2)


def test_exact_hessian():
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=0, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((5, 1, 3))
    target = tf.random.normal((5, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Test the shape for the first layer
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    inv_hessian = ihvp_calculator.inv_hessian
    assert inv_hessian.shape == (6, 6)  # (3 x 2, 3 x 2)

    # Test the shape for the last layer
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(3))
    inv_hessian = ihvp_calculator.inv_hessian
    assert inv_hessian.shape == (2, 2)  # (2 x 1, 2 x 1)

    # Check the result's precision
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    assert almost_equal(inv_hessian, ground_truth_inv_hessian, epsilon=1e-3)


def test_exact_ihvp():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the IHVP using auto-diff and check shapes
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    ihvp = ihvp_calculator.compute_ihvp(train_set.batch(5))
    ihvp_list = []
    for elt in ihvp:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        ihvp_list.append(elt)
    ihvp = tf.concat(ihvp_list, axis=1)
    assert ihvp.shape == (2, 25) # nb_params times nb_elt stacked on the last axis

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    stochastic_hessian = tf.reduce_mean(hessian_list, axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(stochastic_hessian)
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)
    assert almost_equal(ihvp, ground_truth_ihvp, epsilon=1e-2)

    # test with an initialization with the stochastic_hessian
    ihvp_calculator2 = ExactIHVP(influence_model, train_hessian=stochastic_hessian)
    ihvp2 = ihvp_calculator2.compute_ihvp(train_set.batch(5))
    ihvp_list2 = []
    for elt in ihvp2:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        ihvp_list2.append(elt)
    ihvp2 = tf.concat(ihvp_list2, axis=1)
    assert ihvp2.shape == (2, 25)  # nb_params times nb_elt stacked on the last axis
    assert almost_equal(ihvp2, ground_truth_ihvp, epsilon=1e-2)

    # Do the same for when the vector is directly provided
    vectors = tf.random.normal((25, 2)) # 25 grads of 2 parameters
    ihvp_vectors = ihvp_calculator.compute_ihvp(
        group=tf.data.Dataset.from_tensor_slices(vectors).batch(5),
        use_gradient=False
    )
    ihvp_vectors_list = []
    for elt in ihvp_vectors:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        ihvp_vectors_list.append(elt)
    ihvp_vectors = tf.concat(ihvp_vectors_list, axis=1)
    assert ihvp_vectors.shape == (2, 25) # nb_params times nb_elt stacked on the last axis
    ground_truth_ihvp_vector = tf.matmul(ground_truth_inv_hessian, tf.transpose(vectors))
    assert almost_equal(ihvp_vectors, ground_truth_ihvp_vector, epsilon=1e-3)


def test_exact_hvp():
    # Make sure that the shapes are right and that the exact hvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the HVP using auto-diff and check shapes
    hvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    hvp = hvp_calculator.compute_hvp(train_set.batch(5))
    hvp_list = []
    for elt in hvp:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        hvp_list.append(elt)
    hvp = tf.concat(hvp_list, axis=1)
    assert hvp.shape == (2, 25) # nb_params times nb_elt stacked on the last axis

    # Compute the HVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_hessian = tf.reduce_mean(hessian_list, axis=0)
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_hvp = tf.matmul(ground_truth_hessian, ground_truth_grads)
    assert almost_equal(hvp, ground_truth_hvp, epsilon=1e-3)  # I was forced to increase from 1e-6

    # test with an initialization with the stochastic_hessian
    ihvp_calculator2 = ExactIHVP(influence_model, train_hessian=ground_truth_hessian)
    hvp2 = ihvp_calculator2.compute_hvp(train_set.batch(5))
    hvp_list2 = []
    for elt in hvp2:
        assert elt.shape == (2, 5)  # batch_size times (2, 1) stacked on the last axis
        hvp_list2.append(elt)
    hvp2 = tf.concat(hvp_list2, axis=1)
    assert hvp2.shape == (2, 25)  # 5 times (2, 1) stacked on the last axis
    assert almost_equal(hvp2, ground_truth_hvp, epsilon=1e-3)

    # Do the same for when the vector is directly provided
    vectors = tf.random.normal((25, 2))
    hvp_vectors = hvp_calculator.compute_hvp(
        group=tf.data.Dataset.from_tensor_slices(vectors).batch(5),
        use_gradient=False
    )
    hvp_vectors_list = []
    for elt in hvp_vectors:
        assert elt.shape == (2, 5) # nb_params times batch_size stacked on the last axis
        hvp_vectors_list.append(elt)
    hvp_vectors = tf.concat(hvp_vectors_list, axis=1)
    assert hvp_vectors.shape == (2, 25)
    ground_truth_ihvp_vector = tf.matmul(ground_truth_hessian, tf.transpose(vectors))
    assert almost_equal(hvp_vectors, ground_truth_ihvp_vector, epsilon=1e-3)


def test_cgd_hvp():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the IHVP using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, extractor_layer=1, train_dataset=train_set.batch(5))
    hvp = ihvp_calculator.compute_hvp(train_set.batch(5))
    hvp_list = []
    for elt in hvp:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        hvp_list.append(elt)
    hvp = tf.concat(hvp_list, axis=1)
    assert hvp.shape == (2, 25) # nb_params times nb_elt stacked on the last axis

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_hessian = tf.reduce_mean(hessian_list, axis=0)
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_hvp = tf.matmul(ground_truth_hessian, ground_truth_grads)
    assert almost_equal(hvp, ground_truth_hvp, epsilon=1e-3)

    # Do the same for when the vector is directly provided
    vectors = tf.random.normal((25, 2))
    hvp_vectors = ihvp_calculator.compute_hvp(
        group=tf.data.Dataset.from_tensor_slices(vectors).batch(5),
        use_gradient=False
    )
    hvp_vectors_list = []
    for elt in hvp_vectors:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        hvp_vectors_list.append(elt)
    hvp_vectors = tf.concat(hvp_vectors_list, axis=1)
    assert hvp_vectors.shape == (2, 25) # nb_params times nb_elt stacked on the last axis
    ground_truth_hvp_vector = tf.matmul(ground_truth_hessian, tf.transpose(vectors))
    assert almost_equal(hvp_vectors, ground_truth_hvp_vector, epsilon=1e-3)


def test_cgd_ihvp():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the IHVP using auto-diff and check shapes
    ihvp_calculator = ConjugateGradientDescentIHVP(influence_model, extractor_layer=-1, train_dataset=train_set.batch(5))
    ihvp = ihvp_calculator.compute_ihvp(train_set.batch(5))
    ihvp_list = []
    for elt in ihvp:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        ihvp_list.append(elt)
    ihvp = tf.concat(ihvp_list, axis=1)
    assert ihvp.shape == (2, 25) # nb_params times nb_elt stacked on the last axis

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)
    assert almost_equal(ihvp, ground_truth_ihvp, epsilon=1e-3)

    # Do the same for when the vector is directly provided
    vectors = tf.random.normal((25, 2))
    ihvp_vectors = ihvp_calculator.compute_ihvp(
        group=tf.data.Dataset.from_tensor_slices(vectors).batch(5),
        use_gradient=False
    )
    ihvp_vectors_list = []
    for elt in ihvp_vectors:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        ihvp_vectors_list.append(elt)
    ihvp_vectors = tf.concat(ihvp_vectors_list, axis=1)
    assert ihvp_vectors.shape == (2, 25)  # nb_params times nb_elt stacked on the last axis
    ground_truth_ihvp_vector = tf.matmul(ground_truth_inv_hessian, tf.transpose(vectors))
    assert almost_equal(ihvp_vectors, ground_truth_ihvp_vector, epsilon=1e-3)


def test_lissa_ihvp():
    # Make sure that the shapes are right and that the exact ihvp calculation is correct
    # Make sure that the hessian matrix is being calculated right
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # Compute the IHVP using auto-diff and check shapes
    ihvp_calculator = LissaIHVP(influence_model, extractor_layer=-1, train_dataset=train_set.batch(5),
                                damping=1e-4, scale=4., n_opt_iters=200)
    ihvp = ihvp_calculator.compute_ihvp(train_set.batch(5))
    ihvp_list = []
    for elt in ihvp:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        ihvp_list.append(elt)
    ihvp = tf.concat(ihvp_list, axis=1)
    assert ihvp.shape == (2, 25)  # nb_params times nb_elt stacked on the last axis

    # Compute the IHVP symbolically and check results
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_ihvp = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)

    assert almost_equal(ihvp, ground_truth_ihvp, epsilon=1e-2)

    # Do the same for when the vector is directly provided
    vectors = tf.random.normal((25, 2))
    ihvp_vectors = ihvp_calculator.compute_ihvp(
        group=tf.data.Dataset.from_tensor_slices(vectors).batch(5),
        use_gradient=False
    )
    ihvp_vectors_list = []
    for elt in ihvp_vectors:
        assert elt.shape == (2, 5)  # nb_params times batch_size stacked on the last axis
        ihvp_vectors_list.append(elt)
    ihvp_vectors = tf.concat(ihvp_vectors_list, axis=1)
    assert ihvp_vectors.shape == (2, 25)  # nb_params times nb_elt stacked on the last axis
    ground_truth_ihvp_vector = tf.matmul(ground_truth_inv_hessian, tf.transpose(vectors))
    assert almost_equal(ihvp_vectors, ground_truth_ihvp_vector, epsilon=1e-1)
