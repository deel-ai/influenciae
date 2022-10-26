# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Test FirstOrderInfluenceCalculator and influence_abstract interfaces
"""
import os
import shutil

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Reduction, MeanSquaredError, CategoricalCrossentropy

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ConjugateGradientDescentIHVP, CACHE
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils.sorted_dict import ORDER
from ..utils_test import almost_equal, jacobian_ground_truth, hessian_ground_truth

def set_seed():
    tf.random.set_seed(0)

def test_compute_influence_vector():
    """
    Test the compute_influence_vector method
    """
    set_seed()
    # start with a simple model
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake dataset in order to have batched samples
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # compute the influence vector symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_influence = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)

    # Test several configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            influences_vector = []
            for batched_samples in train_set.batch(5):
                batched_influence_vector = influence_calculator._compute_influence_vector(batched_samples)
                assert batched_influence_vector.shape == (5, 2) # 5 times 2 parameters
                assert batched_influence_vector.dtype == tf.float32

                influences_vector.append(batched_influence_vector)
            influences_vector = tf.concat(influences_vector, axis=0)
            # sanity check on final shape
            assert influences_vector.shape == (25, 2)
            # check first order get the right results
            if normalize:
                ground_truth = ground_truth_influence / tf.norm(ground_truth_influence, axis=0, keepdims=True)
            else:
                ground_truth = ground_truth_influence
            assert almost_equal(influences_vector, tf.transpose(ground_truth), epsilon=1e-3)


def test_compute_influence_vector_dataset():
    """
    Test compute influence vector dataset method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake dataset in order to have batched samples
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # compute the influence vector symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, target)], axis=1)
    ground_truth_influence = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            inf_vector_ds = influence_calculator.compute_influence_vector(train_set.batch(5))
            # check first order build the right dataset
            if normalize:
                ground_truth = ground_truth_influence / tf.norm(ground_truth_influence, axis=0, keepdims=True)
            else:
                ground_truth = ground_truth_influence
            gt_dataset = tf.data.Dataset.from_tensor_slices(((inputs, target), tf.transpose(ground_truth))).batch(5)
            for ((gt_x, gt_y), gt_inf_vec), ((batch_x, batch_y), batch_inf_vec) in zip(gt_dataset, inf_vector_ds):
                assert almost_equal(gt_x, batch_x, epsilon=1e-6)
                assert almost_equal(gt_y, batch_y, epsilon=1e-6)
                assert almost_equal(gt_inf_vec, batch_inf_vec, epsilon=1e-3)

    # Test the save & load property
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                    n_samples_for_hessian=25,
                                                    shuffle_buffer_size=25)
    if not os.path.exists("test_temp"):
        os.mkdir("test_temp")    
    inf_vector_ds = influence_calculator.compute_influence_vector(
        train_set.batch(5),
        save_influence_vector_ds_path="test_temp/inf_vector_ds"
    )
    assert os.path.exists("test_temp/inf_vector_ds")
    # assert the saved vector dataset is correct
    inf_vect_ds = tf.data.experimental.load("test_temp/inf_vector_ds").batch(5)
    inf_vect = []
    for elt in inf_vect_ds:
        inf_vect.append(elt)
    inf_vect = tf.concat(inf_vect, axis=0)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)
    assert almost_equal(inf_vect, tf.transpose(gt_inf_vec), epsilon=1E-3)

    shutil.rmtree("test_temp/")


def test_preprocess_sample_to_evaluate():
    """
    Test the preprocess_sample_to_evaluate method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # build a fake dataset in order to have batched samples
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)

            for batched_samples in train_set.batch(5):
                preprocess = influence_calculator._preprocess_samples(batched_samples)
                assert preprocess.shape == (5 ,influence_model.nb_params)


def test_compute_influence_value_from_influence_vector():
    """
    Test the compute_influence_value_from_influence_vector method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake datasets in order to have batched samples
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # compute the influence values symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                            for inp, y in zip(inputs_test, targets_test)], axis=1)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            if normalize:
                ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
            else:
                ground_truth = gt_inf_vec

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            influences_values = []
            for batched_samples in test_set.batch(5):
                batched_grads = influence_calculator._preprocess_samples(batched_samples)
                batched_influence_val = influence_calculator._estimate_influence_value_from_influence_vector(
                    batched_grads, tf.transpose(ground_truth))
                assert batched_influence_val.shape == (5, 25) # 5 samples to evaluate per batch and inv vect has 25 elts
                influences_values.append(batched_influence_val)
            influences_values = tf.concat(influences_values, axis=0)
            # sanity check on final shape
            assert influences_values.shape == (25, 25)
            # check first order get the right results
            gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), ground_truth)
            assert almost_equal(gt_inf_values, influences_values, epsilon=1e-3)


def test_compute_pairwise_influence_value():
    """
    Test the compute_pairwise_influence_value method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake datasets in order to have batched samples
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    # compute the influence values symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            # compute the gt influence values
            if normalize:
                ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
            else:
                ground_truth = gt_inf_vec

            gt_inf_values = tf.reduce_sum(
                tf.multiply(tf.transpose(ground_truth_grads_train), tf.transpose(ground_truth)), # element-wise
                axis=1, keepdims=True
            ) # sum over all parameters

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            influences_values = []
            for batched_samples in train_set.batch(5):
                batched_inf_values = influence_calculator._compute_influence_value_from_batch(batched_samples)
                assert batched_inf_values.shape == (5, 1) # 5 samples to evaluate per batch and inv vect has 25 elts
                influences_values.append(batched_inf_values)
            influences_values = tf.concat(influences_values, axis=0)
            # sanity check on final shape
            assert influences_values.shape == (25, 1)

            # check first order get the right results
            assert almost_equal(gt_inf_values, influences_values, epsilon=1e-3)


def test_compute_top_k_from_training_dataset():
    """
    Test the compute_top_k_from_training_dataset method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake datasets in order to have batched samples
    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    # compute the influence values symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]
    orders = [ORDER.ASCENDING, ORDER.DESCENDING]

    for order in orders:
        for ihvp_calculator in ihvp_objects:
            for normalize in normalization:
                # compute the gt influence values
                if normalize:
                    ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
                else:
                    ground_truth = gt_inf_vec

                gt_inf_values = tf.reduce_sum(
                    tf.multiply(tf.transpose(ground_truth_grads_train), tf.transpose(ground_truth)), # element-wise
                    axis=1, keepdims=True
                ) # sum over all parameters

                if order == ORDER.DESCENDING:
                    gt_top_k_influences = tf.math.top_k(tf.transpose(gt_inf_values), k=5)
                    gt_top_k_influences_values = gt_top_k_influences.values
                else:
                    gt_top_k_influences = tf.math.top_k(-tf.transpose(gt_inf_values), k=5)
                    gt_top_k_influences_values = - gt_top_k_influences.values

                gt_top_k_samples = tf.gather(inputs_train, gt_top_k_influences.indices)
                # gt_top_k_influences = gt_top_k_influences.values

                influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                             n_samples_for_hessian=25,
                                                             shuffle_buffer_size=25,
                                                             normalize=normalize)

                top_k_samples, top_k_influences = influence_calculator.compute_top_k_from_training_dataset(train_set.batch(5), k=5, order=order)
                # check first order get the right results
                assert almost_equal(gt_top_k_influences_values, top_k_influences, epsilon=1e-3)
                assert almost_equal(gt_top_k_samples, top_k_samples, epsilon=1e-3)


def test_compute_influence_values_dataset():
    """
    Test the compute_influence_values_dataset method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake datasets in order to have batched samples
    inputs = tf.random.normal((25, 1, 3))
    targets = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, targets))

    # compute the influence values symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads = tf.concat([jacobian_ground_truth(inp[0], kernel, y) for inp, y in zip(inputs, targets)], axis=1)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            # compute the gt influence values
            if normalize:
                ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
            else:
                ground_truth = gt_inf_vec

            gt_inf_values = tf.reduce_sum(
                tf.multiply(tf.transpose(ground_truth_grads), tf.transpose(ground_truth)), # element-wise
                axis=1, keepdims=True
            ) # sum over all parameters

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            inf_values_ds = influence_calculator.compute_influence_values(train_set.batch(5))
            gt_dataset = tf.data.Dataset.from_tensor_slices(((inputs, targets), gt_inf_values)).batch(5)
            for ((gt_x, gt_y), gt_inf_val), ((batch_x, batch_y), batch_inf_val) in zip(gt_dataset, inf_values_ds):
                assert almost_equal(gt_x, batch_x, epsilon=1e-6)
                assert almost_equal(gt_y, batch_y, epsilon=1e-6)
                assert almost_equal(gt_inf_val, batch_inf_val, epsilon=1e-3)


def test_compute_influence_values():
    """
    Test the compute_influence_values method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # build a fake datasets in order to have batched samples
    inputs = tf.random.normal((25, 1, 3))
    targets = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, targets))
  
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                    n_samples_for_hessian=25,
                                                    shuffle_buffer_size=25)
    inf_values_ds = influence_calculator.compute_influence_values(train_set.batch(5))
    expected_inf_values = []
    for _, batch_inf_val in inf_values_ds: # since asserted previous test
        expected_inf_values.append(batch_inf_val)
    expected_inf_values = tf.concat(expected_inf_values, axis=0)
    computed_inf_values = influence_calculator._compute_influence_values(
        train_set.batch(5)
    )
    assert almost_equal(expected_inf_values, computed_inf_values, epsilon=1E-4)


def test_compute_influence_values_from_tensor():
    """
    Test the compute_influence_values_from_tensor method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake datasets in order to have batched samples
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # compute the influence values symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                            for inp, y in zip(inputs_test, targets_test)], axis=1)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            if normalize:
                ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
            else:
                ground_truth = gt_inf_vec

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            influences_values = []
            for batched_samples in test_set.batch(5):
                local_values = []
                for batched_train in train_set.batch(5):
                    batched_influence_val = influence_calculator._estimate_individual_influence_values_from_batch(batched_train, batched_samples)
                    assert batched_influence_val.shape == (5, 5) # 5 samples to evaluate per batch and local inv vect has 5 elts
                    local_values.append(batched_influence_val)
                local_values = tf.concat(local_values, axis=1)
                influences_values.append(local_values)
            influences_values = tf.concat(influences_values, axis=0)
            # sanity check on final shape
            assert influences_values.shape == (25, 25)
            # check first order get the right results
            gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), ground_truth)
            assert tf.reduce_max(tf.abs(gt_inf_values - influences_values)) < 5E-4


def test_compute_inf_values_with_inf_vect_dataset():
    """
    Test the compute_inf_values_with_inf_vect_dataset method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake datasets in order to have batched samples
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # compute the influence values symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                            for inp, y in zip(inputs_test, targets_test)], axis=1)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            if normalize:
                ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
            else:
                ground_truth = gt_inf_vec
            gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), ground_truth)
            gt_ihvp_dataset = tf.data.Dataset.from_tensor_slices(((inputs_train, targets_train), tf.transpose(ground_truth))).batch(5)

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            influence_values = []
            for samples_to_evaluate in test_set.batch(5):
                _, samples_inf_values_ds = influence_calculator._estimate_inf_values_with_inf_vect_dataset(gt_ihvp_dataset, samples_to_evaluate)
                samples_inf_values = []
                for _, inf_values in samples_inf_values_ds:
                    assert inf_values.shape == (5, 5) # (batch_size_evaluate, batch_size_inf_vect_dataset)
                    samples_inf_values.append(inf_values)
                samples_inf_values = tf.concat(samples_inf_values, axis=1)
                assert samples_inf_values.shape == (5, 25) # (batch_size_evaluate, nb_elt_in_inf_vect_dataset)
                influence_values.append(samples_inf_values)
            influence_values = tf.concat(influence_values, axis=0)
            assert influence_values.shape == (25, 25) # (nb_elt_to_evaluate, nb_elt_in_inf_vect_dataset)
            assert tf.reduce_max(tf.abs(gt_inf_values - influence_values)) < 1E-4


def test_compute_influence_values_for_dataset_to_evaluate():
    """
    Test compute_influence_values_for_dataset_to_evaluate method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake datasets in order to have batched samples
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # compute the influence values symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                            for inp, y in zip(inputs_test, targets_test)], axis=1)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            if normalize:
                ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
            else:
                ground_truth = gt_inf_vec
            gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), ground_truth)

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            eval_inf_ds = influence_calculator.estimate_influence_values_in_batches(test_set.batch(5),
                                                                                    train_set.batch(5))

            influence_values = []
            for _, samples_inf_ds in eval_inf_ds:
                samples_inf_values = []
                for _, inf_values in samples_inf_ds:
                    assert inf_values.shape == (5, 5) # (batch_size_evaluate, batch_size_inf_vect_dataset)
                    samples_inf_values.append(inf_values)
                samples_inf_values = tf.concat(samples_inf_values, axis=1)
                assert samples_inf_values.shape == (5, 25) # (batch_size_evaluate, nb_elt_in_inf_vect_dataset)
                influence_values.append(samples_inf_values)
            influence_values = tf.concat(influence_values, axis=0)
            assert influence_values.shape == (25, 25) # (nb_elt_to_evaluate, nb_elt_in_inf_vect_dataset)
            assert tf.reduce_max(tf.abs(gt_inf_values - influence_values)) < 1E-3
    
    ## Test save and load functionnality
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                    n_samples_for_hessian=25,
                                                    shuffle_buffer_size=25)
    gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), gt_inf_vec)

    if not os.path.exists("test_temp"):
        os.mkdir("test_temp")
    ds = influence_calculator.estimate_influence_values_in_batches(test_set.batch(5), train_set.batch(5),
                                                                   save_influence_vector_path="test_temp/influence_vector_ds",
                                                                   save_influence_value_path="test_temp/influence_values_ds")
    assert os.path.exists("test_temp/influence_vector_ds")
    assert os.path.exists("test_temp/influence_values_ds")
    # assert the saved vector dataset is correct
    inf_vect_ds = tf.data.experimental.load("test_temp/influence_vector_ds").batch(5)
    inf_vect = []
    for elt in inf_vect_ds:
        inf_vect.append(elt)
    inf_vect = tf.concat(inf_vect, axis=0)
    assert almost_equal(inf_vect, tf.transpose(gt_inf_vec), epsilon=1E-3)

    list_dir = os.listdir("test_temp/influence_values_ds")

    influence_values = []
    for path, eval_batch in zip(list_dir, test_set.batch(5)):
        batch_inf_ds = influence_calculator._load_dataset(f"test_temp/influence_values_ds/{path}")
        samples_inf_values = []
        for _, inf_values in batch_inf_ds:
            assert inf_values.shape == (5, 5) # (batch_size_evaluate, batch_size_inf_vect_dataset)
            samples_inf_values.append(inf_values)
        samples_inf_values = tf.concat(samples_inf_values, axis=1)
        assert samples_inf_values.shape == (5, 25) # (batch_size_evaluate, nb_elt_in_inf_vect_dataset)
        influence_values.append(samples_inf_values)
    influence_values = tf.concat(influence_values, axis=0)
    assert influence_values.shape == (25, 25) # (nb_elt_to_evaluate, nb_elt_in_inf_vect_dataset)
    assert tf.reduce_max(tf.abs(gt_inf_values - influence_values)) < 1E-3

    loaded_inf_vect_ds = influence_calculator.estimate_influence_values_in_batches(test_set.batch(5),
                                                                                   train_set.batch(5),
                                                                                   load_influence_vector_path="test_temp/influence_vector_ds")

    influence_values = []
    for _, samples_inf_ds in loaded_inf_vect_ds:
        samples_inf_values = []
        for _, inf_values in samples_inf_ds:
            assert inf_values.shape == (5, 5) # (batch_size_evaluate, batch_size_inf_vect_dataset)
            samples_inf_values.append(inf_values)
        samples_inf_values = tf.concat(samples_inf_values, axis=1)
        assert samples_inf_values.shape == (5, 25) # (batch_size_evaluate, nb_elt_in_inf_vect_dataset)
        influence_values.append(samples_inf_values)
    influence_values = tf.concat(influence_values, axis=0)
    assert influence_values.shape == (25, 25) # (nb_elt_to_evaluate, nb_elt_in_inf_vect_dataset)
    assert tf.reduce_max(tf.abs(gt_inf_values - influence_values)) < 1E-3

    shutil.rmtree("test_temp/")


def test_top_k_dataset():
    """
    Test top_k_dataset method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # get the exact kernel
    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    # build a fake datasets in order to have batched samples
    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # compute the influence values symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    gt_inf_vec = tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train)

    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                            for inp, y in zip(inputs_test, targets_test)], axis=1)

    # Test different configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)), ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5))]
    normalization = [True, False]
    orders = [ORDER.ASCENDING, ORDER.DESCENDING]

    for order in orders:
        for ihvp_calculator in ihvp_objects:
            # ihvp_calculator = object(influence_model, train_set.batch(5))
            for normalize in normalization:
                if normalize:
                    ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
                else:
                    ground_truth = gt_inf_vec
                gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), ground_truth)

                if order == ORDER.DESCENDING:
                    gt_top_k_influences = tf.math.top_k(gt_inf_values, k=3) # (nb_samples_to_evaluate, 3)
                    gt_top_k_influences_values = gt_top_k_influences.values
                else:
                    gt_top_k_influences = tf.math.top_k(-gt_inf_values, k=3)  # (nb_samples_to_evaluate, 3)
                    gt_top_k_influences_values = - gt_top_k_influences.values

                gt_top_k_samples = tf.gather(inputs_train, gt_top_k_influences.indices) # (nb_samples_to_evaluate, single_input_shape)
                # gt_top_k_influences = gt_top_k_influences.values # (nb_samples_to_evaluate, 3)

                influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                             n_samples_for_hessian=25,
                                                             shuffle_buffer_size=25,
                                                             normalize=normalize)

                top_dataset_ds = influence_calculator.top_k(test_set.batch(5), train_set.batch(5), k=3, order=order)

                top_k_influences, top_k_samples = [], []
                for _, influences_values, training_samples in top_dataset_ds:
                    top_k_influences.append(influences_values)
                    top_k_samples.append(training_samples)
                top_k_influences = tf.concat(top_k_influences, axis=0)
                top_k_samples = tf.concat(top_k_samples, axis=0)
                assert top_k_influences.shape == (25, 3)
                assert top_k_samples.shape == (25, 3, 1, 3)

                if isinstance(ihvp_calculator, ExactIHVP):
                    assert tf.reduce_max(tf.abs(gt_top_k_influences_values - top_k_influences)) < 5E-4
                else:
                    assert tf.reduce_max(tf.abs(gt_top_k_influences_values - top_k_influences)) < 1E-3
    
    # Test save & load functionnalities
    ihvp_calculator = ExactIHVP(influence_model, train_set.batch(5))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                    n_samples_for_hessian=25,
                                                    shuffle_buffer_size=25)

    gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), gt_inf_vec)

    gt_top_k_influences = tf.math.top_k(gt_inf_values, k=3) # (nb_samples_to_evaluate, 3)
    gt_top_k_samples = tf.gather(inputs_train, gt_top_k_influences.indices) # (nb_samples_to_evaluate, single_input_shape)
    gt_top_k_influences = gt_top_k_influences.values # (nb_samples_to_evaluate, 3)

    if not os.path.exists("test_temp"):
        os.mkdir("test_temp")
    ds = influence_calculator.top_k(test_set.batch(5), train_set.batch(5), k=3,
                                    save_influence_vector_ds_path="test_temp/influence_vector_ds",
                                    save_top_k_ds_path="test_temp/top_k_ds")
    assert os.path.exists("test_temp/influence_vector_ds")
    assert os.path.exists("test_temp/top_k_ds")

    load_ds = influence_calculator._load_dataset("test_temp/top_k_ds")
    top_k_influences, top_k_samples = [], []
    for _, influences_values, training_samples in load_ds:
        top_k_influences.append(influences_values)
        top_k_samples.append(training_samples)
    top_k_influences = tf.concat(top_k_influences, axis=0)
    top_k_samples = tf.concat(top_k_samples, axis=0)
    assert top_k_influences.shape == (25, 3)
    assert top_k_samples.shape == (25, 3, 1, 3)

    assert tf.reduce_max(tf.abs(gt_top_k_influences - top_k_influences)) < 5E-4
    assert almost_equal(gt_top_k_samples, top_k_samples, epsilon=1E-6)

    other_load_ds = influence_calculator.top_k(test_set.batch(5), train_set.batch(5), k=3,
                                               influence_vector_in_cache=CACHE.MEMORY,
                                               load_influence_vector_ds_path="test_temp/influence_vector_ds")
    top_k_influences, top_k_samples = [], []
    for _, influences_values, training_samples in other_load_ds:
        top_k_influences.append(influences_values)
        top_k_samples.append(training_samples)
    top_k_influences = tf.concat(top_k_influences, axis=0)
    top_k_samples = tf.concat(top_k_samples, axis=0)
    assert top_k_influences.shape == (25, 3)
    assert top_k_samples.shape == (25, 3, 1, 3)

    assert tf.reduce_max(tf.abs(gt_top_k_influences - top_k_influences)) < 5E-4
    assert almost_equal(gt_top_k_samples, top_k_samples, epsilon=1E-6)

    shutil.rmtree("test_temp/")


def test_compute_influence_group():
    """
    Test the compute_influence_group method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    inputs_train = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))

    # Compute the influence vector symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    reduced_ground_truth_grads = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_influence_group = tf.matmul(ground_truth_inv_hessian, reduced_ground_truth_grads)

    # Check results
    ihvp_objects = [
        ExactIHVP(influence_model, train_set.batch(5)),
        ConjugateGradientDescentIHVP(influence_model, -2, train_set.batch(5))
    ]

    for ihvp_calculator in ihvp_objects:
        influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                        n_samples_for_hessian=25,
                                                        shuffle_buffer_size=25)

        influence_group = influence_calculator.compute_influence_vector_group(train_set.batch(25))
        assert influence_group.shape == (1, 2)
        if isinstance(ihvp_calculator, ExactIHVP):
            assert tf.reduce_max(tf.abs(influence_group - tf.transpose(ground_truth_influence_group))) < 5E-4
        else:
            assert tf.reduce_max(tf.abs(influence_group - tf.transpose(ground_truth_influence_group))) < 1E-3


def test_compute_influence_values_group():
    """
    Test the compute_influence_values_group method
    """
    set_seed()
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    kernel = tf.reshape(tf.concat([tf.reshape(layer.weights[0], -1) for layer in influence_model.layers], axis=0), -1)

    inputs_train = tf.random.normal((25, 1, 3))
    inputs_test = tf.random.normal((25, 1, 3))
    targets_train = tf.random.normal((25, 1))
    targets_test = tf.random.normal((25, 1))

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train))
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test))

    # Compute the influence vector symbolically
    hessian_list = tf.concat([
        tf.expand_dims(hessian_ground_truth(tf.squeeze(inp), kernel), axis=0) for inp in inputs_train
    ], axis=0)
    ground_truth_inv_hessian = tf.linalg.pinv(tf.reduce_mean(hessian_list, axis=0))
    ground_truth_grads_train = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                          for inp, y in zip(inputs_train, targets_train)], axis=1)
    ground_truth_grads_train = tf.reduce_sum(ground_truth_grads_train, axis=1, keepdims=True)
    ground_truth_grads_test = tf.concat([jacobian_ground_truth(inp[0], kernel, y)
                                         for inp, y in zip(inputs_test, targets_test)], axis=1)
    ground_truth_grads_test = tf.reduce_sum(ground_truth_grads_test, axis=1, keepdims=True)
    ground_truth_influence_values_group = tf.matmul(ground_truth_grads_test,
                                                    tf.matmul(ground_truth_inv_hessian, ground_truth_grads_train),
                                                    transpose_a=True)

    # Check resultss
    ihvp_objects = [
        ExactIHVP(influence_model, train_set.batch(5)),
        ConjugateGradientDescentIHVP(influence_model, -2, train_set.batch(5))
    ]

    for ihvp_calculator in ihvp_objects:
        influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                        n_samples_for_hessian=25,
                                                        shuffle_buffer_size=25)
        influence = influence_calculator.compute_influence_values_group(train_set.batch(25), test_set.batch(25))
        assert influence.shape == (1, 1)
        assert tf.reduce_max(tf.abs(influence - tf.transpose(ground_truth_influence_values_group))) < 1E-3


def test_cnn_shapes():
    """
    Test all methods with a more challenging model
    """
    set_seed()
    model_feature = Sequential()
    model_feature.add(Input(shape=(5, 5, 3), dtype=tf.float64))
    model_feature.add(Conv2D(4, kernel_size=(2, 2),
                             activation='relu', dtype=tf.float64))
    model_feature.add(Flatten(dtype=tf.float64))
    model_feature.add(Dense(10, kernel_initializer=tf.ones_initializer, use_bias=True, dtype=tf.float64))
    model_feature.add(Dense(10, kernel_initializer=tf.ones_initializer, use_bias=False, dtype=tf.float64))

    model_feature.compile(loss=CategoricalCrossentropy(from_logits=False, reduction=Reduction.NONE), optimizer='sgd')

    influence_model = InfluenceModel(model_feature, loss_function=CategoricalCrossentropy(from_logits=False, reduction=Reduction.NONE))

    inputs_train = tf.random.normal((50, 5, 5, 3), dtype=tf.float64)
    targets_train = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10)), 50)), 10)
    inputs_test = tf.random.normal((60, 5, 5, 3), dtype=tf.float64)
    targets_test = tf.keras.utils.to_categorical(tf.transpose(tf.random.categorical(tf.ones((1, 10), dtype=tf.float64), 60)), 10)

    train_set = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(5)
    test_set = tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).batch(10)

    ihvp_objects = [
        ExactIHVP(influence_model, train_set),
        ConjugateGradientDescentIHVP(influence_model, -2, train_set)
    ]
    nb_params = influence_model.nb_params

    for ihvp_calculator in ihvp_objects:

        influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set, ihvp_calculator,
                                                        n_samples_for_hessian=25,
                                                        shuffle_buffer_size=25)
        iter_test = iter(test_set)
        iter_train = iter(train_set)

        test_batch = next(iter_test)
        train_batch = next(iter_train)

        # compute_influence_values_from_tensor
        inf_val_from_tensor = influence_calculator._estimate_individual_influence_values_from_batch(
            train_samples=train_batch,
            samples_to_evaluate=test_batch
        )
        assert inf_val_from_tensor.shape == (10, 5) # (test_batch_size, train_batch_size)

        # compute_influence_values_for_dataset_to_evaluate
        inf_val_dataset = influence_calculator.estimate_influence_values_in_batches(test_set, train_set)

        iter_inf_val_dataset = iter(inf_val_dataset)
        batch_samples, batched_associated_ds = next(iter_inf_val_dataset)
        assert batch_samples[0].shape==(10, 5, 5, 3)
        assert batch_samples[1].shape==(10, 10)

        iter_batched_associated_ds = iter(batched_associated_ds)
        batch, batch_inf = next(iter_batched_associated_ds)
        assert batch[0].shape == (5, 5, 5, 3) # (train_batch_size, *input_shape)
        assert batch[1].shape == (5, 10) # (train_batch_size, *target_shape)
        assert batch_inf.shape == (10, 5) # (test_batch_size, train_batch_size)

        # compute_influence_vector_dataset
        inf_vect_ds = influence_calculator.compute_influence_vector(
            train_set
        )

        iter_inf_vect = iter(inf_vect_ds)
        (batch_x, batch_y), inf_vect = next(iter_inf_vect)
        assert batch_x.shape == (5, 5, 5, 3) # (train_batch_size, *input_shape)
        assert batch_y.shape == (5, 10) # (train_batch_size, *target_shape)
        assert inf_vect.shape == (5, nb_params) # (train_batch_size, nb_params)

        # compute_influence_values_dataset
        inf_values_dataset = influence_calculator.compute_influence_values(
            train_set
        )

        iter_inf_val_ds = iter(inf_values_dataset)
        (batch_x, batch_y), batch_inf = next(iter_inf_val_ds)
        assert batch_x.shape == (5, 5, 5, 3) # (train_batch_size, *input_shape)
        assert batch_y.shape == (5, 10) # (train_batch_size, *targets_shape)
        assert batch_inf.shape == (5, 1) # (train_batch_size, 1)

        # compute_influence_values
        inf_values = influence_calculator._compute_influence_values(
            train_set
        )
        assert inf_values.shape == (50, 1)

        # compute_top_k_from_training_dataset
        top_k_train_samples, top_k_inf_val = influence_calculator.compute_top_k_from_training_dataset(
            train_set,
            k=3
        )
        assert top_k_train_samples.shape == (3, 5, 5, 3) # (k, *input_shape)
        assert top_k_inf_val.shape == (3,)

        # top_k_dataset
        top_k_dataset = influence_calculator.top_k(test_set, train_set, k=3, d_type=tf.float64)
        iter_top_k = iter(top_k_dataset)
        (batch_evaluate_x, batch_evaluate_y), k_inf_val, k_training_samples = next(iter_top_k)
        assert batch_evaluate_x.shape == (10, 5, 5, 3)
        assert batch_evaluate_y.shape == (10, 10)
        assert k_inf_val.shape == (10, 3,)
        assert k_training_samples.shape == (10, 3, 5, 5, 3)

        # Test the group influence methods
        influence_group = influence_calculator.compute_influence_vector_group(train_set)
        assert influence_group.shape == (1, 650)
        influence_group_values = influence_calculator.compute_influence_values_group(
            train_set,
            tf.data.Dataset.from_tensor_slices((inputs_test, targets_test)).take(50).batch(5)
        )
        assert influence_group_values.shape == (1, 1)
