# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from operator import gt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, MeanSquaredError

from deel.influenciae.common import InfluenceModel
from deel.influenciae.influence.inverse_hessian_vector_product import ExactIHVP, ConjugateGradientDescentIHVP
from deel.influenciae.influence.first_order_influence_calculator import FirstOrderInfluenceCalculator

from ..utils import almost_equal, jacobian_ground_truth, hessian_ground_truth

#TODO: Add the different loading and testing options, add a conv2d UC

def test__normalize_if_needed():
    pass

def test_compute_influence_vector():
    # start with a simple model
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
        for normalize in normalization:
            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            influences_vector = []
            for batched_samples in train_set.batch(5):
                batched_influence_vector = influence_calculator.compute_influence_vector(batched_samples)
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
    #TODO: Add more coverage by using load and save attributes of compute_influence_vector_dataset
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
        for normalize in normalization:
            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)
            inf_vector_ds = influence_calculator.compute_influence_vector_dataset(train_set.batch(5))
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


def test_preprocess_sample_to_evaluate():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # build a fake dataset in order to have batched samples
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
        for normalize in normalization:
            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)

            for batched_samples in train_set.batch(5):
                preprocess = influence_calculator.preprocess_sample_to_evaluate(batched_samples)
                assert preprocess.shape == (5 ,influence_model.nb_params)


def test_compute_influence_value_from_influence_vector():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
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
                batched_grads = influence_calculator.preprocess_sample_to_evaluate(batched_samples)
                batched_influence_val = influence_calculator.compute_influence_value_from_influence_vector(
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
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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
    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
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
                batched_inf_values = influence_calculator.compute_pairwise_influence_value(batched_samples)
                assert batched_inf_values.shape == (5, 1) # 5 samples to evaluate per batch and inv vect has 25 elts
                influences_values.append(batched_inf_values)
            influences_values = tf.concat(influences_values, axis=0)
            # sanity check on final shape
            assert influences_values.shape == (25, 1)

            # check first order get the right results
            assert almost_equal(gt_inf_values, influences_values, epsilon=1e-3)


def test_compute_top_k_from_training_dataset():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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
    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
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
            gt_top_k_influences = tf.math.top_k(tf.transpose(gt_inf_values), k=5)
            gt_top_k_samples = tf.gather(inputs_train, gt_top_k_influences.indices)
            gt_top_k_influences = gt_top_k_influences.values

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)

            top_k_samples, top_k_influences = influence_calculator.compute_top_k_from_training_dataset(train_set.batch(5), k=5)
            # check first order get the right results
            assert almost_equal(gt_top_k_influences, top_k_influences, epsilon=1e-3)
            assert almost_equal(gt_top_k_samples, top_k_samples, epsilon=1e-3)


def test_compute_influence_values_dataset():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
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
            inf_values_ds = influence_calculator.compute_influence_values_dataset(train_set.batch(5))
            gt_dataset = tf.data.Dataset.from_tensor_slices(((inputs, targets), gt_inf_values)).batch(5)
            for ((gt_x, gt_y), gt_inf_val), ((batch_x, batch_y), batch_inf_val) in zip(gt_dataset, inf_values_ds):
                assert almost_equal(gt_x, batch_x, epsilon=1e-6)
                assert almost_equal(gt_y, batch_y, epsilon=1e-6)
                assert almost_equal(gt_inf_val, batch_inf_val, epsilon=1e-3)


def test_compute_influence_values_from_tensor():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
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
                    batched_influence_val = influence_calculator.compute_influence_values_from_tensor(batched_train, batched_samples)
                    assert batched_influence_val.shape == (5, 5) # 5 samples to evaluate per batch and local inv vect has 5 elts
                    local_values.append(batched_influence_val)
                local_values = tf.concat(local_values, axis=1)
                influences_values.append(local_values)
            influences_values = tf.concat(influences_values, axis=0)
            # sanity check on final shape
            assert influences_values.shape == (25, 25)
            # check first order get the right results
            gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), ground_truth)
            assert almost_equal(gt_inf_values, influences_values, epsilon=1e-3)


def test_compute_inf_values_with_inf_vect_dataset():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
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
                samples_inf_values_ds = influence_calculator._compute_inf_values_with_inf_vect_dataset(gt_ihvp_dataset, samples_to_evaluate)
                samples_inf_values = []
                for _, inf_values in samples_inf_values_ds:
                    assert inf_values.shape == (5, 5) # (batch_size_evaluate, batch_size_inf_vect_dataset)
                    samples_inf_values.append(inf_values)
                samples_inf_values = tf.concat(samples_inf_values, axis=1)
                assert samples_inf_values.shape == (5, 25) # (batch_size_evaluate, nb_elt_in_inf_vect_dataset)
                influence_values.append(samples_inf_values)
            influence_values = tf.concat(influence_values, axis=0)
            assert influence_values.shape == (25, 25) # (nb_elt_to_evaluate, nb_elt_in_inf_vect_dataset)
            assert almost_equal(gt_inf_values, influence_values, epsilon=1e-3)


def test_compute_influence_values_for_sample_to_evaluate():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
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
            influence_values = []
            for samples_to_evaluate in test_set.batch(5):
                samples_inf_values_ds = influence_calculator.compute_influence_values_for_sample_to_evaluate(train_set.batch(5), samples_to_evaluate)
                samples_inf_values = []
                for _, inf_values in samples_inf_values_ds:
                    assert inf_values.shape == (5, 5) # (batch_size_evaluate, batch_size_inf_vect_dataset)
                    samples_inf_values.append(inf_values)
                samples_inf_values = tf.concat(samples_inf_values, axis=1)
                assert samples_inf_values.shape == (5, 25) # (batch_size_evaluate, nb_elt_in_inf_vect_dataset)
                influence_values.append(samples_inf_values)
            influence_values = tf.concat(influence_values, axis=0)
            assert influence_values.shape == (25, 25) # (nb_elt_to_evaluate, nb_elt_in_inf_vect_dataset)
            assert almost_equal(gt_inf_values, influence_values, epsilon=1e-3)


def test_compute_influence_values_for_dataset_to_evaluate():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
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
            eval_inf_ds = influence_calculator.compute_influence_values_for_dataset_to_evaluate(
                test_set.batch(5),
                train_set.batch(5)
            )

            influence_values = []
            for samples_inf_ds in eval_inf_ds:
                samples_inf_values = []
                for _, inf_values in samples_inf_ds:
                    assert inf_values.shape == (5, 5) # (batch_size_evaluate, batch_size_inf_vect_dataset)
                    samples_inf_values.append(inf_values)
                samples_inf_values = tf.concat(samples_inf_values, axis=1)
                assert samples_inf_values.shape == (5, 25) # (batch_size_evaluate, nb_elt_in_inf_vect_dataset)
                influence_values.append(samples_inf_values)
            influence_values = tf.concat(influence_values, axis=0)
            assert influence_values.shape == (25, 25) # (nb_elt_to_evaluate, nb_elt_in_inf_vect_dataset)
            assert almost_equal(gt_inf_values, influence_values, epsilon=1e-3)


def test_top_k():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
        for normalize in normalization:
            if normalize:
                ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
            else:
                ground_truth = gt_inf_vec
            gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), ground_truth)

            gt_top_k_influences = tf.math.top_k(gt_inf_values, k=5) # (nb_samples_to_evaluate, 5)
            gt_top_k_samples = tf.gather(inputs_train, gt_top_k_influences.indices) # (nb_samples_to_evaluate, single_input_shape)
            gt_top_k_influences = gt_top_k_influences.values # (nb_samples_to_evaluate, 5)

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)

            top_k_influences, top_k_samples = [], []
            for samples_to_evaluate in test_set.batch(5):
                _, samples_top_k_influences, samples_top_k_samples = influence_calculator.top_k(
                    samples_to_evaluate, train_set.batch(5), k=5
                )
                top_k_influences.append(samples_top_k_influences)
                top_k_samples.append(samples_top_k_samples)
            top_k_influences = tf.concat(top_k_influences, axis=0)
            top_k_samples = tf.concat(top_k_samples, axis=0)
            assert top_k_influences.shape == (25, 5)
            assert top_k_samples.shape == (25, 5, 1, 3)
            print(f"res: {tf.reduce_sum(tf.abs(gt_top_k_influences - top_k_influences))}")
            assert almost_equal(gt_top_k_influences, top_k_influences, epsilon=1E-3)
            assert almost_equal(gt_top_k_samples, top_k_samples, epsilon=1E-6)


def test_top_k_dataset():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # buimd the influence model
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

    # Compute the influence vector using auto-diff and check shapes
    ihvp_objects = [ExactIHVP] #TODO: Add CGD
    normalization = [True, False]

    for object in ihvp_objects:
        ihvp_calculator = object(influence_model, train_set.batch(5))
        for normalize in normalization:
            if normalize:
                ground_truth = gt_inf_vec / tf.norm(gt_inf_vec, axis=0, keepdims=True)
            else:
                ground_truth = gt_inf_vec
            gt_inf_values = tf.matmul(tf.transpose(ground_truth_grads_test), ground_truth)

            gt_top_k_influences = tf.math.top_k(gt_inf_values, k=5) # (nb_samples_to_evaluate, 5)
            gt_top_k_samples = tf.gather(inputs_train, gt_top_k_influences.indices) # (nb_samples_to_evaluate, single_input_shape)
            gt_top_k_influences = gt_top_k_influences.values # (nb_samples_to_evaluate, 5)

            influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_set.batch(5), ihvp_calculator,
                                                         n_samples_for_hessian=25,
                                                         shuffle_buffer_size=25,
                                                         normalize=normalize)

            top_dataset_ds = influence_calculator.top_k_dataset(
                test_set.batch(5), train_set.batch(5), k=5
            )
            top_k_influences, top_k_samples = [], []
            for _, influences_values, training_samples in top_dataset_ds:
                top_k_influences.append(influences_values)
                top_k_samples.append(training_samples)
            top_k_influences = tf.concat(top_k_influences, axis=0)
            top_k_samples = tf.concat(top_k_samples, axis=0)
            assert top_k_influences.shape == (25, 5)
            assert top_k_samples.shape == (25, 5, 1, 3)
            print(f"res: {tf.reduce_sum(tf.abs(gt_top_k_influences - top_k_influences))}")
            assert almost_equal(gt_top_k_influences, top_k_influences, epsilon=1E-3)
            assert almost_equal(gt_top_k_samples, top_k_samples, epsilon=1E-6)

def test_save_dataset():
    pass
