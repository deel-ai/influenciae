# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

from deel.influenciae.common.influence_abstract import VectorBasedInfluenceCalculator

def almost_equal(arr1, arr2, epsilon=1e-6):
    """Ensure two array are almost equal at an epsilon"""
    return np.sum(np.abs(arr1 - arr2)) < epsilon


def assert_tensor_equal(tensor1, tensor2):
    return tf.debugging.assert_equal(tensor1, tensor2)


def generate_data(x_shape=(32, 32, 3), num_labels=10, samples=100):
    x = np.random.rand(samples, *x_shape).astype(np.float32)
    y = to_categorical(np.random.randint(0, num_labels, samples), num_labels)

    return x, y


def generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Dense(output_shape))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    return model


def jacobian_ground_truth(input_vector, kernel_matrix, target):
    """Symbolically calculates the jacobian for the small 2 layer network in the tests"""
    # input_vector = [A0, A1, A2]
    # kernel_matrix = [W03, W04, W13, W14, W23, W24, W35, W45]
    # target = y
    j1 = 2. * tf.square(input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
                        input_vector[2] * kernel_matrix[4]) * kernel_matrix[6] + \
         2. * (input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
               input_vector[2] * kernel_matrix[4]) * (
                     input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] +
                     input_vector[2] * kernel_matrix[5]) * kernel_matrix[7] - \
         2. * (input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
               input_vector[2] * kernel_matrix[4]) * target
    j2 = 2. * tf.square(input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] +
                        input_vector[2] * kernel_matrix[5]) * kernel_matrix[7] + \
         2. * (input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
               input_vector[2] * kernel_matrix[4]) * kernel_matrix[6] * \
         (input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] + input_vector[2] * kernel_matrix[
             5]) - \
         2. * (input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] +
               input_vector[2] * kernel_matrix[5]) * target

    return tf.convert_to_tensor([j1, j2], dtype=tf.float32)


def hessian_ground_truth(input_vector, kernel_matrix):
    """Symbolically calculates the hessian for the small 2 layer network in the tests"""
    # input_vector = [A0, A1, A2]
    # kernel_matrix = [W03, W04, W13, W14, W23, W24, W35, W45]
    h1 = 2. * tf.square(input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
                        input_vector[2] * kernel_matrix[4])
    h23 = 2. * (input_vector[0] * kernel_matrix[0] + input_vector[1] * kernel_matrix[2] +
                input_vector[2] * kernel_matrix[4]) * (input_vector[0] * kernel_matrix[1] +
                                                       input_vector[1] * kernel_matrix[3] +
                                                       input_vector[2] * kernel_matrix[5])
    h4 = 2. * tf.square(input_vector[0] * kernel_matrix[1] + input_vector[1] * kernel_matrix[3] +
                        input_vector[2] * kernel_matrix[5])

    return tf.convert_to_tensor([[h1, h23], [h23, h4]], dtype=tf.float32)

def assert_inheritance(
        method,
        nb_params,
        train_set,
        test_set
    ):
    iter_test = iter(test_set)
    iter_train = iter(train_set)

    test_batch = next(iter_test)
    train_batch = next(iter_train)

    # compute_influence_values_from_tensor
    inf_val_from_tensor = method._compute_individual_influence_values_from_batch(
        train_samples=train_batch,
        samples_to_evaluate=test_batch
    )
    assert inf_val_from_tensor.shape == (10, 5) # (test_batch_size, train_batch_size)

    # compute_influence_values_for_sample_to_evaluate
    batch_samples, iter_inf_val_sample_ds = method._compute_influence_values_in_batches(
        train_set,
        test_batch
    )
    assert batch_samples==test_batch
    iter_inf_val_sample_ds = iter(iter_inf_val_sample_ds)
    train_sample, inf_val_test_batch = next(iter_inf_val_sample_ds)
    assert inf_val_test_batch.shape == (10, 5) # (test_batch_size, len train_set)

    # compute_influence_values_for_dataset_to_evaluate
    inf_val_dataset = method.compute_influence_values_in_batches(
        test_set,
        train_set
    )
    iter_inf_val_dataset = iter(inf_val_dataset)
    batch_samples, batched_associated_ds = next(iter_inf_val_dataset)
    assert batch_samples[0].shape==(10, 5, 5, 3)
    assert batch_samples[1].shape==(10, 1)
    iter_batched_associated_ds = iter(batched_associated_ds)
    (batch_x, batch_y), batch_inf = next(iter_batched_associated_ds)
    assert batch_x.shape == (5, 5, 5, 3) # (train_batch_size, *input_shape)
    assert batch_y.shape == (5, 1) # (train_batch_size, *target_shape)
    assert batch_inf.shape == (10, 5) # (test_batch_size, train_batch_size)

    # compute_influence_vector_dataset
    inf_vect_ds = method.compute_influence_vector(
        train_set
    )
    iter_inf_vect = iter(inf_vect_ds)
    (batch_x, batch_y), inf_vect = next(iter_inf_vect)
    assert batch_x.shape == (5, 5, 5, 3) # (train_batch_size, *input_shape)
    assert batch_y.shape == (5, 1) # (train_batch_size, *target_shape)
    assert inf_vect.shape == (5, nb_params) # (train_batch_size, nb_params)

    # compute_influence_values_dataset
    inf_values_dataset = method.compute_influence_values(
        train_set
    )
    iter_inf_val_ds = iter(inf_values_dataset)
    (batch_x, batch_y), batch_inf = next(iter_inf_val_ds)
    assert batch_x.shape == (5, 5, 5, 3) # (train_batch_size, *input_shape)
    assert batch_y.shape == (5, 1) # (train_batch_size, *targets_shape)
    assert batch_inf.shape == (5, 1) # (train_batch_size, 1)

    # compute_influence_values
    inf_values = method._compute_influence_values(
        train_set
    )
    assert inf_values.shape == (10,1)

    # compute_top_k_from_training_dataset
    top_k_train_samples, top_k_inf_val = method.compute_top_k_from_training_dataset(
        train_set,
        k=3
    )
    assert top_k_train_samples.shape == (3, 5, 5, 3) # (k, *input_shape)
    assert top_k_inf_val.shape == (3,)

    # top_k
    _, top_k_inf_val, top_k_samples = method._top_k_from_batch(
        test_batch,
        train_set,
        k=3,
        d_type=tf.float64
    )
    assert top_k_inf_val.shape == (10, 3,) # (test_batch_size, k, 1)
    assert top_k_samples.shape == (10, 3, 5, 5, 3) # (test_batch_size, k, *input_shape)

    # top_k_dataset
    top_k_dataset = method.top_k(
        test_set,
        train_set,
        k=3,
        d_type=tf.float64
    )
    iter_top_k = iter(top_k_dataset)
    (batch_evaluate_x, batch_evaluate_y), k_inf_val, k_training_samples = next(iter_top_k)
    assert batch_evaluate_x.shape == (10, 5, 5, 3)
    assert batch_evaluate_y.shape == (10, 1)
    assert k_inf_val.shape == (10, 3,)
    assert k_training_samples.shape == (10, 3, 5, 5, 3)
