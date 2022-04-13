# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical


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
