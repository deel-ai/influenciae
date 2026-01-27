# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np

# Lazy imports for framework-agnostic testing
# TensorFlow imports are deferred to avoid import errors when TF is not installed
tf = None
Sequential = None
Input = None
Conv2D = None
Dense = None
Flatten = None
to_categorical = None


def _ensure_tensorflow():
    """Lazily import TensorFlow and Keras components."""
    global tf, Sequential, Input, Conv2D, Dense, Flatten, to_categorical
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        from tensorflow.keras.models import Sequential as _Sequential
        from tensorflow.keras.layers import Input as _Input, Conv2D as _Conv2D, Dense as _Dense, Flatten as _Flatten
        from tensorflow.keras.utils import to_categorical as _to_categorical
        Sequential = _Sequential
        Input = _Input
        Conv2D = _Conv2D
        Dense = _Dense
        Flatten = _Flatten
        to_categorical = _to_categorical


from deel.influenciae.common.base_influence import BaseInfluenceCalculator

def almost_equal(arr1, arr2, epsilon=1e-6):
    """Ensure two array are almost equal at an epsilon"""
    # Handle PyTorch tensors
    try:
        import torch
        if isinstance(arr1, torch.Tensor):
            arr1 = arr1.detach().cpu().numpy()
        if isinstance(arr2, torch.Tensor):
            arr2 = arr2.detach().cpu().numpy()
    except ImportError:
        pass
    # Handle TensorFlow tensors
    if hasattr(arr1, 'numpy'):
        arr1 = arr1.numpy()
    if hasattr(arr2, 'numpy'):
        arr2 = arr2.numpy()
    # Handle lists/tuples
    if isinstance(arr1, (list, tuple)):
        arr1 = np.array(arr1)
    if isinstance(arr2, (list, tuple)):
        arr2 = np.array(arr2)
    return np.sum(np.abs(arr1 - arr2)) < epsilon


def relative_almost_equal(arr1, arr2, percent=0.01):
    """Ensure two array are almost equal at a percent"""
    return np.sum(np.abs(arr1 - arr2)) / np.sum(np.abs(arr1)) < percent


def assert_tensor_equal(tensor1, tensor2):
    """Assert two tensors are equal. Works with both TensorFlow and PyTorch tensors."""
    # Handle PyTorch tensors
    try:
        import torch
        if isinstance(tensor1, torch.Tensor) or isinstance(tensor2, torch.Tensor):
            if isinstance(tensor1, torch.Tensor):
                tensor1 = tensor1.detach().cpu().numpy()
            if isinstance(tensor2, torch.Tensor):
                tensor2 = tensor2.detach().cpu().numpy()
            np.testing.assert_array_equal(tensor1, tensor2)
            return
    except ImportError:
        pass
    # TensorFlow tensors
    _ensure_tensorflow()
    return tf.debugging.assert_equal(tensor1, tensor2)


def generate_data(x_shape=(32, 32, 3), num_labels=10, samples=100):
    _ensure_tensorflow()
    x = np.random.rand(samples, *x_shape).astype(np.float32)
    y = to_categorical(np.random.randint(0, num_labels, samples), num_labels)

    return x, y


def generate_model(input_shape=(32, 32, 3), output_shape=10):
    _ensure_tensorflow()
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
    _ensure_tensorflow()
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
    _ensure_tensorflow()
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
    inf_val_from_tensor = method._estimate_individual_influence_values_from_batch(
        train_samples=train_batch,
        samples_to_evaluate=test_batch
    )
    assert inf_val_from_tensor.shape == (10, 5) # (test_batch_size, train_batch_size)

    # compute_influence_values_for_dataset_to_evaluate
    inf_val_dataset = method.estimate_influence_values_in_batches(test_set, train_set)
    iter_inf_val_dataset = iter(inf_val_dataset)
    batch_samples, batched_associated_ds = next(iter_inf_val_dataset)
    assert batch_samples[0].shape == (10, 5, 5, 3)
    assert batch_samples[1].shape == (10, 1)
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
    assert batch_x.shape == (5, 5, 5, 3)  # (train_batch_size, *input_shape)
    assert batch_y.shape == (5, 1)  # (train_batch_size, *target_shape)
    if isinstance(inf_vect, tuple):  # when testing rps_l2
        inf_vect, z_batch = inf_vect
        assert inf_vect.shape == (5, 1)
        assert z_batch.shape == (5, 64)
    else:
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
    assert inf_values.shape == (10, 1)

    # compute_top_k_from_training_dataset
    top_k_train_samples, top_k_inf_val = method.compute_top_k_from_training_dataset(
        train_set,
        k=3
    )
    assert top_k_train_samples.shape == (3, 5, 5, 3) # (k, *input_shape)
    assert top_k_inf_val.shape == (3,)

    # top_k_dataset
    # Use appropriate dtype based on backend
    if hasattr(method, 'backend'):
        from deel.influenciae.common import Framework
        if method.backend.framework == Framework.PYTORCH:
            import torch
            d_type = torch.float64
        else:
            _ensure_tensorflow()
            d_type = tf.float64
    else:
        _ensure_tensorflow()
        d_type = tf.float64
    top_k_dataset = method.top_k(test_set, train_set, k=3, d_type=d_type)
    iter_top_k = iter(top_k_dataset)
    (batch_evaluate_x, batch_evaluate_y), k_inf_val, k_training_samples = next(iter_top_k)
    assert batch_evaluate_x.shape == (10, 5, 5, 3)
    assert batch_evaluate_y.shape == (10, 1)
    assert k_inf_val.shape == (10, 3,)
    assert k_training_samples.shape == (10, 3, 5, 5, 3)
