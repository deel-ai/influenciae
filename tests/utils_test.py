# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np


def _to_numpy(value):
    """Convert TensorFlow/PyTorch tensors or sequences to numpy arrays."""
    if isinstance(value, np.ndarray):
        return value

    try:
        import torch
        if isinstance(value, torch.Tensor):
            value_cpu = value.detach().cpu()
            try:
                return value_cpu.numpy()
            except RuntimeError as exc:
                if "Numpy is not available" not in str(exc):
                    raise
                return np.asarray(value_cpu.tolist())
    except ImportError:
        pass

    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except RuntimeError as exc:
            if "Numpy is not available" not in str(exc):
                raise
            if hasattr(value, "tolist"):
                return np.asarray(value.tolist())
            raise

    if hasattr(value, "tolist"):
        return np.asarray(value.tolist())

    if isinstance(value, (list, tuple)):
        return np.array(value)

    return np.array(value)

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
    arr1 = _to_numpy(arr1)
    arr2 = _to_numpy(arr2)
    return np.sum(np.abs(arr1 - arr2)) < epsilon


def allclose(arr1, arr2, epsilon=1e-4, rtol=None):
    """Check if two tensors/arrays are close using numpy allclose semantics."""
    arr1 = _to_numpy(arr1)
    arr2 = _to_numpy(arr2)
    if rtol is None:
        rtol = epsilon
    return np.allclose(arr1, arr2, atol=epsilon, rtol=rtol)


def max_abs_almost_equal(arr1, arr2, epsilon=1e-6):
    """Check if two tensors/arrays are close using max-absolute error."""
    arr1 = _to_numpy(arr1).astype(np.float64)
    arr2 = _to_numpy(arr2).astype(np.float64)
    return np.max(np.abs(arr1 - arr2)) <= epsilon


def relative_almost_equal(arr1, arr2, percent=0.01):
    """Ensure two array are almost equal at a percent"""
    arr1 = _to_numpy(arr1)
    arr2 = _to_numpy(arr2)
    return np.sum(np.abs(arr1 - arr2)) / (np.sum(np.abs(arr1)) + 1e-10) < percent


def assert_tensor_equal(tensor1, tensor2):
    """Assert two tensors are equal. Works with both TensorFlow and PyTorch tensors."""
    tensor1_np = _to_numpy(tensor1)
    tensor2_np = _to_numpy(tensor2)
    np.testing.assert_array_equal(tensor1_np, tensor2_np)


def assert_close(arr1, arr2, epsilon=1e-6):
    """Assert two tensors/arrays are close with max-absolute tolerance."""
    arr1 = _to_numpy(arr1).astype(np.float64)
    arr2 = _to_numpy(arr2).astype(np.float64)
    max_diff = np.max(np.abs(arr1 - arr2))
    assert max_diff < epsilon, f"Max difference {max_diff} >= {epsilon}"


def assert_allclose(arr1, arr2, rtol=1e-5, atol=1e-6):
    """Assert two tensors/arrays are close with allclose semantics."""
    arr1 = _to_numpy(arr1).astype(np.float64)
    arr2 = _to_numpy(arr2).astype(np.float64)
    if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
        abs_err = np.max(np.abs(arr1 - arr2))
        rel_err = np.max(np.abs(arr1 - arr2) / (np.abs(arr2) + 1e-12))
        raise AssertionError(
            f"Not close: max_abs={abs_err:.3e}, max_rel={rel_err:.3e}, rtol={rtol}, atol={atol}"
        )


def assert_relative_almost_equal(arr1, arr2, percent=0.1):
    """Assert closeness with a max-relative-error criterion."""
    arr1 = _to_numpy(arr1).astype(np.float64)
    arr2 = _to_numpy(arr2).astype(np.float64)
    relative_error = np.abs(arr1 - arr2) / (np.abs(arr2) + 1e-12)
    max_rel = np.max(relative_error)
    if max_rel >= percent:
        raise AssertionError(f"Relative error too large: max_rel={max_rel:.3e} >= {percent:.3e}")


def set_seed_tf(seed=0):
    """Set TensorFlow random seed."""
    _ensure_tensorflow()
    tf.random.set_seed(seed)


def set_seed_torch(seed=0, include_cuda=False):
    """Set PyTorch random seed."""
    import torch

    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_seed_torch_numpy(seed=0, include_cuda=False):
    """Set random seeds for PyTorch and NumPy."""
    import torch

    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def make_linear_model_torch(dtype=None):
    """Build the small 2-layer linear PyTorch model used in analytical tests."""
    import torch
    import torch.nn as nn

    if dtype is None:
        dtype = torch.float64

    return nn.Sequential(
        nn.Linear(3, 2, bias=False, dtype=dtype),
        nn.Linear(2, 1, bias=False, dtype=dtype),
    )


def build_regression_tensors_torch(n_samples, seed, dtype=None):
    """Build synthetic regression tensors matching TF/PyTorch influence tests."""
    import torch

    if dtype is None:
        dtype = torch.float64

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn((n_samples, 1, 3), generator=generator, dtype=dtype)
    targets = torch.randn((n_samples, 1, 1), generator=generator, dtype=dtype)
    return inputs, targets


def build_loader_torch(inputs, targets, batch_size=5, shuffle=False):
    """Create a deterministic DataLoader from input/target tensors."""
    from torch.utils.data import DataLoader, TensorDataset

    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=shuffle)


def ground_truth_grads_hessian_last_layer_torch(model, inputs, targets, return_hessian_stack=False):
    """Compute analytical gradients/Hessians wrt last layer weights for MSE loss."""
    import torch

    w1 = model[0].weight.detach()
    w2 = model[1].weight.detach().squeeze(0)

    grads = []
    hessians = []

    for inp, target in zip(inputs, targets):
        x = inp.squeeze(0)
        y = target.reshape(-1)[0]

        z = w1 @ x
        pred = w2 @ z
        err = pred - y

        grads.append(2.0 * err * z)
        hessians.append(2.0 * torch.outer(z, z))

    grads_mat = torch.stack(grads, dim=0).T
    hessian_stack = torch.stack(hessians, dim=0)
    if return_hessian_stack:
        return grads_mat, hessian_stack

    hessian_mean = hessian_stack.mean(dim=0)
    return grads_mat, hessian_mean


def mse_loss_no_reduction(predictions, targets):
    """MSE loss without reduction, returning one scalar per sample."""
    return ((predictions - targets) ** 2).mean(dim=-1)


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
