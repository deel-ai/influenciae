# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from deel.influenciae.utils import find_layer, is_dataset_batched, dataset_to_tensor, array_to_dataset


def test_find_layer():
    """Ensure we can properly target layer using int or string"""
    dummy_model = tf.keras.models.Sequential()
    conv2d1 = Conv2D(4, (1, 1), name="conv2d_1", input_shape=(5, 5, 1))
    conv2d2 = Conv2D(4, (1, 1), name="conv2d_2")
    maxpool = MaxPooling2D()
    flatten = Flatten()
    dense = Dense(5)
    layers = [conv2d1, conv2d2, maxpool, flatten, dense]

    for l in layers:
        dummy_model.add(l)

    for i in range(len(layers)):
        assert find_layer(dummy_model, i).name == layers[i].name

    assert find_layer(dummy_model, "conv2d_1") == layers[0]
    assert find_layer(dummy_model, "conv2d_2") == layers[1]


def test_is_batched_dataset():
    # Ensure we can detect when a tf.dataset is batched
    for shape in [(8, 3, 3, 1), (8, 3, 3), (8, 3)]:
        ds1 = tf.data.Dataset.from_tensor_slices(np.random.random(size=shape))

        assert not is_dataset_batched(ds1)
        assert is_dataset_batched(ds1.batch(1))


def test_dataset_to_tensor():
    x = tf.random.normal((25, 32, 32, 3))
    y = tf.random.normal((25,))
    only_x_ds = tf.data.Dataset.from_tensor_slices(x)
    xy_ds = tf.data.Dataset.from_tensor_slices((x, y))

    # Check that it correctly converts datasets with only x samples
    only_x_tensor = dataset_to_tensor(only_x_ds.batch(6))
    assert tf.reduce_all(x == only_x_tensor)

    # Check that it correctly converts datasets with x and y
    x_tensor, y_tensor = dataset_to_tensor(xy_ds.batch(6))
    assert tf.reduce_all(x == x_tensor)
    assert tf.reduce_all(y == y_tensor)


def test_array_to_dataset():
    x = tf.random.normal((25, 32, 32, 3))
    y = tf.random.normal((25,))
    only_x_ds = array_to_dataset(x, batch_size=6, shuffle=False, buffer_size=25)
    assert tf.reduce_all(x == tf.concat([b for b in only_x_ds], axis=0))

    xy_ds = array_to_dataset((x, y), batch_size=6, shuffle=False, buffer_size=25)
    assert tf.reduce_all(x == tf.concat([b for b in xy_ds.map(lambda z, w: z)], axis=0))
    assert tf.reduce_all(y == tf.concat([b for b in xy_ds.map(lambda z, w: w)], axis=0))

    r = tf.range(0., 25., dtype=tf.float32)
    r_ds = array_to_dataset(r, batch_size=5, shuffle=True, buffer_size=25)
    r_sorted = tf.sort(tf.concat([b for b in r_ds], axis=0))
    assert tf.reduce_all(r == r_sorted)
