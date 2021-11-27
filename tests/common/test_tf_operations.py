import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from influenciae.common import find_layer, is_dataset_batched


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
