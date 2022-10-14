# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf

from deel.influenciae.utils.sorted_dict import BatchSort, ORDER


def test_batched_sorted_dict_1():
    bsd = BatchSort(batch_shape=(2,), k_shape=(1, 4), dtype=tf.float32, order=ORDER.DESCENDING)

    # 1
    v = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 1, 1], dtype=tf.float32), axis=0)
    k = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32), axis=0)
    bsd.add_all(k, v)

    # 2
    v = tf.expand_dims(tf.convert_to_tensor([0, 0, 0, 0, 1], dtype=tf.float32), axis=0) * 2
    k = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32),
                       axis=0) * 10
    bsd.add_all(k, v)

    # 3
    v = tf.expand_dims(tf.convert_to_tensor([1, 1, -1, -1, -1], dtype=tf.float32), axis=0) * 3
    k = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32),
                       axis=0) * 100
    bsd.add_all(k, v)

    # 4
    v = tf.expand_dims(tf.convert_to_tensor([1, -1, -1, -1, -1], dtype=tf.float32), axis=0) * 2.5
    k = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32),
                       axis=0) * 1000
    bsd.add_all(k, v)

    key, k = bsd.get()

    values_expected = tf.convert_to_tensor([[3., 3., 2.5, 2.]])

    key_expected = tf.convert_to_tensor([[[200, 200], [300, 300], [2000, 2000], [60, 60]]], dtype=tf.float32)

    assert tf.reduce_max(tf.abs(key - key_expected)) < 1E-6
    assert tf.reduce_max(tf.abs(k - values_expected)) < 1E-6


def test_batched_sorted_dict_2():
    bsd = BatchSort(batch_shape=(2,), k_shape=(1, 4), dtype=tf.float32, order=ORDER.ASCENDING)

    # 1
    v = tf.expand_dims(tf.convert_to_tensor([-1, -1, -1, -1, -1], dtype=tf.float32), axis=0)
    k = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32), axis=0)
    bsd.add_all(k, v)

    # 2
    v = tf.expand_dims(tf.convert_to_tensor([0, 0, 0, 0, -1], dtype=tf.float32), axis=0) * 2
    k = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32),
                       axis=0) * 10
    bsd.add_all(k, v)

    # 3
    v = tf.expand_dims(tf.convert_to_tensor([-1, -1, 1, 1, 1], dtype=tf.float32), axis=0) * 3
    k = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32),
                       axis=0) * 100
    bsd.add_all(k, v)

    # 4
    v = tf.expand_dims(tf.convert_to_tensor([-1, 1, 1, 1, 1], dtype=tf.float32), axis=0) * 2.5
    k = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32),
                       axis=0) * 1000
    bsd.add_all(k, v)

    key, k = bsd.get()

    values_expected = tf.convert_to_tensor([[-3., -3., -2.5, -2.]])

    key_expected = tf.convert_to_tensor([[[200, 200], [300, 300], [2000, 2000], [60, 60]]], dtype=tf.float32)

    assert tf.reduce_max(tf.abs(key - key_expected)) < 1E-6
    assert tf.reduce_max(tf.abs(k - values_expected)) < 1E-6


def test_batched_sorted_dict_3():
    bsd = BatchSort(batch_shape=(2,), k_shape=(2, 4), dtype=tf.float32, order=ORDER.DESCENDING)

    # 1
    v = tf.convert_to_tensor([[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]], dtype=tf.float32)
    k = tf.convert_to_tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                              [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=tf.float32)
    bsd.add_all(k, v)

    # 2
    v = tf.convert_to_tensor([[-1, 1, -1, -1, -1], [-1, -1, -1, -1, -1]], dtype=tf.float32) * 2
    k = tf.convert_to_tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                              [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=tf.float32) * 10
    bsd.add_all(k, v)

    # 3
    v = tf.convert_to_tensor([[-1, -1, 1, 1, 1], [1, 1, 1, -1, -1]], dtype=tf.float32) * 3
    k = tf.convert_to_tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                              [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=tf.float32) * 100
    bsd.add_all(k, v)

    # 4
    v = tf.convert_to_tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, 1, -1]], dtype=tf.float32) * 2.5
    k = tf.convert_to_tensor([[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                              [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]], dtype=tf.float32) * 1000
    bsd.add_all(k, v)

    key, k = bsd.get()

    values_expected = tf.convert_to_tensor([[3., 3., 3., 2.], [3., 3., 3.0, 2.5]])

    key_expected = tf.convert_to_tensor([[[400, 400], [500, 500], [600, 600], [30, 30]],
                                         [[200, 200], [300, 300], [400, 400], [5000, 5000]]], dtype=tf.float32)

    assert tf.reduce_max(tf.abs(key - key_expected)) < 1E-6
    assert tf.reduce_max(tf.abs(k - values_expected)) < 1E-6
