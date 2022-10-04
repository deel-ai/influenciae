# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf

from deel.influenciae.utils import BatchSort

def test_maximum_sorted_dict():
    # TODO: Refacto
    # v = MaximumSortedDict(3)
    # v.add(5, 'a')
    # v.add(4, 'b')
    # v.add(3, 'd')
    # v.add(6, 'd')
    # v.add(1, 'd')
    # assert list(v.get_key_values().keys()) == [6, 5, 4]
    pass


def test_batched_sorted_dict():
    # TODO: Refacto
    # bsd = BatchedSortedDict(batch_size=5, size_maximum=3)

    # # 1
    # keys = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 1, 1], dtype=tf.float32), axis=1)
    # values = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]), axis=1)
    # bsd.add_all(keys, values)

    # # 2
    # keys = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 1, 1], dtype=tf.float32), axis=1) * 2
    # values = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]), axis=1) * 10
    # bsd.add_all(keys, values)

    # # 3
    # keys = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 1, 1], dtype=tf.float32), axis=1) * 3
    # values = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]), axis=1) * 100
    # bsd.add_all(keys, values)

    # # 4
    # keys = tf.expand_dims(tf.convert_to_tensor([1, 1, -1, 1, 1], dtype=tf.float32), axis=1) * 2.5
    # values = tf.expand_dims(tf.convert_to_tensor([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]), axis=1) * 1000
    # bsd.add_all(keys, values)

    # key, values = bsd.get()

    # key_expected = tf.convert_to_tensor([[3., 2.5, 2.],
    #                                      [3., 2.5, 2.],
    #                                      [3., 2., 1.],
    #                                      [3., 2.5, 2.],
    #                                      [3., 2.5, 2.]])

    # values_expected = tf.convert_to_tensor([[[200, 200],
    #                                          [2000, 2000],
    #                                          [20, 20]],

    #                                         [[300, 300],
    #                                          [3000, 3000],
    #                                          [30, 30]],

    #                                         [[400, 400],
    #                                          [40, 40],
    #                                          [4, 4]],

    #                                         [[500, 500],
    #                                          [5000, 5000],
    #                                          [50, 50]],

    #                                         [[600, 600],
    #                                          [6000, 6000],
    #                                          [60, 60]]])

    # assert tf.reduce_all(key == key_expected)
    # assert tf.reduce_all(values == values_expected)
    pass
