# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from deel.influenciae.utils.nearest_neighbors import LinearNearestNeighbors, ORDER
import tensorflow as tf


def test_linear_nearest():
    linear_nearest = LinearNearestNeighbors()

    def dot_product_fun(x1, x2):
        influence_values = tf.matmul(x1, tf.transpose(x2))
        return influence_values

    x = tf.convert_to_tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32) * 10
    y = tf.convert_to_tensor([1, 2, 3, 4, 5, 6], dtype=tf.float32)
    v = tf.convert_to_tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(((x, y), v)).batch(2, drop_remainder=True)

    linear_nearest.build(dataset=dataset,
                         dot_product_fun=dot_product_fun,
                         k=4,
                         query_batch_size=3,
                         d_type=tf.float32,
                         order=ORDER.DESCENDING)

    vector_to_find = tf.convert_to_tensor([[1, 1], [-0.5, 1], [0, -1]], dtype=tf.float32)
    influences_values, training_samples = linear_nearest.query(vector_to_find=vector_to_find)

    influences_values_expected = tf.convert_to_tensor([[12, 10, 8, 6], [3., 2.5, 2., 1.5], [-1, -2, -3, -4]],
                                                      dtype=tf.float32)
    training_samples_expected = tf.convert_to_tensor([[[60, 60], [50, 50], [40, 40], [30, 30]],
                                                      [[60, 60], [50, 50], [40, 40], [30, 30]],
                                                      [[10, 10], [20, 20], [30, 30], [40, 40]]], dtype=tf.float32)

    assert tf.reduce_max(tf.abs(influences_values - influences_values_expected)) < 1E-6
    assert tf.reduce_max(tf.abs(training_samples - training_samples_expected)) < 1E-6
