# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TODO
"""
import tensorflow as tf
import numpy as np

class BatchSort:
    """
    #TODO: Add documentation
    """
    def __init__(self, batch_shape, k_shape, dtype=tf.float32):
        self.k = k_shape[1]
        shape = tf.concat((k_shape, batch_shape), axis=0)
        self.best_batch = tf.Variable(tf.zeros(shape, dtype=dtype), trainable=False)
        self.best_values = tf.Variable(tf.ones(k_shape, dtype=dtype) * (-np.inf), trainable=False)

    def add_all(self, batch_key: tf.Tensor, batch_values: tf.Tensor) -> None:
        """
        TODO
        """
        current_score = tf.concat([self.best_values, batch_values], axis=1)
        current_batch = tf.concat([self.best_batch, batch_key], axis=1)
        indexes = tf.argsort(current_score,axis=1,direction='DESCENDING')
        indexes = indexes[:, :self.k]

        current_best_score = tf.gather(current_score, indexes,batch_dims=1)
        current_best_samples = tf.gather(current_batch, indexes,batch_dims=1)

        self.best_values.assign(current_best_score)
        self.best_batch.assign(current_best_samples)

    def get(self):
        """
        TODO
        """
        return self.best_batch, self.best_values

    def reset(self):
        """
        TODO
        """
        self.best_batch.assign(tf.zeros_like(self.best_batch))
        self.best_values.assign((-np.inf) * tf.ones_like(self.best_values))
