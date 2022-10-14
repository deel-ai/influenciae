from deel.influenciae.benchmark.base_benchmark import MissingLabelEvaluator, BaseTrainingProcedure
import tensorflow as tf
from typing import Optional, Tuple, Any
import numpy as np
import os
import shutil


class MockTrainingProcedure(BaseTrainingProcedure):

    def train(
            self, training_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset,
            train_batch_size: int = 128, test_batch_size: int = 128,
            log_path: Optional[str] = None) -> Tuple[float, float, tf.keras.Model, Any]:
        """
        TODO
        """
        raise NotImplementedError


def test_noise():
    np.random.seed(0)
    tf.random.set_seed(0)

    size = 10000
    x = tf.linspace(1, size, size)
    class_nbr = 10
    y = tf.concat([tf.ones((size, 1), dtype=tf.float32), tf.zeros((size, class_nbr - 1), dtype=tf.float32)], axis=1)
    training_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # .batch(100)

    misslabeling_ratio = 0.1
    evaluator = MissingLabelEvaluator(training_dataset,
                                      test_dataset=None,
                                      training_procedure=MockTrainingProcedure(),
                                      nb_classes=class_nbr,
                                      misslabeling_ratio=misslabeling_ratio,
                                      train_batch_size=128,
                                      test_batch_size=128,
                                      config=None)

    noisy_dataset, noise_indexes = evaluator.build_noisy_training_dataset()

    noise_ratio_computed = tf.shape(noise_indexes)[1] / size

    assert tf.abs((noise_ratio_computed - misslabeling_ratio) / misslabeling_ratio) < 1E-1

    count = 0
    for _, y in noisy_dataset:
        count = count + tf.reduce_sum(y[..., 1:]).numpy()

    assert count - tf.shape(noise_indexes)[1] == 0

    curve = evaluator._MissingLabelEvaluator__compute_curve(sorted_influences_indexes=[2, 6, 3, 4, 5, 1, 7, 8, 9, 10],
                                                            noisy_label_indexes=[6, 8])
    curve_expected = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    assert np.max(np.abs(curve - curve_expected)) < 1E-6

    curves = [[0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
              [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]]
    mean_curve = np.mean(curves, axis=0)
    roc = np.mean(mean_curve)

    evaluator._MissingLabelEvaluator__save("./tmp_test_bench_base/exp1", curves, mean_curve, roc)

    result = np.load(os.path.join("./tmp_test_bench_base/exp1.npy"), allow_pickle=True)
    shutil.rmtree("./tmp_test_bench_base/")

    assert np.max(np.abs(curve - result[0][0])) < 1E-6
    assert np.max(np.abs(mean_curve - result[1])) < 1E-6
    assert np.max(np.abs(roc - result[2])) < 1E-6
