# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Test CIFAR-10 Benchmark module
"""
import numpy as np
from tensorflow.keras.losses import Reduction, CategoricalCrossentropy

from deel.influenciae.benchmark.influence_factory import TracInFactory, RPSLJEFactory, FirstOrderFactory, RPSL2Factory
from deel.influenciae.benchmark.cifar10_benchmark import Cifar10MislabelingDetectorEvaluator


def test_first_order_exact():
    take_batch = 10
    cifar10_evaluator = Cifar10MislabelingDetectorEvaluator(epochs=5,
                                                            model_type='efficient_net',
                                                            mislabeling_ratio=0.0005,
                                                            use_regu=True,
                                                            force_overfit=False,
                                                            train_batch_size=3,
                                                            test_batch_size=3,
                                                            epochs_to_save=None,
                                                            take_batch=take_batch,
                                                            verbose_training=False)

    influence_factory = FirstOrderFactory('exact')
    result = cifar10_evaluator.evaluate(influence_factory=influence_factory, nbr_of_evaluation=2, verbose=False)
    assert np.shape(result[0]) == (2, take_batch)
    assert np.shape(result[1]) == (take_batch,)


def test_first_order_cgd():
    take_batch = 11
    cifar10_evaluator = Cifar10MislabelingDetectorEvaluator(epochs=5,
                                                            model_type='resnet',
                                                            mislabeling_ratio=0.0005,
                                                            use_regu=True,
                                                            force_overfit=False,
                                                            train_batch_size=10,
                                                            test_batch_size=10,
                                                            epochs_to_save=None,
                                                            take_batch=take_batch,
                                                            verbose_training=False)

    influence_factory = FirstOrderFactory('cgd', n_cgd_iters=3)
    result = cifar10_evaluator.evaluate(influence_factory=influence_factory, nbr_of_evaluation=2, verbose=False)
    assert np.shape(result[0]) == (2, take_batch)
    assert np.shape(result[1]) == (take_batch,)


def test_tracein():
    take_batch = 11
    cifar10_evaluator = Cifar10MislabelingDetectorEvaluator(epochs=5,
                                                            model_type='resnet',
                                                            mislabeling_ratio=0.0005,
                                                            use_regu=True,
                                                            force_overfit=False,
                                                            train_batch_size=10,
                                                            test_batch_size=10,
                                                            epochs_to_save=[1, 3],
                                                            take_batch=take_batch,
                                                            verbose_training=False)

    influence_factory = TracInFactory()
    result = cifar10_evaluator.evaluate(influence_factory=influence_factory, nbr_of_evaluation=2, verbose=False)
    assert np.shape(result[0]) == (2, take_batch)
    assert np.shape(result[1]) == (take_batch,)


def test_rps_lje():
    take_batch = 11
    cifar10_evaluator = Cifar10MislabelingDetectorEvaluator(epochs=5,
                                                            model_type='resnet',
                                                            mislabeling_ratio=0.0005,
                                                            use_regu=True,
                                                            force_overfit=False,
                                                            train_batch_size=10,
                                                            test_batch_size=10,
                                                            epochs_to_save=None,
                                                            take_batch=take_batch,
                                                            verbose_training=False)

    influence_factory = RPSLJEFactory('exact')
    result = cifar10_evaluator.evaluate(influence_factory=influence_factory, nbr_of_evaluation=2, verbose=False)
    assert np.shape(result[0]) == (2, take_batch)
    assert np.shape(result[1]) == (take_batch,)


def test_rps_l2():
    take_batch = 11
    cifar10_evaluator = Cifar10MislabelingDetectorEvaluator(epochs=5,
                                                            model_type='resnet',
                                                            mislabeling_ratio=0.0005,
                                                            use_regu=True,
                                                            force_overfit=False,
                                                            train_batch_size=10,
                                                            test_batch_size=10,
                                                            epochs_to_save=None,
                                                            take_batch=take_batch,
                                                            verbose_training=False)

    influence_factory = RPSL2Factory(CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE),
                                     lambda_regularization=10.0)
    result = cifar10_evaluator.evaluate(influence_factory=influence_factory, nbr_of_evaluation=2, verbose=False)
    assert np.shape(result[0]) == (2, take_batch)
    assert np.shape(result[1]) == (take_batch,)
