# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import pytest

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import (Reduction, MeanSquaredError)

from deel.influenciae.common import (
    InfluenceModel,
    ExactIHVP,
    ConjugateGradientDescentIHVP,
    LissaIHVP,
    KfacIHVP,
    EkfacIHVP,
    InverseHessianVectorProductFactory,
    ExactIHVPFactory,
    CGDIHVPFactory,
    LissaIHVPFactory,
    KfacIHVPFactory,
    EkfacIHVPFactory,
)

from ..utils_test import almost_equal


pytestmark = pytest.mark.tensorflow


def test_exact_factory():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target)).batch(5)

    ihvp = ExactIHVP(influence_model, train_set)
    exact_factory = ExactIHVPFactory()
    assert isinstance(exact_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = exact_factory.build(influence_model, train_set)
    assert isinstance(ihvp_from_factory, ExactIHVP)

    assert almost_equal(ihvp.inv_hessian, ihvp_from_factory.inv_hessian, epsilon=1e-3)


def test_cgd_factory():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target)).batch(5)

    n_cgd_iters = 100

    # case 1: model feature extractor
    feature_extractor = Sequential(model.layers[:1])
    ihvp = ConjugateGradientDescentIHVP(influence_model, 1, train_set, n_cgd_iters, feature_extractor)
    cgd_factory = CGDIHVPFactory(feature_extractor, n_cgd_iters, 1)
    assert isinstance(cgd_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = cgd_factory.build(influence_model, train_set)
    assert ihvp.extractor_layer == ihvp_from_factory.extractor_layer
    assert len(ihvp.feature_extractor.layers) == len(ihvp_from_factory.feature_extractor.layers)
    assert ihvp.weights == ihvp_from_factory.weights
    for ihvp_batch, factory_batch in zip(ihvp.train_set, ihvp_from_factory.train_set):
        assert almost_equal(ihvp_batch[0], factory_batch[0])
        assert almost_equal(ihvp_batch[1], factory_batch[1])

    # case 2: layer position feature extractor
    feature_extractor = 1
    ihvp = ConjugateGradientDescentIHVP(influence_model, 1, train_set, n_cgd_iters, None)
    cgd_factory = CGDIHVPFactory(feature_extractor, n_cgd_iters)
    assert isinstance(cgd_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = cgd_factory.build(influence_model, train_set)
    assert ihvp.extractor_layer == ihvp_from_factory.extractor_layer
    assert len(ihvp.feature_extractor.layers) == len(ihvp_from_factory.feature_extractor.layers)
    assert ihvp.weights == ihvp_from_factory.weights
    for ihvp_batch, factory_batch in zip(ihvp.train_set, ihvp_from_factory.train_set):
        assert almost_equal(ihvp_batch[0], factory_batch[0])
        assert almost_equal(ihvp_batch[1], factory_batch[1])

    # case 3: model feature extractor without layer position
    feature_extractor = Sequential(model.layers[:1])
    with pytest.raises(AssertionError):
        cgd_factory = CGDIHVPFactory(feature_extractor, n_cgd_iters)


def test_lissa_factory():
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target)).batch(5)

    n_lissa_iters = 100

    # case 1: model feature extractor
    feature_extractor = Sequential(model.layers[:1])
    ihvp = LissaIHVP(influence_model, 1, train_set, n_lissa_iters, feature_extractor)
    lissa_factory = LissaIHVPFactory(feature_extractor, n_lissa_iters, 1)
    assert isinstance(lissa_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = lissa_factory.build(influence_model, train_set)
    assert ihvp.extractor_layer == ihvp_from_factory.extractor_layer
    assert len(ihvp.feature_extractor.layers) == len(ihvp_from_factory.feature_extractor.layers)
    assert ihvp.weights == ihvp_from_factory.weights
    for ihvp_batch, factory_batch in zip(ihvp.train_set, ihvp_from_factory.train_set):
        assert almost_equal(ihvp_batch[0], factory_batch[0])
        assert almost_equal(ihvp_batch[1], factory_batch[1])

    # case 2: layer position feature extractor
    feature_extractor = 1
    ihvp = LissaIHVP(influence_model, 1, train_set, n_lissa_iters, None)
    lissa_factory = LissaIHVPFactory(feature_extractor, n_lissa_iters)
    assert isinstance(lissa_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = lissa_factory.build(influence_model, train_set)
    assert ihvp.extractor_layer == ihvp_from_factory.extractor_layer
    assert len(ihvp.feature_extractor.layers) == len(ihvp_from_factory.feature_extractor.layers)
    assert ihvp.weights == ihvp_from_factory.weights
    for ihvp_batch, factory_batch in zip(ihvp.train_set, ihvp_from_factory.train_set):
        assert almost_equal(ihvp_batch[0], factory_batch[0])
        assert almost_equal(ihvp_batch[1], factory_batch[1])

    # case 3: model feature extractor without layer position
    feature_extractor = Sequential(model.layers[:1])
    with pytest.raises(AssertionError):
        lissa_factory = LissaIHVPFactory(feature_extractor, n_lissa_iters)


def test_kfac_factory(tmp_path):
    tf.random.set_seed(42)

    model = Sequential([
        Input(shape=(4,), dtype=tf.float64),
        Dense(3, use_bias=False, dtype=tf.float64),
        Dense(2, use_bias=False, dtype=tf.float64),
    ])
    influence_model = InfluenceModel(
        model,
        start_layer=0,
        last_layer=-1,
        loss_function=MeanSquaredError(reduction=Reduction.NONE),
    )

    inputs = tf.random.normal((10, 4), dtype=tf.float64)
    target = tf.random.normal((10, 2), dtype=tf.float64)
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target)).batch(5)

    factors_path = str(tmp_path / "kfac_factors")
    kfac_factory = KfacIHVPFactory(
        damping=1e-3,
        layer_collection="recursive",
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="memory",
        factors_path=factors_path,
        overwrite_factors=True,
    )
    assert isinstance(kfac_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = kfac_factory.build(influence_model, train_set)
    assert isinstance(ihvp_from_factory, KfacIHVP)
    assert ihvp_from_factory.layer_map.layer_collection == "recursive"
    assert ihvp_from_factory.factors.module_partition_size == 1
    assert ihvp_from_factory.factors.offload_activations_to_cpu
    assert ihvp_from_factory.factors.data_partition_size == 2
    assert ihvp_from_factory.factors.accumulator_offload_mode == "memory"
    assert ihvp_from_factory.factors_path == factors_path
    assert ihvp_from_factory.overwrite_factors


def test_ekfac_factory(tmp_path):
    tf.random.set_seed(42)

    model = Sequential([
        Input(shape=(4,), dtype=tf.float64),
        Dense(3, use_bias=False, dtype=tf.float64),
        Dense(2, use_bias=False, dtype=tf.float64),
    ])
    influence_model = InfluenceModel(
        model,
        start_layer=0,
        last_layer=-1,
        loss_function=MeanSquaredError(reduction=Reduction.NONE),
    )

    inputs = tf.random.normal((10, 4), dtype=tf.float64)
    target = tf.random.normal((10, 2), dtype=tf.float64)
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target)).batch(5)

    factors_path = str(tmp_path / "ekfac_factors")
    ekfac_factory = EkfacIHVPFactory(
        damping=1e-3,
        layer_collection="recursive",
        module_partition_size=1,
        offload_activations_to_cpu=True,
        data_partition_size=2,
        accumulator_offload_mode="memory",
        factors_path=factors_path,
        overwrite_factors=True,
    )
    assert isinstance(ekfac_factory, InverseHessianVectorProductFactory)

    ihvp_from_factory = ekfac_factory.build(influence_model, train_set)
    assert isinstance(ihvp_from_factory, EkfacIHVP)
    assert ihvp_from_factory.layer_map.layer_collection == "recursive"
    assert ihvp_from_factory.factors.module_partition_size == 1
    assert ihvp_from_factory.factors.offload_activations_to_cpu
    assert ihvp_from_factory.factors.data_partition_size == 2
    assert ihvp_from_factory.factors.accumulator_offload_mode == "memory"
    assert ihvp_from_factory.factors_path == factors_path
    assert ihvp_from_factory.overwrite_factors
