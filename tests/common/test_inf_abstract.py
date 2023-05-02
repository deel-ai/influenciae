# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Reduction, MeanSquaredError

from deel.influenciae.common import InfluenceModel
from deel.influenciae.common import ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP
from deel.influenciae.influence import FirstOrderInfluenceCalculator


def test_instantiation():
    """
    Test that the instantiation happens as it should
    """
    # start with a simple model
    model = Sequential([Input(shape=(1, 3)), Dense(2, use_bias=False), Dense(1, use_bias=False)])
    model.build(input_shape=(1, 3))

    # build the influence model
    influence_model = InfluenceModel(model, start_layer=-1, loss_function=MeanSquaredError(reduction=Reduction.NONE))

    # build a fake dataset in order to have batched samples
    inputs = tf.random.normal((25, 1, 3))
    target = tf.random.normal((25, 1))
    train_set = tf.data.Dataset.from_tensor_slices((inputs, target))

    # Test several configurations
    ihvp_objects = [ExactIHVP(influence_model, train_set.batch(5)),
                    ConjugateGradientDescentIHVP(influence_model, -1, train_set.batch(5)),
                    LissaIHVP(influence_model, -1, train_set.batch(5))]
    ihvp_strings = ["exact", "cgd", "lissa"]
    ihvp_classes = [ExactIHVP, ConjugateGradientDescentIHVP, LissaIHVP]
    normalization = [True, False]

    for ihvp_calculator in ihvp_objects:
        for normalize in normalization:
            FirstOrderInfluenceCalculator(
                influence_model,
                train_set.batch(5),
                ihvp_calculator,
                n_samples_for_hessian=25,
                shuffle_buffer_size=25,
                normalize=normalize
            )

    for ihvp_string, classes in zip(ihvp_strings, ihvp_classes):
        for normalize in normalization:
            influence_calculator = FirstOrderInfluenceCalculator(
                influence_model,
                train_set.batch(5),
                ihvp_string,
                n_samples_for_hessian=25,
                shuffle_buffer_size=25,
                normalize=normalize
            )
            assert isinstance(influence_calculator.ihvp_calculator, classes)
