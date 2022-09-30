# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TODO
"""
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential # pylint: disable=E0611

from ..common import InfluenceModel
from ..common import InverseHessianVectorProductFactory

from ..influence import FirstOrderInfluenceCalculator
from ..types import Union, Optional


class RPSLJE(FirstOrderInfluenceCalculator):
    """
    Representer Point Selection via Local Jacobian Expansion for Post-hoc Classifier Explanation of Deep Neural
    Networks and Ensemble Models
    https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    ihvp_calculator
        Either a string containing the IHVP method ('exact' or 'cgd'), an IHVPCalculator
        object or an InverseHessianVectorProduct object.
    target_layer
        TODO: Define the argument
    """

    def __init__(
            self,
            model_influence: InfluenceModel,
            dataset: tf.data.Dataset,
            ihvp_calculator_factory: InverseHessianVectorProductFactory,
            n_samples_for_hessian: Optional[int] = None,
            target_layer: Union[int, str] = -1,
            shuffle_buffer_size: Optional[int] = 10000
    ):
        ihvp_calculator = ihvp_calculator_factory.build(model_influence, dataset)
        first_order_calculator = FirstOrderInfluenceCalculator(model=model_influence,
                                                               dataset=dataset,
                                                               ihvp_calculator=ihvp_calculator,
                                                               n_samples_for_hessian=n_samples_for_hessian,
                                                               shuffle_buffer_size=shuffle_buffer_size,
                                                               normalize=False)

        influence_vector_dataset = first_order_calculator.compute_influence_vector_dataset(dataset)

        size = tf.data.experimental.cardinality(dataset)
        iter_dataset = iter(influence_vector_dataset)

        weight_size = tf.reduce_sum(
            tf.stack([tf.reduce_prod(tf.shape(w)) for w in model_influence.model.layers[target_layer].weights]))

        def body(i, v, nb):
            current_vector = next(iter_dataset)[1]

            nb_next = nb + tf.cast(tf.shape(current_vector)[0], dtype=nb.dtype)

            v_current = tf.reduce_sum(current_vector, axis=0)

            v_next = (nb / nb_next) * v + v_current / nb_next

            return i + tf.constant(1, dtype=size.dtype), v_next, nb_next

        dtype_ = dataset.element_spec[0].dtype
        _, influence_vector, __ = tf.while_loop(cond=lambda i, v, nb: i < size,
                                                        body=body,
                                                        loop_vars=[tf.constant(0, dtype=size.dtype),
                                                                   tf.zeros((weight_size,),
                                                                            dtype=dtype_),
                                                                   tf.constant(0.0, dtype=dtype_)])

        layers_end = model_influence.model.layers[target_layer:]
        weights = [lay.weights for lay in layers_end]
        weights = list(itertools.chain(*weights))

        model_end = tf.keras.models.clone_model(Sequential(layers_end))

        input_layer_shape = model_influence.model.layers[target_layer].input.type_spec.shape
        model_end.build(input_layer_shape)
        model_end.set_weights(weights)

        self._reshape_assign(model_end.layers[0].weights, influence_vector)

        features_extractor = Sequential(model_influence.model.layers[:target_layer])

        model = InfluenceModel(Sequential([features_extractor, model_end]), 1,
                               loss_function=model_influence.loss_function)

        ihvp_calculator = ihvp_calculator_factory.build(model, dataset)

        super().__init__(model=model,
                         dataset=dataset,
                         ihvp_calculator=ihvp_calculator,
                         n_samples_for_hessian=n_samples_for_hessian,
                         shuffle_buffer_size=shuffle_buffer_size,
                         normalize=False)

    def _reshape_assign(self, weights, influence_vector):
        index = 0
        for w in weights:
            shape = tf.shape(w)
            size = tf.reduce_prod(shape)
            v = influence_vector[index:(index + size)]
            index += size
            v = tf.reshape(v, shape)
            w.assign(w - v)
