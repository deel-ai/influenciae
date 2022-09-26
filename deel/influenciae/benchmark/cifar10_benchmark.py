# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
CIFAR-10 Benchmark module
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0, VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from .base_benchmark import BaseTrainingProcedure, MissingLabelEvaluator
from .model_resnet import ResNet

from ..types import Tuple, Union, Any

# class EfficientNetCIFAR(Model):
#     def __init__(self, model: Union[str, Model], use_regu=True, **kwargs):
#         super(EfficientNetCIFAR, self).__init__(**kwargs)
#         if isinstance(model, Model):
#             self.base_model = model
#         else:
#             if model == 'resnet':
#                 self.base_model = ResNet(input_shape=(32, 32, 3), include_top=False, block='basic_block',
#                                          residual_unit='v1',
#                                          repetitions=(3, 3, 3), initial_filters=16, initial_pooling=None,
#                                          final_pooling=None)
#             elif model == 'efficient_net':
#                 self.base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(32, 32, 3))
#             elif model == 'vgg19':
#                 self.base_model = VGG19(include_top=False, weights=None, input_shape=(32, 32, 3))
#             else:
#                 raise Exception('unknown model=' + model)

#         self.flatten = Flatten()

#         if use_regu:
#             self.dense_1 = Dense(128, kernel_regularizer=L1L2(l1=1e-4, l2=1e-4))
#         else:
#             self.dense_1 = Dense(128)

#         self.lrelu = tf.keras.layers.LeakyReLU()
#         self.drop_1 = Dropout(0.4)
#         if use_regu:
#             self.dense_2 = Dense(10, kernel_regularizer=L1L2(l1=1e-4, l2=1e-4), kernel_initializer="he_normal")
#         else:
#             self.dense_2 = Dense(10)

#     def call(self, inputs, training=None, mask=None):
#         x = self.base_model(inputs)
#         x = self.flatten(x)
#         x = self.dense_1(x)
#         x = self.drop_1(x)
#         x = self.lrelu(x)
#         x = self.dense_2(x)

#         return x

class EfficientNetCIFAR(Sequential):
    def __init__(self, model: Union[str, Model], use_regu=True, **kwargs):
        super(EfficientNetCIFAR, self).__init__(**kwargs)
        if isinstance(model, Model):
            base_model = model
        else:
            if model == 'resnet':
                base_model = ResNet(input_shape=(32, 32, 3), include_top=False, block='basic_block',
                                         residual_unit='v1',
                                         repetitions=(3, 3, 3), initial_filters=16, initial_pooling=None,
                                         final_pooling=None)
            elif model == 'efficient_net':
                base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(32, 32, 3))
            elif model == 'vgg19':
                base_model = VGG19(include_top=False, weights=None, input_shape=(32, 32, 3))
            else:
                raise Exception('unknown model=' + model)
        self.add(base_model)
        self.add(Flatten())

        if use_regu:
            dense_1 = Dense(128, kernel_regularizer=L1L2(l1=1e-4, l2=1e-4))
        else:
            dense_1 = Dense(128)
        self.add(dense_1)

        self.add(Dropout(0.4))
        self.add(tf.keras.layers.LeakyReLU())
        
        if use_regu:
            dense_2 = Dense(10, kernel_regularizer=L1L2(l1=1e-4, l2=1e-4), kernel_initializer="he_normal")
        else:
            dense_2 = Dense(10)
        
        self.add(dense_2)

    # def call(self, inputs, training=None, mask=None):
    #     x = self(inputs)

    #     return x


class Cifar10TrainingProcedure(BaseTrainingProcedure):
    """
    TODO: Docs
    """
    def __init__(self, epochs=60, model_type: str = 'resnet', use_regu: bool = True, sgd=False):
        self.epochs = epochs
        self.model_type = model_type
        self.use_regu = use_regu
        self.sgd = sgd

    def train(
        self,
        training_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        train_batch_size: int = 128,
        test_batch_size: int = 128) -> Tuple[float, float, tf.keras.Model, Any]:
        """
        TODO: Docs
        """
        random_translation = tf.keras.layers.RandomTranslation(0.1, 0.1)

        def preprocess(x):
            x = random_translation(x)
            x = tf.image.random_flip_left_right(x)
            return x

        training_dataset = training_dataset.batch(train_batch_size)
        training_dataset_augment = training_dataset.map(lambda x, y: (preprocess(x), y))
        training_dataset_augment = training_dataset_augment.prefetch(100)

        test_dataset = test_dataset.batch(test_batch_size).prefetch(100)

        model = EfficientNetCIFAR(self.model_type, self.use_regu)

        loss = CategoricalCrossentropy(from_logits=True)

        if self.sgd:
            def LearningRateSchedulerMaxMin(lr_start=0.01, lr_end=0.0001, epochs=100):
                lr_decay = (lr_end / lr_start) ** (1. / epochs)
                return tf.keras.callbacks.LearningRateScheduler(lambda e: lr_start * lr_decay ** e)

            lr_scheduler = LearningRateSchedulerMaxMin(0.1, 0.0001, epochs=200)
            callbacks = [lr_scheduler]
            optimizer = tf.keras.optimizers.SGD(0.1, momentum=0.9)
        else:
            def lr_schedule(epoch: int):
                learning_rate = 1e-3  # 0.001
                if epoch >= 10:
                    learning_rate /= 2
                if epoch >= 20:
                    learning_rate /= 2
                if epoch >= 30:
                    learning_rate /= 2
                if epoch >= 40:
                    learning_rate /= 2
                if epoch >= 50:
                    learning_rate /= 2
                return learning_rate

            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                              cooldown=1,
                                                              patience=3,
                                                              min_lr=0.5e-6)
            callbacks = [lr_reducer, lr_scheduler]
            optimizer = Adam(learning_rate=1e-3)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        """
        training_dataset_augment = training_dataset_augment.take(3)
        test_dataset = test_dataset.take(3)
        training_dataset = training_dataset.take(3)
        """

        model.fit(training_dataset_augment, epochs=self.epochs, verbose=1,
                  validation_data=test_dataset,
                  callbacks=callbacks)
        _, train_stats = model.evaluate(training_dataset, batch_size=train_batch_size, verbose=0)
        _, test_stats = model.evaluate(test_dataset, batch_size=test_batch_size, verbose=0)

        # TODO: implement histo train pour traceIn
        return train_stats, test_stats, model, None


class Cifar10MissingLabelEvaluator(MissingLabelEvaluator):
    """
    TODO: Docs
    """
    def __init__(
        self,
        epochs: int = 60,
        model_type: str = 'resnet',
        misslabeling_ratio: float = 0.0005,
        use_regu: bool = True,
        sgd: bool = False,
        train_batch_size: int = 128,
        test_batch_size: int = 128):

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = 2. * x_train / 255. - 1.
        x_test = 2. * x_test / 255. - 1.

        y_train = tf.squeeze(tf.one_hot(y_train, depth=10, axis=1),axis=-1)
        y_test = tf.squeeze(tf.one_hot(y_test, depth=10, axis=1),axis=-1)

        training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        training_dataset = training_dataset.take(10)
        test_dataset = test_dataset.take(10)
        training_procedure = Cifar10TrainingProcedure(epochs, model_type, use_regu, sgd)
        super(Cifar10MissingLabelEvaluator, self).__init__(training_dataset, test_dataset, training_procedure,
                                                           nb_classes=10,misslabeling_ratio=misslabeling_ratio,
                                                           train_batch_size=train_batch_size, test_batch_size=test_batch_size)
