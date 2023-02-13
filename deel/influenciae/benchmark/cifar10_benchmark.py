# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
A module implementing a mislabeled sample detection on the CIFAR-10 image classifiction
dataset.
For each evaluation, a model will be trained on the noisy dataset, the self-influence
will be computed for each point, and they will get sorted by this value. A well
performing influence calculator technique should be able to put forth the mislabeled
samples among the top-most influential points.
"""
import ssl

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model # pylint: disable=E0611
from tensorflow.keras.models import Sequential # pylint: disable=E0611
from tensorflow.keras.applications import EfficientNetB0, VGG19 # pylint: disable=E0611
from tensorflow.keras.layers import Dense, Flatten, Dropout # pylint: disable=E0611
from tensorflow.keras.regularizers import L1L2 # pylint: disable=E0611
from tensorflow.keras.losses import CategoricalCrossentropy # pylint: disable=E0611
from tensorflow.keras.optimizers import Adam # pylint: disable=E0611

from .base_benchmark import BaseTrainingProcedure, MislabelingDetectorEvaluator, ModelsSaver
from .model_resnet import ResNet

from ..types import Tuple, Union, Any, Optional, List

ssl._create_default_https_context = ssl._create_unverified_context # pylint: disable=W0212


class ConvNetCIFAR(Sequential):
    """
    A class with some convolutional neural network implementations on which to benchmark the
    different influence calculation techniques. The feature extractor will either be a
    ResNet18, an EfficientNetB0 or a VGG-19.

    Parameters
    ----------
    model
        Either a string for the base model ('resnet', 'efficient_net' or 'vgg19'), or a TF
        base model itself.
    use_regularization
        A boolean indicating whether to add regularization on the final model's last layer.
    """
    def __init__(self, model: Union[str, Model], use_regularization: bool = True, **kwargs):
        super().__init__(**kwargs)
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
                raise ValueError('unknown model=' + model)
        self.add(base_model)
        self.add(Flatten())

        if use_regularization:
            dense_1 = Dense(128, kernel_regularizer=L1L2(l1=1e-4, l2=1e-4))
        else:
            dense_1 = Dense(128)
        self.add(dense_1)

        self.add(Dropout(0.4))
        self.add(tf.keras.layers.LeakyReLU())

        if use_regularization:
            dense_2 = Dense(10, kernel_regularizer=L1L2(l1=1e-4, l2=1e-4), kernel_initializer="he_normal")
        else:
            dense_2 = Dense(10)

        self.add(dense_2)


class Cifar10TrainingProcedure(BaseTrainingProcedure):
    """
    A class for streamlining the process of training models on the CIFAR-10 dataset.

    Parameters
    ----------
    epochs
        An integer indicating for how long the model should be trained
    model_type
        A string with the type of model to use. Either 'resnet', 'efficient_net' or 'vgg19'.
    use_regu
        A boolean indicating whether L1L2 regularization should be used on the last layer.
    force_overfit
        A boolean for if the training schedule to be used should try to overfit the model or not.
    epochs_to_save
        A list of integers for the eventual saving of the model's checkpoints and training information.
        Useful for computing influence-related quantities using TracIn.
    verbose
        A boolean indicating whether to print the progress to stdout.
    use_tensorboard
        A boolean indicating the use of tensorboard for logging.
    """
    def __init__(
            self,
            epochs: int = 60,
            model_type: str = 'resnet',
            use_regu: bool = True,
            force_overfit: bool = False,
            epochs_to_save: Optional[List[int]] = None,
            verbose: bool = True,
            use_tensorboard: bool = False
    ):
        self.epochs = epochs
        self.model_type = model_type
        self.use_regu = use_regu
        self.force_overfit = force_overfit
        self.epochs_to_save = epochs_to_save
        self.verbose = verbose
        self.use_tensorboard = use_tensorboard
        # first order
        self.schedule = [(10, 0.5), (20, 0.5), (30, 0.5), (40, 0.5), (50, 0.5)]
        self.learning_rate = 1E-3

    def train(
            self,
            training_dataset: tf.data.Dataset,
            test_dataset: tf.data.Dataset,
            train_batch_size: int = 128,
            test_batch_size: int = 128,
            log_path: Optional[str] = None
    ) -> Tuple[float, float, tf.keras.Model, Any]:
        """
        Trains the desired model with the object's training schedule on the provided training dataset, and
        validates its performance on the test dataset. Optionally, it logs the results progressively
        using the tensorboard framework.

        Parameters
        ----------
        training_dataset
            A TF dataset containing the samples on which the model will be trained. Typically, for this
            benchmark, it will be a noisy version of CIFAR-10.
        test_dataset
            A TF dataset with the test data on which to validate the model's accuracy. This data should
            only contain clean samples (i.e. no flipped labels).
        train_batch_size
            An integer specifying the batch size for the training procedure.
        test_batch_size
            An integer specifying the batch size for the validation procedure.
        log_path
            An (optional) string with the path onto which to save the tensorboard logs.

        Returns
        -------
        train_stats, test_stats, model, model_saver_information
            A tuple with the model's train accuracy, test accuracy, final model and the model saver information set to
            None for now
        """
        random_translation = tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest")

        def preprocess(x):
            x = tf.image.random_flip_left_right(x)
            x = random_translation(x)

            return x

        training_dataset = training_dataset.shuffle(1000).batch(train_batch_size)
        training_dataset_augment = training_dataset.map(lambda x, y: (preprocess(x), y))
        training_dataset_augment = training_dataset_augment.prefetch(100)

        test_dataset = test_dataset.batch(test_batch_size).prefetch(100)

        model = ConvNetCIFAR(self.model_type, self.use_regu)

        loss = CategoricalCrossentropy(from_logits=True)

        if self.force_overfit:
            def LearningRateSchedulerMaxMin(lr_start=0.01, lr_end=0.0001, epochs=100):
                lr_decay = (lr_end / lr_start) ** (1. / epochs)
                return tf.keras.callbacks.LearningRateScheduler(lambda e: lr_start * lr_decay ** e)

            lr_scheduler = LearningRateSchedulerMaxMin(0.1, 0.0001, epochs=200)
            callbacks = [lr_scheduler]
            optimizer = tf.keras.optimizers.SGD(0.1, momentum=0.9)
        else:
            def lr_schedule(epoch: int):
                lr = self.learning_rate
                for ep_max, coeff in self.schedule:
                    if epoch >= ep_max:
                        lr *= coeff
                    else:
                        break
                return lr

            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                              cooldown=1,
                                                              patience=3,
                                                              min_lr=0.5e-6)
            callbacks = [lr_reducer, lr_scheduler]
            optimizer = Adam(learning_rate=1e-3)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        model_saver = None
        if self.epochs_to_save is not None:
            model_saver = ModelsSaver(self.epochs_to_save, optimizer)
            callbacks = [model_saver] + callbacks

        if self.use_tensorboard:
            if log_path is None:
                log_path = "./"
            callbacks.append(tf.keras.callbacks.TensorBoard(log_path))

        model.fit(training_dataset_augment, epochs=self.epochs, verbose=1 if self.verbose else 0,
                  validation_data=test_dataset,
                  callbacks=callbacks)
        _, train_stats = model.evaluate(training_dataset, batch_size=train_batch_size, verbose=0)
        _, test_stats = model.evaluate(test_dataset, batch_size=test_batch_size, verbose=0)

        if self.epochs_to_save is not None:
            return train_stats, test_stats, model, (model_saver.models, model_saver.learning_rates)

        return train_stats, test_stats, model, None


class Cifar10MislabelingDetectorEvaluator(MislabelingDetectorEvaluator):
    """
    An implementation of the mislabeled sample detection benchmark on the CIFAR10 image classification
    dataset. Useful for streamlining the evaluation of the different influence calculator techniques.

    Parameters
    ----------
    epochs
        An integer indicating for how long the model should be trained
    model_type
        A string with the type of model to use. Either 'resnet', 'efficient_net' or 'vgg19'.
    mislabeling_ratio
        A float with the ratio of noise to add to the training dataset's labels (ranging from 0. to 1.).
    use_regu
        A boolean indicating whether L1L2 regularization should be used on the last layer.
    force_overfit
        A boolean for if the training schedule to be used should try to overfit the model or not.
    train_batch_size
        An integer with the size of the batches on which to train the model.
    test_batch_size
        An integer with the size of the batches on which to perform the validation.
    influence_batch_size
        An integer with the size of the batches for performing the different operations
    epochs_to_save
        A list of integers for the eventual saving of the model's checkpoints and training information.
        Useful for computing influence-related quantities using TracIn.
    take_batch
        An optional integer indicating the number of batches to take if only one part of the
        training and test datasets are to be used.
    verbose_training
        A boolean indicating if progress about the training procedures should be reported to stdout.
    use_tensorboard
        A boolean indicating the use of tensorboard for logging.
    """
    def __init__(
            self,
            epochs: int = 60,
            model_type: str = 'resnet',
            mislabeling_ratio: float = 0.0005,
            use_regu: bool = True,
            force_overfit: bool = False,
            train_batch_size: int = 128,
            test_batch_size: int = 128,
            influence_batch_size: Optional[int] = None,
            epochs_to_save: Optional[List[int]] = None,
            take_batch: Optional[int] = None,
            verbose_training: bool = True,
            use_tensorboard: bool = False
        ): # pylint: disable=R0913

        config = {
            "epochs": epochs,
            "model_type": model_type,
            "mislabeling_ratio": mislabeling_ratio,
            "use_regularization": use_regu,
            "optimizer": 'sgd' if force_overfit else 'adam',
            "train_batch_size": train_batch_size,
            "test_batch_size": test_batch_size,
            "epochs_to_save": epochs_to_save if epochs_to_save is not None else [],
            "samples_to_take": take_batch if take_batch is not None else -1,
        }

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.cast(2. * x_train / 255. - 1., dtype=tf.float32)
        x_test = tf.cast(2. * x_test / 255. - 1., dtype=tf.float32)

        y_train = tf.cast(tf.squeeze(tf.one_hot(y_train, depth=10, axis=1), axis=-1), dtype=tf.float32)
        y_test = tf.cast(tf.squeeze(tf.one_hot(y_test, depth=10, axis=1), axis=-1), dtype=tf.float32)

        training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if take_batch is not None:
            training_dataset = training_dataset.take(take_batch)
            test_dataset = test_dataset.take(take_batch)
        training_procedure = Cifar10TrainingProcedure(epochs, model_type, use_regu, force_overfit, epochs_to_save,
                                                      verbose_training, use_tensorboard)
        super().__init__(training_dataset, test_dataset, training_procedure,
                         nb_classes=10, mislabeling_ratio=mislabeling_ratio,
                         train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size,
                         influence_batch_size=influence_batch_size,
                         config=config)
