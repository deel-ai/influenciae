# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
TensorFlow-specific utilities for benchmark workflows.
"""
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer  # pylint: disable=E0611

from ..types import Optional, Dict, Any, List


class ModelsSaver(tf.keras.callbacks.Callback):
    """
    Save models and learning rates at selected epochs during training.

    Parameters
    ----------
    epochs_to_save
        A list of integers indicating on which epochs to save a model's checkpoint.
    optimizer
        The model's optimizer.
    saving_path
        An optional string for saving the results on disk.
    """

    def __init__(self, epochs_to_save: List[int], optimizer: Optimizer, saving_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.epochs_to_save = epochs_to_save
        self.optimizer = optimizer

        self.models: List[Any] = []
        self.learning_rates: List[float] = []

        if saving_path is not None and not os.path.exists(saving_path):
            os.mkdir(saving_path)
        self.saving_path = saving_path

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """
        Save model checkpoints and optimizer learning rates at epoch end.

        Parameters
        ----------
        epoch
            Current epoch number.
        logs
            Training/validation metric dictionary for the epoch.
        """
        if epoch in self.epochs_to_save:
            epoch_model = tf.keras.models.clone_model(self.model)
            epoch_model.build(self.model.input_shape)
            epoch_model.set_weights(self.model.get_weights())

            epoch_lr = self.optimizer.lr
            self.models.append(epoch_model)
            self.learning_rates.append(epoch_lr.numpy())

            if self.saving_path is not None:
                tf.data.experimental.save(f"{self.saving_path}/model_ep_{epoch:.6d}")
                np.save(f"{self.saving_path}/learning_rates", np.array(self.learning_rates), allow_pickle=True)
                with open(f"{self.saving_path}/logs.json", "w", encoding='utf8') as file:
                    json.dump(logs, file)
