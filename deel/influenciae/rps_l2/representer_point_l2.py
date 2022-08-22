# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
A module implementing the original Representer point method for calculating the influence
of training points as per
https://proceedings.neurips.cc/paper/2018/file/8a7129b8f3edd95b7d969dfc2c8e9d9d-Paper.pdf
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.regularizers import L2

from ..common import assert_batched_dataset, BacktrackingLineSearch, dataset_size


class RepresenterPointL2:
    """
    A class implementing a method to compute the influence of training points through
    the representer point theorem for kernels.

    Parameters
    ----------
    model
        A TF2 model that has already been trained
    train_set
        A batched TF dataset with the points with which the model was trained
    lambda_regularization
        The coefficient for the regularization of the surrogate last layer that needs
        to be trained for this method
    """
    def __init__(
            self,
            model: Model,
            train_set: tf.data.Dataset,
            lambda_regularization: float
    ):
        assert_batched_dataset(train_set)
        self.n_train = dataset_size(train_set)
        self.feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
        self.model = model
        self.train_set = train_set
        self.lambda_regularization = lambda_regularization
        self.linear_layer = None
        self.weight_matrix_ds = None

    def compute_influence_values(self, dataset_to_evaluate: tf.data.Dataset) -> tf.Tensor:
        """
        Computes the influence of each point in the training set wrt those in the provided dataset.

        Parameters
        ----------
        dataset_to_evaluate
            A TF dataset with the points wrt to which we wish to compute the influence of the points
            in the training dataset

        Returns
        -------
        influence_values
            A TF Tensor with the values of the influence of each training point in each of the test
            points
        """
        assert_batched_dataset(dataset_to_evaluate)
        if self.linear_layer is None:
            self._train_last_layer()
        self.weight_matrix_ds = self._compute_gradients(dataset_to_evaluate).map(
            lambda x, y, g: (x, y, tf.divide(g, -2. * self.lambda_regularization * tf.cast(self.n_train, tf.float32)))
        )
        influence_values = tf.concat([inf for inf in self.weight_matrix_ds.map(lambda x, y, g: g)], axis=0)

        return influence_values

    def predict_with_kernel(self, dataset_to_evaluate: tf.data.Dataset) -> tf.Tensor:
        """
        Using the representer point theorem, use the influence information to estimate the model's
        predictions.

        Parameters
        ----------
        dataset_to_evaluate
            A TF dataset with the points we wish to predict using the kernel

        Returns
        -------
        predictions
            A TF tensor with the predictions
        """
        if self.weight_matrix_ds is None:
            self.compute_influence_values(dataset_to_evaluate)
        return tf.concat([tf.matmul(f_batch, tf.matmul(f_batch, w_batch, transpose_a=True))
                          for f_batch, w_batch in
                          self.weight_matrix_ds.map(lambda x, y, w: (self.feature_extractor(x), w))], axis=0)

    def _compute_gradients(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Computes the gradients of the loss wrt the network's embedding for the point in
        the provided dataset.

        Parameters
        ----------
        dataset
            A TF dataset with the points wrt to which we wish to compute the gradients

        Returns
        -------
        dataset_with_grads
            A TF dataset containing the provided dataset's points and their corresponding gradients
        """
        dataset_with_grads = None
        for x_batch, y_batch in dataset.map(lambda x, y: (self.feature_extractor(x), y)):
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(x_batch)
                tape.watch(self.linear_layer.trainable_weights)
                logits = self.linear_layer(x_batch)
                loss = self.linear_layer.compiled_loss(y_batch, logits)
            grads = tf.reduce_mean(tape.gradient(loss, logits), axis=1)
            if dataset_with_grads is None:
                dataset_with_grads = tf.data.Dataset.from_tensor_slices((x_batch, y_batch, grads))
            else:
                dataset_with_grads = dataset_with_grads.concatenate(
                    tf.data.Dataset.from_tensor_slices((x_batch, y_batch, grads))
                )

        return dataset_with_grads

    def _train_last_layer(self, epochs: int = 100):
        """
        Trains an L2-regularized surrogate linear model to predict like the model on the
        training dataset. The optimization is done using a Backtracking Line-Search
        algorithm with the Armijo condition and SGD as the optimizer as it was done
        in the original implementation.

        Parameters
        ----------
        epochs
            An integer with the amount of epochs to train the surrogate model
        """
        self.linear_layer = self._create_surrogate_model()
        optimizer = BacktrackingLineSearch(batches_per_epoch=self.n_train / self.train_set._batch_size,
                                           c=0.1)  # the optimizer used in the paper's code
        loss_function = self.model.compiled_loss._losses[0] if isinstance(self.model.compiled_loss._losses, list) \
            else self.model.compiled_loss
        mse_loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        feature_map_ds = self.train_set.map(lambda x, y: self.feature_extractor(x))
        surrogate_train_set = tf.data.Dataset.zip((feature_map_ds,
                                                   feature_map_ds.map(lambda x: self.model.layers[-1](x))))
        self.linear_layer.compile(optimizer=optimizer, loss=mse_loss)
        for epoch in range(epochs):
            for x_batch, y_batch in surrogate_train_set:
                with tf.GradientTape() as tape:
                    logits = self.linear_layer(x_batch, training=True)
                    loss = mse_loss(y_batch, logits)
                gradients = tape.gradient(loss, self.linear_layer.trainable_weights)
                optimizer.step(self.linear_layer, loss, x_batch, y_batch, gradients)
        self.linear_layer.compile(optimizer=optimizer, loss=loss_function)

    def _create_surrogate_model(self) -> Model:
        """
        Instances an L2-regularized linear model to use as surrogate with the
        right input and output shapes.

        Returns
        -------
        surrogate_model
            A TF2 L2-regularized linear model
        """
        inputs = Input(shape=self.feature_extractor.output_shape[1:])
        last_layer = Dense(self.model.output_shape[-1], use_bias=False,
                           kernel_regularizer=L2(self.lambda_regularization))
        outputs = last_layer(inputs)
        surrogate_model = Model(inputs=inputs, outputs=outputs)
        surrogate_model.layers[-1].trainable = True
        surrogate_model.compile(loss=self.model.compiled_loss)

        return surrogate_model


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def plot_top25_images(images, influence_values):
        top_results = sorted(zip(images, influence_values), key=lambda z: z[1], reverse=True)
        f, axs = plt.subplots(ncols=5, nrows=5)
        for i, (img, v) in enumerate(top_results[:25]):
            axs[i % 5, i // 5].imshow(img)
            axs[i % 5, i // 5].set_title(f'inf = {v}')
            axs[i % 5, i // 5].set_xticks([])
            axs[i % 5, i // 5].set_yticks([])
        plt.show()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(tf.image.convert_image_dtype(x_train, tf.float32), axis=-1)
    x_test = tf.expand_dims(tf.image.convert_image_dtype(x_test, tf.float32), axis=-1)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, 2, "same", activation='swish'),
        tf.keras.layers.Conv2D(32, 3, 2, "same", activation='swish'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(train_set.shuffle(1024).batch(128), epochs=10, validation_data=test_set.batch(128))
    rps_l2 = RepresenterPointL2(model, train_set.batch(64), lambda_regularization=0.1)
    inf_values = rps_l2.compute_influence_values(test_set.batch(64))
    plot_top25_images(x_test, inf_values)
