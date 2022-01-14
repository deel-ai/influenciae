from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import Model

from influenciae.common.model_wrappers import InfluenceModel
from influenciae.common.tf_operations import is_dataset_batched

from typing import Optional


class InverseHessianVectorProduct(ABC):
    def __init__(self, model: InfluenceModel, train_dataset: tf.data.Dataset):
        """
        An interface for classes that perform hessian-vector products.

        Parameters
        ----------
        model
            A TF model following the InfluenceModel interface whose weights we wish to use for the calculation of
            these (inverse)-hessian-vector products.
        train_dataset
            A batched TF dataset containing the training dataset's point we wish to employ for the estimation of
            the hessian matrix.
        """
        if not is_dataset_batched(train_dataset):
            raise ValueError('The dataset has not been batched yet. This module requires one that has already been batched.')

        self.model = model
        self.train_set = train_dataset

    @abstractmethod
    def compute_ihvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        raise NotImplementedError()


class ExactIHVP(InverseHessianVectorProduct):
    def __init__(self, model: InfluenceModel, train_dataset: tf.data.Dataset):
        """
        A class that performs the 'exact' computation of the inverse-hessian-vector product.
        As such, it will calculate the hessian of the provided model's loss wrt its parameters,
        compute its Moore-Penrose pseudo-inverse (for numerical stability) and the multiply it
        by the gradients.

        To speed up the algorithm, the hessian matrix is calculated once at instantiation.

        For models with a considerable amount of weights, this implementation may be infeasible
        due to its O(n^2) complexity for the hessian, plus the O(n^3) for its inversion.
        If its memory consumption is too high, you should consider using the CGD approximation.

        Parameters
        ----------
        model
            The TF2.X model implementing the InfluenceModel interface.
        train_dataset
            The TF dataset, already batched and containing only the samples we wish to use for
            the computation of the hessian matrix.
        """
        super(ExactIHVP, self).__init__(model, train_dataset)

        self.inv_hessian = self._compute_inv_hessian(self.train_set)
        self.hessian = None

    def _compute_inv_hessian(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Compute the (pseudo)-inverse of the hessian matrix wrt to the model's parameters using
        backward-mode AD.

        Disclaimer: this implementation trades memory usage for speed, so it can be quite
        memory intensive, especially when dealing with big models.

        Parameters
        ----------
        dataset
            A TF dataset containing the whole or part of the training dataset for the
            computation of the inverse of the mean hessian matrix.

        Returns
        ----------
        inv_hessian
            A tf.Tensor with the resulting inverse hessian matrix
        """
        weights = self.model.weights
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_hess:
            tape_hess.watch(weights)
            grads = self.model.batch_gradient(dataset) if dataset._batch_size == 1 \
                else self.model.batch_jacobian(dataset)

        hess = tf.squeeze(tape_hess.jacobian(grads, weights))
        hessian = tf.reduce_mean(tf.reshape(hess, (-1, int(tf.reduce_prod(weights.shape)), int(tf.reduce_prod(weights.shape)))), axis=0)

        return tf.linalg.pinv(hessian)

    def compute_ihvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points using the exact
        formulation.

        Parameters
        ----------
        group
            A TF dataset containing the group of points of which we wish to compute the
            inverse-hessian-vector product.
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing one rank-1 tensor per input point
        """
        if not is_dataset_batched(group):
            raise ValueError('The dataset has not been batched yet. This module requires one that has already been batched.')

        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian(group), (-1, self.inv_hessian.shape[0]))
            ihvp = tf.matmul(self.inv_hessian, grads, transpose_b=True)
        else:
            ihvp = tf.concat([tf.matmul(self.inv_hessian, vector, transpose_b=True) for vector in group], axis=0)

        return ihvp

    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points using the exact formulation

        Args:
        Parameters
        ----------
        group
            A TF dataset containing the group of points of which we wish to compute the
            hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        if not is_dataset_batched(group):
            raise ValueError('The dataset has not been batched yet. This module requires one that has already been batched.')

        if self.hessian is None:
            self.hessian = tf.linalg.pinv(self.inv_hessian)

        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian(group), (-1, self.inv_hessian.shape[0]))
            hvp = tf.matmul(self.hessian, grads, transpose_b=True)
        else:
            hvp = tf.concat([tf.matmul(self.hessian, vector, transpose_b=True) for vector in group], axis=0)

        return hvp


class ConjugateGradientDescentIHVP(InverseHessianVectorProduct):  # TODO(agus) finish this implementation
    def __init__(
            self,
            model: InfluenceModel,
            train_dataset: tf.data.Dataset,
            n_cgd_iters: Optional[int] = 100
    ):
        """
        A class that approximately computes inverse-hessian-vector products leveraging forward-over-backward
        automatic differentiation and Conjugate Gradient Descent to estimate the product directly, without needing to
        calculate the hessian matrix or invert it.

        It is ideal for models containing a considerable amount of parameters. It does however trade memory for
        speed, as the calculations for estimating the inverse hessian operator are repeated for each sample.

        Attributes
        ----------
        model: InfluenceModel
            The TF2.X model implementing the InfluenceModel interface
        train_dataset: tf.data.Dataset
            The TF dataset, already batched and containing only the samples we wish to use for the computation of the
            hessian matrix
        n_cgd_iters: Optional[int]
            The maximum amount of CGD iterations to perform when estimating the inverse-hessian
        """
        if not is_dataset_batched(train_dataset):
            raise ValueError('The dataset has not been batched yet. This module requires one that has already been batched.')
        super(ConjugateGradientDescentIHVP, self).__init__(model, train_dataset)
        self.n_cgd_iters = n_cgd_iters
        self.feature_extractor = None
        self._batch_shape_tensor = None
        self.train_set = self._compute_feature_map_dataset(self.train_set)  # extract the train set's features
        self.model = InfluenceModel(
            # Model(inputs=model.layers[model.target_layer].input, outputs=model.layers[-1].output),
            tf.keras.Sequential(model.layers[model.target_layer:]),  # TODO make it more generic with Model(in, out)
            target_layer=0,
            loss_function=model.loss_function
        )  # model that predicts based on the extracted feature maps
        self.weights = self.model.weights
        self.n_hessian = train_dataset.cardinality().numpy() * train_dataset._batch_size
        if self.n_hessian == tf.data.INFINITE_CARDINALITY:
            raise ValueError("The training dataset is infinite. Please make sure that this is not the case.")
        if self.n_hessian == tf.data.UNKNOWN_CARDINALITY:
            raise ValueError("Impossible to compute the amount of data-points in the training dataset.")

    @property
    def is_positive_definite(self):
        return True

    @property
    def is_self_adjoint(self):
        return True

    def batch_shape_tensor(self):
        return self._batch_shape_tensor

    def _compute_feature_map_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Extracts the feature maps for an entire dataset and creates a TF dataset associating them with
        their corresponding labels.

        Parameters
        ----------
        dataset: tf.data.Dataset
            The TF dataset whose feature maps we wish to extract using the model's first layers

        Returns
        -------
        feature_map_dataset: tf.data.Dataset
            A TF dataset with the pairs (feature_maps, labels), batched using the same batch_size as the one provided
            as input
        """
        if self.feature_extractor is None:
            self.feature_extractor = Model(inputs=self.model.layers[0].input,
                                           outputs=self.model.layers[self.model.target_layer - 1].output)
        feature_maps = tf.concat([self.feature_extractor(x_batch) for x_batch, _ in dataset], axis=0)
        feature_maps = tf.squeeze(feature_maps, axis=0) if feature_maps.shape[0] == 1 else feature_maps
        feature_map_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(feature_maps),
                                                   dataset.unbatch().map(lambda x, y: y))).batch(dataset._batch_size)

        if self._batch_shape_tensor is None:
            self._batch_shape_tensor = feature_maps.shape[1:]

        return feature_map_dataset

    def compute_ihvp(self, group: tf.data.Dataset) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points approximately using
        the Conjugate Gradient Descent formulation.

        Parameters
        ----------
        group
            A TF dataset containing the group of points of which we wish to compute the
            inverse-hessian-vector product.

        Returns
        -------
        ihvp
            A tensor containing one rank-1 tensor per input point.
        """
        if not is_dataset_batched(group):
            raise ValueError('The dataset has not been batched yet. This module requires one that has already been batched.')

        # Transform the dataset into a set of feature maps-labels
        feature_maps = self._compute_feature_map_dataset(group)

        # Compute the IHVP for each pair feature map-label
        ihvp_list = []
        ihvp_shape = None
        grads = self.model.batch_jacobian(feature_maps)
        for x_influence_grad, label in zip(grads, feature_maps.map(lambda x, y: y).unbatch()):
            x_influence_grads = tf.reshape(x_influence_grad, (tf.shape(x_influence_grad)[0], -1))
            # @todo watch for broadcast dynamic shape in the cgd function
            _, hessian_vect_product, _, _, _ = tf.linalg.experimental.conjugate_gradient(self, x_influence_grads,
                                                                                         preconditioner=None, x=None,
                                                                                         tol=1e-05,
                                                                                         max_iter=self.n_cgd_iters)
            ihvp_list.append(hessian_vect_product)
            if ihvp_shape is None:
                ihvp_shape = hessian_vect_product.shape
        ihvp_list = tf.reshape(tf.stack(ihvp_list), (-1, ihvp_shape))

        return ihvp_list

    @tf.function
    def __sub_call(
            self,
            x: tf.Tensor,
            feature_maps_hessian_current: tf.Tensor,
            y_hessian_current: tf.Tensor
    ) -> tf.Tensor:
        """
        Perform the hessian-vector product for a single feature map

        Parameters
        ----------
        x
            The gradient vector to be multiplied by the hessian matrix.
        feature_maps_hessian_current
            The current feature map for the hessian calculation.
        y_hessian_current
            The label corresponding to the current feature map.

        Returns
        -------
        hessian_product
            A tf.Tensor containing the result of the hessian-vector product for a given input point and one pair
            feature map-label.
        """
        with tf.autodiff.ForwardAccumulator(
                self.weights,
                # The "vector" in Hessian-vector product.
                x) as acc:
            backward = self.model.batch_gradient(
                tf.data.Dataset.from_tensor_slices((feature_maps_hessian_current, y_hessian_current)).batch(1)
            )
        hessian_product = acc.jvp(backward)

        weight = tf.cast(tf.shape(feature_maps_hessian_current)[0], dtype=tf.float32) / self.n_hessian
        hessian_product = hessian_product * weight

        return hessian_product

    def __call__(self, x_initial: tf.Tensor) -> tf.Tensor:
        """
        Computes the mean hessian-vector product for a given feature map over a set of points

        Parameters
        ----------
        x_initial
            The point of the dataset over which this product will be computed

        Returns
        -------
            Tensor with the hessian-vector product
        """
        x = tf.reshape(x_initial, tf.shape(self.weights))

        hessian_product = tf.zeros_like(x)
        iter = 0
        for features_block, labels_block in self.train_set:
            for f, l in zip(tf.unstack(features_block), tf.unstack(labels_block)):
                iter += 1
                hessian_product_current = self.__sub_call(x, tf.expand_dims(f, axis=0),
                                                          tf.expand_dims(l, axis=0))
                hessian_product += hessian_product_current
        hessian_product = tf.reshape(hessian_product, tf.shape(x_initial))

        return hessian_product