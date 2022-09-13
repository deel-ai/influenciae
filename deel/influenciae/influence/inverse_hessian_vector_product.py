# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Inverse Hessian Vector Product (ihvp) module
"""

from abc import ABC, abstractmethod
from argparse import ArgumentError

import tensorflow as tf
from tensorflow.keras import Model

from ..common import BaseInfluenceModel, InfluenceModel, conjugate_gradients_solve

from ..types import Optional, Union, Tuple
from ..common import assert_batched_dataset, dataset_size


class InverseHessianVectorProduct(ABC):
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
    def __init__(self, model: InfluenceModel, train_dataset: Optional[tf.data.Dataset]):
        if train_dataset is not None:
            assert_batched_dataset(train_dataset)

        self.model = model
        self.train_set = train_dataset

    @abstractmethod
    def compute_ihvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points.

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
        raise NotImplementedError()

    @abstractmethod
    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points.

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
        raise NotImplementedError()


class ExactIHVP(InverseHessianVectorProduct):
    """
    A class that performs the 'exact' computation of the inverse-hessian-vector product.
    As such, it will calculate the hessian of the provided model's loss wrt its parameters,
    compute its Moore-Penrose pseudo-inverse (for numerical stability) and the multiply it
    by the gradients.

    To speed up the algorithm, the hessian matrix is calculated once at instantiation.

    For models with a considerable amount of weights, this implementation may be infeasible
    due to its O(n^2) complexity for the hessian, plus the O(n^3) for its inversion.
    If its memory consumption is too high, you should consider using the CGD approximation,
    or computing the hessian separately and initializing the ExactIHVP with this hessian while
    setting train_dataset to None. To expect it to work the hessian should be computed for
    the training_set.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface.
    train_dataset
        The TF dataset, already batched and containing only the samples we wish to use for
        the computation of the hessian matrix. Either train_hessian or train_dataset should
        not be None but not both.
    train_hessian
        The estimated hessian matrix of the model's loss wrt its parameters computed with
        the samples used for the model's training. Either hessian or train_dataset should
        not be None but not both.
    parallel_iter
        The number of parallel iterations when computing the jacobian. Default is None
    """
    def __init__(self,
        model: InfluenceModel,
        train_dataset: Optional[tf.data.Dataset] = None,
        train_hessian: Optional[tf.Tensor] = None,
        parallel_iter: Optional[int] = None
    ):
        super().__init__(model, train_dataset)
        if train_dataset is not None:
            self.parallel_iter = parallel_iter
            self.inv_hessian = self._compute_inv_hessian(self.train_set)
            self.hessian = None
        elif train_hessian is not None:
            self.hessian = train_hessian
            self.inv_hessian = tf.linalg.pinv(train_hessian)
        else:
            raise ArgumentError("Either train_dataset or train_hessian can be \
                set to None. Not both")

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

        if self.parallel_iter is None:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_hess:
                tape_hess.watch(weights)
                grads = self.model.batch_gradient(dataset) if dataset._batch_size == 1 \
                    else self.model.batch_jacobian(dataset) # pylint: disable=W0212

            hess = tape_hess.jacobian(grads, weights)
        else:
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_hess:
                tape_hess.watch(weights)
                grads = self.model.batch_gradient(dataset) if dataset._batch_size == 1 \
                    else self.model.batch_jacobian(dataset) # pylint: disable=W0212
                    #TODO: I think batch_jacobian is quite fine in all cases

            hess = tape_hess.jacobian(
                    grads, weights,
                    parallel_iterations=self.parallel_iter,
                    experimental_use_pfor=False
                    )

        hess = [tf.reshape(h, shape=(len(grads), self.model.nb_params, -1)) for h in hess]
        hessian = tf.concat(hess, axis=-1)
        hessian = tf.reduce_mean(hessian, axis=0)

        return tf.linalg.pinv(hessian)

    def compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, tf.Tensor], use_gradient: bool = True) -> tf.Tensor:
        """
        """
        group_data, group_targets = group_batch
        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian_tensor(group_data, group_targets), (-1, self.model.nb_params))
            ihvp = tf.matmul(self.inv_hessian, grads, transpose_b=True)
        else:
            if len(group_batch.element_spec.shape) == 2:
                ihvp = tf.concat(
                    [tf.matmul(self.inv_hessian, tf.reshape(vector, (-1, self.model.nb_params)), transpose_b=True)
                     for vector in group_batch],
                    axis=0
                )
            else:
                ihvp = tf.concat(
                    [tf.matmul(self.inv_hessian, vector, transpose_b=True)
                     for vector in group_batch],
                    axis=0
                )
        return ihvp

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
        assert_batched_dataset(group)

        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian(group), (-1, self.model.nb_params))
            ihvp = tf.matmul(self.inv_hessian, grads, transpose_b=True)
        else:
            if len(group.element_spec.shape) == 2:
                ihvp = tf.concat(
                    [tf.matmul(self.inv_hessian, tf.reshape(vector, (-1, self.model.nb_params)), transpose_b=True)
                     for vector in group],
                    axis=0
                )
            else:
                ihvp = tf.concat(
                    [tf.matmul(self.inv_hessian, vector, transpose_b=True)
                     for vector in group],
                    axis=0
                )

        return ihvp

    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points using the exact formulation.


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
        assert_batched_dataset(group)

        if self.hessian is None:
            self.hessian = tf.linalg.pinv(self.inv_hessian)

        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian(group), (-1, self.inv_hessian.shape[0]))
            hvp = tf.matmul(self.hessian, grads, transpose_b=True)
        else:
            if len(group.element_spec.shape) == 2:
                hvp = tf.concat(
                    [tf.matmul(self.hessian, tf.reshape(vector, (-1, self.inv_hessian.shape[0])), transpose_b=True)
                     for vector in group],
                    axis=0
                )
            else:
                hvp = tf.concat(
                    [tf.matmul(self.hessian, vector, transpose_b=True) for vector in group],
                    axis=0
                )

        return hvp


class ConjugateGradientDescentIHVP(InverseHessianVectorProduct):
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
    extractor_layer:
        TODO: Def and transform
    train_dataset: tf.data.Dataset
        The TF dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_cgd_iters: Optional[int]
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: tf.data.Dataset,
            n_cgd_iters: Optional[int] = 100
    ):
        super().__init__(model, train_dataset)
        self.n_cgd_iters = n_cgd_iters
        self.feature_extractor = None
        self._batch_shape_tensor = None
        self.n_hessian = dataset_size(train_dataset)
        self.extractor_layer = extractor_layer
        self.train_set = self._compute_feature_map_dataset(self.train_set)  # extract the train set's features
        self.model = BaseInfluenceModel(
            # Model(inputs=model.layers[model.target_layer].input, outputs=model.layers[-1].output),
            tf.keras.Sequential(model.layers[extractor_layer:]),  # TODO make it more generic with Model(in, out)
            weights_to_watch=model.weights,
            loss_function=model.loss_function,
            weights_processed=True
        )  # model that predicts based on the extracted feature maps
        self.weights = self.model.weights

    def batch_shape_tensor(self):
        """
        Return the batch shape of a tensor
        """
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
                                           outputs=self.model.layers[self.extractor_layer - 1].output)
        if isinstance(dataset.element_spec, tuple):
            feature_maps = tf.concat([self.feature_extractor(x_batch) for x_batch, _ in dataset], axis=0)
        else:
            feature_maps = tf.concat([self.feature_extractor(x_batch) for x_batch in dataset], axis=0)
        feature_maps = tf.squeeze(feature_maps, axis=0) if feature_maps.shape[0] == 1 else feature_maps
        feature_map_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(feature_maps),
                                                   dataset.unbatch().map(lambda _, y: y))).batch(dataset._batch_size) # pylint: disable=W0212

        if self._batch_shape_tensor is None:
            self._batch_shape_tensor = feature_maps.shape[1:]

        return feature_map_dataset

    def compute_ihvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points approximately using
        the Conjugate Gradient Descent formulation.

        Parameters
        ----------
        group: tf.data.Dataset
            A TF dataset containing the group of points of which we wish to compute the
            inverse-hessian-vector product.
        use_gradient: bool
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing one rank-1 tensor per input point.
        """
        assert_batched_dataset(group)

        # Transform the dataset into a set of feature maps-labels
        if use_gradient:
            feature_maps = self._compute_feature_map_dataset(group)
            grads = self.model.batch_jacobian(feature_maps)
        else:
            grads = group.map(lambda x, y: x).batch(1) if isinstance(group.element_spec, tuple) else group

        # Compute the IHVP for each pair feature map-label
        ihvp_list = []
        ihvp_shape = None
        for x_influence_grad in grads:
            # Squeeze when the grads have some weird shape
            if len(x_influence_grad.shape) > 1:
                x_influence_grad = tf.squeeze(x_influence_grad)
            x_influence_grads = tf.reshape(x_influence_grad, (tf.shape(x_influence_grad)[0], -1))
            inv_hessian_vect_product = conjugate_gradients_solve(self, x_influence_grads, x0=None,
                                                                 maxiter=self.n_cgd_iters)
            ihvp_list.append(inv_hessian_vect_product)
            if ihvp_shape is None:
                ihvp_shape = inv_hessian_vect_product.shape
        ihvp_list = tf.stack(ihvp_list, axis=0)
        ihvp_list = tf.transpose(ihvp_list) if ihvp_list.shape[-1] != 1 \
            else tf.transpose(tf.squeeze(ihvp_list, axis=-1))

        return ihvp_list

    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points approximately using forward-over-backward
        auto-differentiation.

        Parameters
        ----------
        group: tf.data.Dataset
            A TF dataset containing the group of points of which we wish to compute the
            inverse-hessian-vector product.
        use_gradient: bool
            A boolean indicating whether the HVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing one rank-1 tensor per input point.
        """
        assert_batched_dataset(group)

        # Transform the dataset into a set of feature maps-labels
        if use_gradient:
            feature_maps = self._compute_feature_map_dataset(group)
            grads = self.model.batch_jacobian(feature_maps)
        else:
            grads = group.map(lambda x, _: x).batch(1) if isinstance(group.element_spec, tuple) else group

        # Compute the IHVP for each pair feature map-label
        hvp_list = []
        hvp_shape = None
        for x_influence_grad in grads:
            if use_gradient:
                x_influence_grads = tf.reshape(x_influence_grad, (tf.shape(x_influence_grad)[0], -1))
            else:
                x_influence_grads = x_influence_grad[0] #TODO: not pretty

            hessian_vect_product = self(x_influence_grads)
            hvp_list.append(hessian_vect_product)
            if hvp_shape is None:
                hvp_shape = hessian_vect_product.shape
        hvp_list = tf.stack(hvp_list, axis=0) if len(hvp_list) != 1 else hvp_list[0]
        hvp_list = tf.transpose(hvp_list) if hvp_list.shape[-1] != 1 else tf.transpose(tf.squeeze(hvp_list, axis=-1))

        return hvp_list

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
        x: tf.Tensor
            The gradient vector to be multiplied by the hessian matrix.
        feature_maps_hessian_current: tf.Tensor
            The current feature map for the hessian calculation.
        y_hessian_current: tf.Tensor
            The label corresponding to the current feature map.

        Returns
        -------
        hessian_vector_product: tf.Tensor
            A tf.Tensor containing the result of the hessian-vector product for a given input point and one pair
            feature map-label.
        """
        with tf.autodiff.ForwardAccumulator(
                self.weights,
                # The "vector" in Hessian-vector product.
                x) as acc:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.weights)
                loss = self.model.loss_function(y_hessian_current, self.model(feature_maps_hessian_current))
            backward = tape.jacobian(loss, self.weights)
        hessian_vector_product = acc.jvp(backward)
        hvp = [tf.reshape(hessian_vp, shape=(-1,)) for hessian_vp in hessian_vector_product]
        hvp = tf.concat(hvp, axis=0)

        weight = tf.cast(tf.shape(feature_maps_hessian_current)[0], dtype=tf.float32) / \
                 tf.cast(self.n_hessian, dtype=tf.float32)

        hvp = hvp * weight

        return hvp

    def __call__(self, x_initial: tf.Tensor) -> tf.Tensor:
        """
        Computes the mean hessian-vector product for a given feature map over a set of points

        Parameters
        ----------
        x_initial: tf.Tensor
            The point of the dataset over which this product will be computed

        Returns
        -------
        hessian_vector_product: tf.Tensor
            Tensor with the hessian-vector product
        """
        cum_size_list_up = tf.cumsum([tf.size(w) for w in self.weights])
        cum_size_list_down = tf.concat([tf.constant([0], dtype=tf.int32), cum_size_list_up[:-1]], axis=0)
        x = [tf.reshape(x_initial[s_down: s_up], tf.shape(w)) for w, s_down, s_up in zip(self.weights, cum_size_list_down, cum_size_list_up)]

        hessian_vector_product = tf.zeros_like(self.model.nb_params)
        for features_block, labels_block in self.train_set:
            for f, l in zip(tf.unstack(features_block), tf.unstack(labels_block)):
                hessian_product_current = self.__sub_call(x, tf.expand_dims(f, axis=0),
                                                          tf.expand_dims(l, axis=0))
                hessian_vector_product += hessian_product_current
        hessian_vector_product = tf.reshape(hessian_vector_product, tf.shape(x_initial))

        return hessian_vector_product
