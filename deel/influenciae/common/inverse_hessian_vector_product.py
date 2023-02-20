# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing the different approaches for computing and approximating the
inverse-hessian-vector product: an essential block in the computation of influence
functions.
"""
from abc import ABC, abstractmethod
from enum import Enum
from argparse import ArgumentError

import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=E0611
from tensorflow.keras.models import Sequential  # pylint: disable=E0611

from .model_wrappers import BaseInfluenceModel, InfluenceModel

from ..types import Optional, Union, Tuple, List, Callable
from ..utils import assert_batched_dataset, conjugate_gradients_solve, map_to_device


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
            self.cardinality = train_dataset.cardinality()

        self.model = model
        self.train_set = train_dataset


    @abstractmethod
    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points provided in the form of
        a batch of tensors.

        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the inverse-hessian-vector product
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing a rank-1 tensor per input point.
        """
        raise NotImplementedError

    def compute_ihvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.data.Dataset:
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
        assert_batched_dataset(group)

        ihvp_dataset = group.map(lambda *single_batch: self._compute_ihvp_single_batch(single_batch, use_gradient))

        return ihvp_dataset

    @abstractmethod
    def _compute_hvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points.

        Parameters
        ----------
        group_batch
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

    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.data.Dataset:
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
        assert_batched_dataset(group)

        hvp_ds = group.map(lambda *single_batch: self._compute_hvp_single_batch(single_batch, use_gradient))

        return hvp_ds


class ExactIHVP(InverseHessianVectorProduct):
    """
    A class that performs the 'exact' computation of the inverse-hessian-vector product.
    As such, it will calculate the hessian of the provided model's loss wrt its parameters,
    compute its Moore-Penrose pseudo-inverse (for numerical stability) and the multiply it
    by the gradients.

    Notes
    -----
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
    """
    def __init__(
            self,
            model: InfluenceModel,
            train_dataset: Optional[tf.data.Dataset] = None,
            train_hessian: Optional[tf.Tensor] = None,
    ):
        super().__init__(model, train_dataset)
        if train_dataset is not None:
            nb_batch = tf.cast(train_dataset.cardinality(), dtype=tf.int32)
            self.inv_hessian = self._compute_inv_hessian(self.train_set, nb_batch)
            self.hessian = None
        elif train_hessian is not None:
            self.hessian = train_hessian
            self.inv_hessian = tf.linalg.pinv(train_hessian)
        else:
            raise ArgumentError("Either train_dataset or train_hessian can be set to None, but not both")

    @tf.function
    def _compute_inv_hessian(self, dataset: tf.data.Dataset, nb_batch: tf.int32) -> tf.Tensor:
        """
        Compute the (pseudo)-inverse of the hessian matrix wrt to the model's parameters using
        backward-mode AD.

        Disclaimer
        ----------
        This implementation trades memory usage for speed, so it can be quite
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

        hess = tf.zeros((self.model.nb_params, self.model.nb_params), dtype=dataset.element_spec[0].dtype)
        nb_elt = tf.constant(0, dtype=tf.int32)
        nb_batch_saw = tf.constant(0, dtype=tf.int32)
        iter_ds = iter(dataset)

        def hessian_sum(nb_elt, nb_batch_saw, hess):
            batch = next(iter_ds)
            nb_batch_saw += tf.constant(1, dtype=tf.int32)
            curr_nb_elt = tf.shape(batch[0])[0]
            nb_elt += curr_nb_elt
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_hess:
                tape_hess.watch(weights)
                grads = self.model.batch_jacobian_tensor(batch) # pylint: disable=W0212

            curr_hess = tape_hess.jacobian(
                    grads, weights
                    )

            curr_hess = [tf.reshape(h, shape=(len(grads), self.model.nb_params, -1)) for h in curr_hess]
            curr_hess = tf.concat(curr_hess, axis=-1)
            curr_hess = tf.reshape(curr_hess, shape=(len(grads), self.model.nb_params, -1))
            curr_hess = tf.reduce_sum(curr_hess, axis=0)
            hess += tf.cast(curr_hess, dtype=hess.dtype)

            return nb_elt, nb_batch_saw, hess

        nb_elt, _, hess = tf.while_loop(
            cond=lambda __, nb_batch_saw, _: nb_batch_saw < nb_batch,
            body=hessian_sum,
            loop_vars=[nb_elt, nb_batch_saw, hess]
        )

        hessian = hess / tf.cast(nb_elt, dtype=hess.dtype)

        return tf.linalg.pinv(hessian)

    @tf.function
    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points provided in the form of
        a batch of tensors by computing the exact inverse hessian matrix and performing the product
        operation.

        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the inverse-hessian-vector product.
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing a rank-1 tensor per input point.
        """
        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian_tensor(group_batch), (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        ihvp = tf.matmul(self.inv_hessian, tf.cast(grads, dtype=self.inv_hessian.dtype), transpose_b=True)
        return ihvp

    @tf.function
    def _compute_hvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points provided in the form of a tuple
        of tensors by computing the hessian matrix and performing the product operation.

        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        if use_gradient:
            grads = tf.reshape(self.model.batch_jacobian_tensor(group_batch), (-1, self.model.nb_params))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        hvp = tf.matmul(self.hessian, grads, transpose_b=True)

        return hvp

    def compute_hvp(self, group: tf.data.Dataset, use_gradient: bool = True) -> tf.data.Dataset:
        """
        Computes the hessian-vector product of a group of points provided in the form of a tuple
        of tensors by computing the hessian matrix and performing the product operation.

        Parameters
        ----------
        group
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        if self.hessian is None:
            self.hessian = tf.linalg.pinv(self.inv_hessian)
        return super().compute_hvp(group, use_gradient)


class ForwardOverBackwardHVP:
    """
    A class for efficiently computing Hessian-vector products using forward-over-backward
    auto-differentiation.
    This module is used for the approximate IHVP calculations (CGD and LISSA).

    Parameters
    ----------
    model
        A TF model following the InfluenceModel interface.
    train_dataset
        A (batched) TF dataset with the data-points that will be used for the hessian.
    weights
        The target weights on which to calculate the HVP.
    """
    def __init__(
            self,
            model: BaseInfluenceModel,
            train_dataset: tf.data.Dataset,
            weights: Optional[List[tf.Tensor]] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.cardinality = train_dataset.cardinality()

        if weights is None:
            self.weights = model.weights
        else:
            self.weights = weights

    @staticmethod
    def _reshape_vector(grads: tf.Tensor, weights: tf.Tensor) -> List[tf.Tensor]:
        """
        Reshapes the gradient vector to the right shape for being input into the HVP computation.

        Parameters
        ----------
        grads
            A tensor with the computed gradients.
        weights
            A tensor with the target weights.

        Returns
        -------
        grads_reshape
            A list with the gradients in the right shape.
        """
        grads_reshape = []
        index = 0
        for w in weights:
            shape = tf.shape(w)
            size = tf.reduce_prod(shape)
            g = grads[index:(index + size)]
            grads_reshape.append(tf.reshape(g, shape))
            index += size
        return grads_reshape

    @tf.function
    def _sub_call(
            self,
            x: tf.Tensor,
            feature_maps_hessian_current: tf.Tensor,
            y_hessian_current: tf.Tensor
    ) -> tf.Tensor:
        """
        Performs the hessian-vector product for a single feature map.

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
        hessian_vector_product
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

        weight = tf.cast(tf.shape(feature_maps_hessian_current)[0], dtype=hvp.dtype)

        hvp = hvp * weight

        return hvp

    def __call__(self, x_initial: tf.Tensor) -> tf.Tensor:
        """
        Computes the mean hessian-vector product for a given feature map over a set of points.

        Parameters
        ----------
        x_initial
            The point of the dataset over which this product will be computed

        Returns
        -------
        hessian_vector_product
            Tensor with the hessian-vector product
        """
        x = self._reshape_vector(x_initial, self.model.weights)

        hvp_init = tf.zeros((self.model.nb_params,), dtype=x_initial.dtype)
        dataset_iterator = iter(self.train_dataset)

        def body_func(i, hessian_vector_product, nb_hessian):
            features_block, labels_block = next(dataset_iterator)

            def batched_hvp(elt):
                f, l = elt
                hessian_product_current = self._sub_call(x, tf.expand_dims(f, axis=0), tf.expand_dims(l, axis=0))

                return hessian_product_current

            hessian_vector_product_inner = tf.reduce_sum(
                tf.map_fn(fn=batched_hvp, elems=[features_block, labels_block], fn_output_signature=x_initial.dtype),
                axis=0
            )

            hessian_vector_product += hessian_vector_product_inner
            return i + 1, hessian_vector_product, nb_hessian + tf.shape(features_block)[0]

        _, hessian_vector_product, nb_hessian = tf.while_loop(
            cond=lambda i, _, __: i < self.cardinality,
            body=body_func,
            loop_vars=[tf.constant(0, dtype=tf.int64), hvp_init, tf.constant(0, dtype=tf.int32)]
        )

        hessian_vector_product = tf.reshape(hessian_vector_product, (self.model.nb_params, 1)) / \
                                 tf.cast(nb_hessian, dtype=hessian_vector_product.dtype)

        return hessian_vector_product


class IterativeIHVP(InverseHessianVectorProduct):
    """
    A class that approximately computes inverse-hessian-vector products leveraging forward-over-backward
    automatic differentiation with an iterative procedure to estimate the product directly, without needing to
    calculate the hessian matrix or invert it.
    Notes
    -----
    It is ideal for models containing a considerable amount of parameters. It does however trade memory for
    speed, as the calculations for estimating the inverse hessian operator are repeated for each sample.
    Parameters
    ----------
    iterative_function
        The procedure to compute the inverse hessian product operation
    model
        The TF2.X model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The TF dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full TF graph must be provided for the computation of
        the different feature maps.
    """
    def __init__(
            self,
            iterative_function,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: tf.data.Dataset,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Model] = None,
    ):
        super().__init__(model, train_dataset)
        self.n_opt_iters = n_opt_iters
        self._batch_shape_tensor = None
        self.extractor_layer = extractor_layer

        if feature_extractor is None:
            assert isinstance(model.model, Sequential)
            self.feature_extractor = tf.keras.Sequential(self.model.layers[:self.extractor_layer])
        else:
            assert isinstance(feature_extractor, Model)
            self.feature_extractor = feature_extractor

        self.train_set = self._compute_feature_map_dataset(self.train_set)  # extract the train set's features
        self.model = BaseInfluenceModel(
            tf.keras.Sequential(model.layers[extractor_layer:]),
            weights_to_watch=model.weights,
            loss_function=model.loss_function,
            weights_processed=True
        )  # model that predicts based on the extracted feature maps
        self.weights = self.model.weights
        self.hessian_vector_product = ForwardOverBackwardHVP(self.model, self.train_set, self.weights)
        self.iterative_function = iterative_function

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
        dataset
            The TF dataset whose feature maps we wish to extract using the model's first layers
        Returns
        -------
        feature_map_dataset
            A TF dataset with the pairs (feature_maps, labels), batched using the same batch_size as the one provided
            as input
        """
        feature_map_dataset = map_to_device(dataset, lambda x_batch, y: (self.feature_extractor(x_batch), y)).cache()

        if self._batch_shape_tensor is None:
            self._batch_shape_tensor = tf.shape(next(iter(feature_map_dataset))[0])

        return feature_map_dataset

    def _compute_ihvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the inverse-hessian-vector product of a group of points provided in the form of
        a batch of tensors by inverting the hessian-vector product that is calculated through
        forward-over-backward AD.
        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the inverse-hessian-vector product.
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.
        Returns
        -------
        ihvp
            A tensor containing a rank-1 tensor per input point.
        """
        # Transform the dataset into a set of feature maps-labels
        if use_gradient:
            feature_maps = self.feature_extractor(group_batch[0])
            grads = self.model.batch_jacobian_tensor((feature_maps, *group_batch[1:]))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        # Compute the IHVP for each pair feature map-label
        def cgd_func(single_grad):
            inv_hessian_vect_product = self.iterative_function(self.hessian_vector_product,
                                                               tf.expand_dims(single_grad, axis=-1),
                                                               self.n_opt_iters)
            return inv_hessian_vect_product

        ihvp_list = tf.map_fn(fn=cgd_func, elems=grads)

        ihvp_list = tf.transpose(ihvp_list) if ihvp_list.shape[-1] != 1 \
            else tf.transpose(tf.squeeze(ihvp_list, axis=-1))

        return ihvp_list

    def _compute_hvp_single_batch(self, group_batch: Tuple[tf.Tensor, ...], use_gradient: bool = True) -> tf.Tensor:
        """
        Computes the hessian-vector product of a group of points provided in the form of a tuple
        of tensors through forward-over-backward AD.
        Parameters
        ----------
        group_batch
            A Tuple with a single batch of tensors containing the points of which we wish to
            compute the hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.
        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        # Transform the dataset into a set of feature maps-labels
        if use_gradient:
            feature_maps = self.feature_extractor(group_batch[0])
            grads = self.model.batch_jacobian_tensor((feature_maps, *group_batch[1:]))
        else:
            grads = tf.reshape(group_batch[0], (-1, self.model.nb_params))

        # Compute the HVP for each pair features map - label
        def single_hvp(single_grad):
            hvp = self.hessian_vector_product(tf.expand_dims(single_grad, axis=-1))
            return hvp

        hvp_list = tf.map_fn(fn=single_hvp, elems=grads)

        hvp_list = tf.transpose(hvp_list) if hvp_list.shape[-1] != 1 else tf.transpose(tf.squeeze(hvp_list, axis=-1))

        return hvp_list


class ConjugateGradientDescentIHVP(IterativeIHVP):
    """
    A class that approximately computes inverse-hessian-vector products leveraging forward-over-backward
    automatic differentiation and Conjugate Gradient Descent to estimate the product directly, without needing to
    calculate the hessian matrix or invert it.
    Notes
    -----
    It is ideal for models containing a considerable amount of parameters. It does however trade memory for
    speed, as the calculations for estimating the inverse hessian operator are repeated for each sample.
    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The TF dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full TF graph must be provided for the computation of
        the different feature maps.
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: tf.data.Dataset,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Model] = None,
    ):
        def iterative_function(operator, v, maxiter):  # pylint: disable=W0613
            return conjugate_gradients_solve(operator, v, x0=None, maxiter=self.n_opt_iters)
        super().__init__(iterative_function, model, extractor_layer, train_dataset, n_opt_iters, feature_extractor)


class LissaIHVP(IterativeIHVP):
    """
    A class that approximately computes inverse-hessian-vector products leveraging forward-over-backward
    automatic differentiation and lissa [https://arxiv.org/pdf/1703.04730.pdf , https://arxiv.org/pdf/1602.03943.pdf]
    to estimate the product directly, without needing to calculate the hessian matrix or invert it.

    [A^{-1}v]_{j+1} = v + (I - (A + d * I))[A^{-1}v]_j * v

    Notes
    -----
    It is ideal for models containing a considerable amount of parameters. It does however trade memory for
    speed, as the calculations for estimating the inverse hessian operator are repeated for each sample.

    Parameters
    ----------
    model
        The TF2.X model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The TF dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full TF graph must be provided for the computation of
        the different feature maps.
    damping
        A damping parameter to regularize a nearly singular operator.
    scale
        A rescaling factor to verify the hypothesis of norm(operator / scale) < 1.
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: tf.data.Dataset,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Model] = None,
            damping: float = 1e-4,
            scale: float = 10.
    ):
        super().__init__(self.lissa, model, extractor_layer, train_dataset, n_opt_iters, feature_extractor)
        self.damping = tf.convert_to_tensor(damping, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

    def lissa(self, operator: Callable, v: tf.Tensor, maxiter: int):
        """
        Performs the Linear time Stochastic Second-order Algorithm (LiSSA) optimization procedure to solve
        a problem of the shape Ax = b by iterating as follows:

            [A^{-1}v]_{j+1} = v + (I - (A + d * I))[A^{-1}v]_j * v

        Parameters
        ----------
        operator
            The operator that transforms the input vector v into Av
        v
            The vector v of the problem
        maxiter
            Number of iterations of the algorithm

        Returns
        -------
        ihvp_result
            A tensor containing inv(A)v
        """
        _, ihvp_result = tf.while_loop(lambda index, ihvp: index < maxiter,
                                       lambda index, ihvp: (index + 1,
                                                            v + tf.cast(1. - self.damping, dtype=tf.float32) * ihvp -
                                                            operator(ihvp) / self.scale),
                                       [tf.constant(0, dtype=tf.int32), v])
        ihvp_result /= self.scale

        return ihvp_result


class IHVPCalculator(Enum):
    """
    Inverse Hessian Vector Product Calculator interface.
    """
    Exact = ExactIHVP
    Cgd = ConjugateGradientDescentIHVP
    Lissa = LissaIHVP

    @staticmethod
    def from_string(ihvp_calculator: str) -> 'IHVPCalculator':
        """
        Restore an IHVPCalculator from string.

        Parameters
        ----------
        ihvp_calculator
            String indicated the method use to compute the inverse hessian vector product,
            e.g 'exact' or 'cgd'.

        Returns
        -------
        ivhp_calculator
            IHVPCalculator object.
        """
        assert ihvp_calculator in ['exact', 'cgd', 'lissa'], "Only 'exact', 'lissa' and 'cgd' inverse hessian " \
                                                             "vector product calculators are supported."
        if ihvp_calculator == 'exact':
            return IHVPCalculator.Exact
        if ihvp_calculator == 'lissa':
            return IHVPCalculator.Lissa

        return IHVPCalculator.Cgd
