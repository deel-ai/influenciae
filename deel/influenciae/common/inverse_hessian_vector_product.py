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
from typing import cast

from .backend import BaseBackend
from .model_wrappers import BaseInfluenceModel, InfluenceModel
from .kfac_factors import LayerParameterMap, KroneckerFactors, EKFACFactors

from ..types import Optional, Union, Tuple, List, Callable, Any
from ..utils.conjugate_gradients import conjugate_gradients_solve


class InverseHessianVectorProduct(ABC):
    """
    An interface for classes that perform hessian-vector products.

    Parameters
    ----------
    model
       A model following the InfluenceModel interface whose weights we wish to use for the calculation of
       these (inverse)-hessian-vector products.
    train_dataset
       A batched dataset containing the training dataset's point we wish to employ for the estimation of
       the hessian matrix.
    """
    def __init__(self, model: BaseInfluenceModel, train_dataset: Optional[Any]):
        self.model = model
        self.train_set = train_dataset
        self.backend: BaseBackend = model.backend

        if train_dataset is not None:
            self.cardinality = self.backend.get_dataset_cardinality(train_dataset)

    @abstractmethod
    def _compute_ihvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
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

    def compute_ihvp(self, group: Any, use_gradient: bool = True) -> Any:
        """
        Computes the inverse-hessian-vector product of a group of points.

        Parameters
        ----------
        group
            A dataset containing the group of points of which we wish to compute the
            inverse-hessian-vector product.
        use_gradient
            A boolean indicating whether the IHVP is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        ihvp
            A tensor containing one rank-1 tensor per input point
        """
        self.backend.assert_batched_dataset(group)

        ihvp_dataset = self.backend.map_dataset(
            group,
            lambda *single_batch: self._compute_ihvp_single_batch(single_batch, use_gradient)
        )

        return ihvp_dataset

    @abstractmethod
    def _compute_hvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
        """
        Computes the hessian-vector product of a group of points.

        Parameters
        ----------
        group_batch
            A dataset containing the group of points of which we wish to compute the
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

    def compute_hvp(self, group: Any, use_gradient: bool = True) -> Any:
        """
        Computes the hessian-vector product of a group of points.

        Parameters
        ----------
        group
            A dataset containing the group of points of which we wish to compute the
            hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        self.backend.assert_batched_dataset(group)

        hvp_ds = self.backend.map_dataset(
            group,
            lambda *single_batch: self._compute_hvp_single_batch(single_batch, use_gradient)
        )

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
        The model implementing the InfluenceModel interface.
    train_dataset
        The dataset, already batched and containing only the samples we wish to use for
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
            train_dataset: Optional[Any] = None,
            train_hessian: Optional[Any] = None,
    ):
        super().__init__(model, train_dataset)
        if train_dataset is not None:
            self.inv_hessian = self._compute_inv_hessian(self.train_set)
            self.hessian = None
        elif train_hessian is not None:
            self.hessian = train_hessian
            self.inv_hessian = self.backend.pinv(train_hessian)
        else:
            raise ArgumentError(None, "Either train_dataset or train_hessian can be set to None, but not both")

    def _compute_inv_hessian(self, dataset: Any) -> Any:
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
            A dataset containing the whole or part of the training dataset for the
            computation of the inverse of the mean hessian matrix.

        Returns
        ----------
        inv_hessian
            A tensor with the resulting inverse hessian matrix
        """
        hessian = self.backend.compute_hessian(
            self.model.model,
            self.model.weights,
            self.model.loss_function,
            dataset,
            self.model.nb_params
        )
        return self.backend.pinv(hessian)

    def _compute_ihvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
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
            grads = self.backend.reshape(
                self.model.batch_jacobian_tensor(group_batch),
                (-1, self.model.nb_params)
            )
        else:
            grads = self.backend.reshape(group_batch[0], (-1, self.model.nb_params))

        inv_hess_dtype = self.backend.get_dtype(self.inv_hessian)
        grads_cast = self.backend.cast(grads, inv_hess_dtype)
        ihvp = self.backend.matmul(self.inv_hessian, self.backend.transpose(grads_cast))
        return ihvp

    def _compute_hvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
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
            grads = self.backend.reshape(
                self.model.batch_jacobian_tensor(group_batch),
                (-1, self.model.nb_params)
            )
        else:
            grads = self.backend.reshape(group_batch[0], (-1, self.model.nb_params))

        hvp = self.backend.matmul(self.hessian, self.backend.transpose(grads))
        return hvp

    def compute_hvp(self, group: Any, use_gradient: bool = True) -> Any:
        """
        Computes the hessian-vector product of a group of points provided in the form of a tuple
        of tensors by computing the hessian matrix and performing the product operation.

        Parameters
        ----------
        group
            A dataset containing the points of which we wish to compute the hessian-vector product.
        use_gradient
            A boolean indicating whether the hvp is with the gradients wrt to the loss of the
            points in group or with these vectors instead.

        Returns
        -------
        hvp
            A tensor containing one rank-1 tensor per input point
        """
        if self.hessian is None:
            self.hessian = self.backend.pinv(self.inv_hessian)
        return super().compute_hvp(group, use_gradient)


class ForwardOverBackwardHVP:
    """
    A class for efficiently computing Hessian-vector products using forward-over-backward
    auto-differentiation.
    This module is used for the approximate IHVP calculations (CGD and LISSA).

    Parameters
    ----------
    model
        A model following the InfluenceModel interface.
    train_dataset
        A (batched) dataset with the data-points that will be used for the hessian.
    weights
        The target weights on which to calculate the HVP.
    stochastic_hvp
        Whether to use a stochastic approximation by sampling a subset of batches.
    hvp_steps_per_iter
        Number of batches to use when stochastic_hvp is enabled.
    hvp_batch_size
        Optional batch size to rebatch the train dataset for the HVP operator.
    """
    def __init__(
            self,
            model: BaseInfluenceModel,
            train_dataset: Any,
            weights: Optional[List[Any]] = None,
            stochastic_hvp: bool = False,
            hvp_steps_per_iter: int = 1,
            hvp_batch_size: Optional[int] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.backend: BaseBackend = model.backend
        self.cardinality = self.backend.get_dataset_cardinality(train_dataset)

        if weights is None:
            self.weights = model.weights
        else:
            self.weights = weights

        self._weight_shapes = [self.backend.tensor_shape(w) for w in self.weights]
        self._weight_slices = []
        start = 0
        for shape in self._weight_shapes:
            size = 1
            for dim in shape:
                size *= int(dim)
            self._weight_slices.append((start, start + size, shape))
            start += size

        self.stochastic_hvp = stochastic_hvp
        self.hvp_steps_per_iter = max(1, int(hvp_steps_per_iter))
        self.hvp_batch_size = hvp_batch_size
        self._stochastic_dataset = None
        if stochastic_hvp:
            stochastic_dataset = self.train_dataset
            if hvp_batch_size is not None:
                stochastic_dataset = self.backend.unbatch_dataset(stochastic_dataset)
                stochastic_dataset = self.backend.batch_dataset(stochastic_dataset, hvp_batch_size)
            self._stochastic_dataset = stochastic_dataset

    def _reshape_vector(self, grads: Any) -> List[Any]:
        """
        Reshapes the gradient vector to the right shape for being input into the HVP computation.

        Parameters
        ----------
        grads
            A tensor with the computed gradients.
        Returns
        -------
        grads_reshape
            A list with the gradients in the right shape.
        """
        grads_reshape = []
        for start, end, shape in self._weight_slices:
            g = grads[start:end]
            grads_reshape.append(self.backend.reshape(g, shape))
        return grads_reshape

    def _sub_call(
            self,
            x: List[Any],
            feature_maps_hessian_current: Any,
            y_hessian_current: Any
    ) -> Any:
        """
        Performs the hessian-vector product for a batch of feature maps.

        Parameters
        ----------
        x
            The gradient vector (reshaped to weight shapes) to be multiplied by the hessian matrix.
        feature_maps_hessian_current
            The current batch of feature maps for the hessian calculation.
        y_hessian_current
            The labels corresponding to the current feature maps.

        Returns
        -------
        hessian_vector_product
            A tensor containing the summed hessian-vector product for the batch.
        """
        hvp = self.backend.compute_hvp_batch(
            self.model.model,
            self.weights,
            self.model.loss_function,
            x,
            feature_maps_hessian_current,
            y_hessian_current
        )

        return hvp

    def __call__(self, x_initial: Any) -> Any:
        """
        Computes the mean hessian-vector product for a given feature map over a set of points.

        Parameters
        ----------
        x_initial
            The vector (or matrix of column vectors) over which this product will be computed

        Returns
        -------
        hessian_vector_product
            Tensor with the hessian-vector product. If x_initial has multiple columns, the result
            is stacked on the last axis.
        """
        if self.backend.tensor_ndim(x_initial) == 1:
            x_matrix = self.backend.expand_dims(x_initial, axis=-1)
        else:
            x_matrix = x_initial

        dataset = self.train_dataset
        if self.stochastic_hvp:
            dataset = self._stochastic_dataset
            dataset = self.backend.take_dataset(dataset, self.hvp_steps_per_iter)

        rhs_list = self.backend.transpose(x_matrix)

        def rhs_hvp(*inputs):
            if len(inputs) == 1:
                rhs_vec, features_block, labels_block = inputs[0]
            else:
                rhs_vec, features_block, labels_block = inputs
            rhs_weights = self._reshape_vector(rhs_vec)
            return self._sub_call(rhs_weights, features_block, labels_block)

        hvp_init = self.backend.transpose(self.backend.zeros_like(rhs_list))
        nb_hessian = 0
        hessian_vector_product = hvp_init

        for batch in dataset:
            features_block, labels_block = batch[0], batch[1]
            rhs_count = self.backend.get_batch_size(rhs_list)
            features_tiled = self.backend.repeat(
                self.backend.expand_dims(features_block, axis=0),
                repeats=rhs_count,
                axis=0
            )
            labels_tiled = self.backend.repeat(
                self.backend.expand_dims(labels_block, axis=0),
                repeats=rhs_count,
                axis=0
            )
            hvp_batch = self.backend.map_fn(
                fn=rhs_hvp,
                elems=(rhs_list, features_tiled, labels_tiled),
                output_signature=self.backend.get_dtype(rhs_list)
            )
            hvp_batch = self.backend.transpose(hvp_batch)
            hessian_vector_product = hessian_vector_product + hvp_batch
            nb_hessian += self.backend.get_batch_size(features_block)

        hessian_vector_product = hessian_vector_product / self.backend.cast(
            nb_hessian,
            self.backend.get_dtype(hessian_vector_product)
        )

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
        The model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full model graph must be provided for the computation of
        the different feature maps.
    stochastic_hvp
        Whether to use a stochastic approximation by sampling a subset of batches.
    hvp_steps_per_iter
        Number of batches to use when stochastic_hvp is enabled.
    hvp_batch_size
        Optional batch size to rebatch the train dataset for the HVP operator.
    """
    def __init__(
            self,
            iterative_function: Callable,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: Any,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Any] = None,
            stochastic_hvp: bool = False,
            hvp_steps_per_iter: int = 1,
            hvp_batch_size: Optional[int] = None,
    ):
        super().__init__(model, train_dataset)
        self.n_opt_iters = 100 if n_opt_iters is None else int(n_opt_iters)
        self._batch_shape_tensor: Optional[Tuple[int, ...]] = None
        self.extractor_layer = extractor_layer
        extractor_layer_idx = self._resolve_extractor_layer_idx(model.model, extractor_layer)

        if feature_extractor is None:
            assert self.backend.is_sequential_model(model.model), \
                "Model must be Sequential if feature_extractor is not provided"
            layers = self.backend.get_layers(model.model)
            self.feature_extractor = self.backend.create_sequential_from_layers(layers[:extractor_layer_idx])
        else:
            self.feature_extractor = feature_extractor

        self.train_set = self._compute_feature_map_dataset(self.train_set)  # extract the train set's features

        # Create model that predicts based on the extracted feature maps
        layers = self.backend.get_layers(model.model)
        self.model = BaseInfluenceModel(
            self.backend.create_sequential_from_layers(layers[extractor_layer_idx:]),
            weights_to_watch=model.weights,
            loss_function=model.loss_function,
            weights_processed=True
        )
        self.weights = self.model.weights
        self.hessian_vector_product = ForwardOverBackwardHVP(
            self.model,
            self.train_set,
            self.weights,
            stochastic_hvp=stochastic_hvp,
            hvp_steps_per_iter=hvp_steps_per_iter,
            hvp_batch_size=hvp_batch_size,
        )
        self.iterative_function = iterative_function

    def _resolve_extractor_layer_idx(self, full_model: Any, extractor_layer: Union[int, str]) -> int:
        """Resolve a layer name/index to a concrete integer index."""
        if isinstance(extractor_layer, str):
            layer_idx, _ = self.backend.find_layer_by_name(full_model, extractor_layer)
            return layer_idx
        return extractor_layer

    def batch_shape_tensor(self):
        """
        Return the batch shape of a tensor
        """
        return self._batch_shape_tensor

    def _compute_feature_map_dataset(self, dataset: Any) -> Any:
        """
        Extracts the feature maps for an entire dataset and creates a dataset associating them with
        their corresponding labels.
        Parameters
        ----------
        dataset
            The dataset whose feature maps we wish to extract using the model's first layers
        Returns
        -------
        feature_map_dataset
            A dataset with the pairs (feature_maps, labels), batched using the same batch_size as the one provided
            as input
        """
        feature_map_dataset = self.backend.map_dataset(
            dataset,
            lambda x_batch, y: (self.backend.forward(self.feature_extractor, x_batch), y)
        )
        feature_map_dataset = self.backend.cache_dataset(feature_map_dataset)

        if self._batch_shape_tensor is None:
            # Get shape from first batch
            for batch in feature_map_dataset:
                self._batch_shape_tensor = self.backend.tensor_shape(batch[0])
                break

        return feature_map_dataset

    def _compute_ihvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
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
            feature_maps = self.backend.forward(self.feature_extractor, group_batch[0])
            grads = self.model.batch_jacobian_tensor((feature_maps, *group_batch[1:]))
        else:
            grads = self.backend.reshape(group_batch[0], (-1, self.model.nb_params))

        # Compute the IHVP for each pair feature map-label
        rhs = self.backend.transpose(grads)
        ihvp_list = self.iterative_function(self.hessian_vector_product, rhs, self.n_opt_iters)
        return ihvp_list

    def _compute_hvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
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
            feature_maps = self.backend.forward(self.feature_extractor, group_batch[0])
            grads = self.model.batch_jacobian_tensor((feature_maps, *group_batch[1:]))
        else:
            grads = self.backend.reshape(group_batch[0], (-1, self.model.nb_params))

        # Compute the HVP for each pair features map - label
        rhs = self.backend.transpose(grads)
        hvp_list = self.hessian_vector_product(rhs)
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
        The model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full model graph must be provided for the computation of
        the different feature maps.
    stochastic_hvp
        Whether to use a stochastic approximation by sampling a subset of batches.
    hvp_steps_per_iter
        Number of batches to use when stochastic_hvp is enabled.
    hvp_batch_size
        Optional batch size to rebatch the train dataset for the HVP operator.
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: Any,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Any] = None,
            stochastic_hvp: bool = False,
            hvp_steps_per_iter: int = 1,
            hvp_batch_size: Optional[int] = None,
    ):
        def iterative_function(operator, v, maxiter):  # pylint: disable=W0613
            return conjugate_gradients_solve(operator, v, x0=None, maxiter=self.n_opt_iters)
        super().__init__(
            iterative_function,
            model,
            extractor_layer,
            train_dataset,
            n_opt_iters,
            feature_extractor,
            stochastic_hvp=stochastic_hvp,
            hvp_steps_per_iter=hvp_steps_per_iter,
            hvp_batch_size=hvp_batch_size,
        )


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
        The model implementing the InfluenceModel interface
    extractor_layer
        An integer indicating the position of the last layer of the feature extraction network.
    train_dataset
        The dataset, already batched and containing only the samples we wish to use for the computation of the
        hessian matrix
    n_opt_iters
        The maximum amount of CGD iterations to perform when estimating the inverse-hessian
    feature_extractor
        If the feature extraction model is not Sequential, the full model graph must be provided for the computation of
        the different feature maps.
    damping
        A damping parameter to regularize a nearly singular operator.
    scale
        A rescaling factor to verify the hypothesis of norm(operator / scale) < 1.
    stochastic_hvp
        Whether to use a stochastic approximation by sampling a subset of batches.
    hvp_steps_per_iter
        Number of batches to use when stochastic_hvp is enabled.
    hvp_batch_size
        Optional batch size to rebatch the train dataset for the HVP operator.
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: Any,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Any] = None,
            damping: float = 1e-4,
            scale: float = 10.,
            stochastic_hvp: bool = False,
            hvp_steps_per_iter: int = 1,
            hvp_batch_size: Optional[int] = None,
    ):
        super().__init__(
            self.lissa,
            model,
            extractor_layer,
            train_dataset,
            n_opt_iters,
            feature_extractor,
            stochastic_hvp=stochastic_hvp,
            hvp_steps_per_iter=hvp_steps_per_iter,
            hvp_batch_size=hvp_batch_size,
        )
        self.damping = float(damping)
        self.scale = float(scale)

    def lissa(self, operator: Callable, v: Any, maxiter: int) -> Any:
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
        ihvp = v
        one_minus_damping = 1.0 - self.damping

        for _ in range(maxiter):
            ihvp = v + one_minus_damping * ihvp - operator(ihvp) / self.scale

        ihvp_result = ihvp / self.scale
        return ihvp_result


class KfacIHVP(InverseHessianVectorProduct):
    """
    Inverse-Hessian-Vector Product approximation using Kronecker-Factored
    Approximate Curvature (K-FAC).

    K-FAC approximates the Fisher information matrix block-diagonally per layer,
    factoring each block as a Kronecker product of the input activation covariance
    ``A_l`` and the output gradient covariance ``G_l``.

    For numerical stability and accurate damping, this implementation applies
    damping in the joint Kronecker eigenspace, i.e. it computes
    ``1 / (kron(Lambda_G, Lambda_A) + damping)`` after eigendecomposing ``A`` and ``G``.

    Notes
    -----
    Only ``nn.Linear`` / ``Dense`` and ``nn.Conv2d`` / ``Conv2D`` layers are
    supported.  Parameters from unsupported layers are left untouched (their IHVP
    contribution is set to zero, equivalent to infinite damping).

    Parameters
    ----------
    model
        The model implementing the InfluenceModel interface.
    train_dataset
        A batched dataset for estimating the Kronecker factors.
    damping
        Tikhonov damping added to the Kronecker eigenvalues before inversion.
    target_layers
        Optional list of layer indices to restrict K-FAC to.  ``None`` means
        all supported layers.
    fisher_type
        Fisher variant used for curvature estimation: ``"empirical"`` (default)
        or ``"true"``.
    module_partition_size
        Optional number of supported layers to process per pass while computing
        factors. ``None`` means all supported layers at once.
    offload_activations_to_cpu
        If ``True``, hook-captured activations/gradients are offloaded to CPU
        between capture and factor accumulation.
    data_partition_size
        Optional number of batches per data partition during factor estimation.
    layer_collection
        Layer traversal mode used by K-FAC mapping: ``"top_level"``
        (default) or ``"recursive"``.
    factors_path
        Optional directory path used to cache/load computed factors.
    overwrite_factors
        If ``True``, ignore any existing checkpoint at ``factors_path`` and
        recompute factors before saving.
    """

    def __init__(
        self,
        model: InfluenceModel,
        train_dataset: Any,
        damping: float = 1e-4,
        target_layers: Optional[List[int]] = None,
        fisher_type: str = "empirical",
        module_partition_size: Optional[int] = None,
        offload_activations_to_cpu: bool = False,
        data_partition_size: Optional[int] = None,
        layer_collection: str = "top_level",
        factors_path: Optional[str] = None,
        overwrite_factors: bool = False,
    ):
        super().__init__(model, train_dataset)
        self.damping = damping
        self.fisher_type = fisher_type
        self.factors_path = factors_path
        self.overwrite_factors = overwrite_factors

        # Build layer map and compute factors
        self.layer_map = LayerParameterMap(
            model,
            self.backend,
            target_layers,
            layer_collection=layer_collection,
        )

        checkpoint_path = factors_path
        should_load_factors = (
            checkpoint_path is not None
            and not overwrite_factors
            and KroneckerFactors.checkpoint_exists(checkpoint_path)
        )

        if should_load_factors:
            assert checkpoint_path is not None
            self.factors = KroneckerFactors.load_from_dir(
                model=model,
                backend=self.backend,
                layer_map=self.layer_map,
                path=cast(str, checkpoint_path),
                fisher_type=fisher_type,
                module_partition_size=module_partition_size,
                offload_activations_to_cpu=offload_activations_to_cpu,
                data_partition_size=data_partition_size,
            )
        else:
            self.factors = KroneckerFactors(
                model,
                train_dataset,
                self.backend,
                self.layer_map,
                fisher_type=fisher_type,
                module_partition_size=module_partition_size,
                offload_activations_to_cpu=offload_activations_to_cpu,
                data_partition_size=data_partition_size,
            )
            if checkpoint_path is not None:
                self.factors.save_to_dir(checkpoint_path)

        # Pre-compute eigenspaces and inverse damped Kronecker eigenvalues.
        self.Q_A = {}
        self.Q_G = {}
        self.kron_inv_eigs = {}
        for info in self.layer_map.layers_info:
            idx = info.layer_idx
            if idx not in self.factors.A:
                continue

            a_factor = self._symmetrize_factor(self.factors.A[idx])
            g_factor = self._symmetrize_factor(self.factors.G[idx])

            a_dtype = self.backend.get_dtype(a_factor)
            g_dtype = self.backend.get_dtype(g_factor)

            lam_a, q_a = self.backend.eigh(
                self.backend.cast(a_factor, self.backend.float64_dtype())
            )
            lam_g, q_g = self.backend.eigh(
                self.backend.cast(g_factor, self.backend.float64_dtype())
            )

            lam_a = self.backend.cast(lam_a, a_dtype)
            q_a = self.backend.cast(q_a, a_dtype)
            lam_g = self.backend.cast(lam_g, g_dtype)
            q_g = self.backend.cast(q_g, g_dtype)

            lam_g_for_outer = self.backend.cast(lam_g, self.backend.get_dtype(lam_a))
            kron_eigs = self.backend.reshape(self.backend.outer(lam_g_for_outer, lam_a), (-1,))
            denom = self.backend.maximum(kron_eigs + damping, 1e-12)

            self.Q_A[idx] = q_a
            self.Q_G[idx] = q_g
            self.kron_inv_eigs[idx] = 1.0 / denom

    def _symmetrize_factor(self, factor: Any) -> Any:
        """Return the symmetric part of a factor matrix."""
        return 0.5 * (factor + self.backend.transpose(factor))

    def _compute_ihvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
        """
        Compute K-FAC IHVP for a single batch.

        For each supported layer, extracts the gradient slice, rotates it in the
        Kronecker eigenbasis, applies the inverse damped eigenvalues, rotates
        back, and writes the result to the flat vector.

        Parameters
        ----------
        group_batch
            A tuple with a single batch of tensors.
        use_gradient
            If True, compute gradients from the batch; otherwise treat batch[0]
            as pre-computed gradient vectors.

        Returns
        -------
        ihvp
            Tensor of shape ``(nb_params, batch_size)``.
        """
        if use_gradient:
            grads = self.backend.reshape(
                self.model.batch_jacobian_tensor(group_batch),
                (-1, self.model.nb_params)
            )
        else:
            grads = self.backend.reshape(group_batch[0], (-1, self.model.nb_params))

        batch_size = self.backend.get_batch_size(grads)

        # Start with zeros (unsupported layers contribute nothing)
        result = self.backend.zeros_like(grads)

        for info in self.layer_map.layers_info:
            idx = info.layer_idx
            if idx not in self.kron_inv_eigs:
                continue

            # Extract per-layer gradient slice: (batch, layer_params)
            layer_grads = grads[:, info.flat_start:info.flat_end]

            q_a = self.Q_A[idx]  # (n_in_eff, n_in_eff)
            q_g = self.Q_G[idx]  # (n_out, n_out)
            inv_eigs = self.kron_inv_eigs[idx]  # (n_out * n_in_eff,)

            n_out = int(self.backend.tensor_shape(q_g)[0])
            n_in_eff = int(self.backend.tensor_shape(q_a)[0])

            q_g_t = self.backend.transpose(q_g)
            q_a_t = self.backend.transpose(q_a)

            if self.backend.framework.value == "tensorflow":
                v_mat = self.backend.reshape(layer_grads, (batch_size, n_in_eff, n_out))

                # Rotate: V'_tf = Q_A^T @ V_tf @ Q_G.
                v_rotated = self.backend.matmul(
                    self.backend.matmul(q_a_t, v_mat),
                    q_g,
                )

                # Convert inverse eigenvalues to TF flattening order.
                inv_mat = self.backend.reshape(inv_eigs, (n_out, n_in_eff))
                inv_tf = self.backend.reshape(self.backend.transpose(inv_mat), (-1,))

                v_rot_flat = self.backend.reshape(v_rotated, (batch_size, -1))
                v_scaled = v_rot_flat * self.backend.expand_dims(inv_tf, axis=0)
                v_scaled_mat = self.backend.reshape(v_scaled, (batch_size, n_in_eff, n_out))

                ihvp_layer = self.backend.matmul(
                    self.backend.matmul(q_a, v_scaled_mat),
                    q_g_t,
                )
            else:
                v_mat = self.backend.reshape(layer_grads, (batch_size, n_out, n_in_eff))

                # Rotate: V' = Q_G^T @ V @ Q_A.
                v_rotated = self.backend.matmul(
                    self.backend.matmul(q_g_t, v_mat),
                    q_a,
                )

                v_rot_flat = self.backend.reshape(v_rotated, (batch_size, -1))
                v_scaled = v_rot_flat * self.backend.expand_dims(inv_eigs, axis=0)
                v_scaled_mat = self.backend.reshape(v_scaled, (batch_size, n_out, n_in_eff))

                ihvp_layer = self.backend.matmul(
                    self.backend.matmul(q_g, v_scaled_mat),
                    q_a_t,
                )

            # Flatten back to (batch, layer_params)
            ihvp_flat = self.backend.reshape(ihvp_layer, (batch_size, -1))

            result = self._write_layer_slice(result, ihvp_flat, info.flat_start, info.flat_end)

        return self.backend.transpose(result)  # (nb_params, batch_size)

    def _write_layer_slice(self, result: Any, values: Any, start: int, end: int) -> Any:
        """Write *values* into columns [start:end] of *result*.

        Since in-place assignment may not be supported by all backends, this
        constructs a new tensor.
        """
        backend = self.backend
        nb_params = int(backend.tensor_shape(result)[1])

        parts = []
        if start > 0:
            parts.append(result[:, :start])
        parts.append(values)
        if end < nb_params:
            parts.append(result[:, end:])

        return backend.concat(parts, axis=1)

    def _compute_hvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
        """HVP is not directly supported for K-FAC; raises NotImplementedError."""
        raise NotImplementedError(
            "K-FAC provides an approximate IHVP, not a direct HVP. "
            "Use compute_ihvp() instead."
        )


class EkfacIHVP(InverseHessianVectorProduct):
    """
    Inverse-Hessian-Vector Product approximation using Eigenvalue-corrected
    Kronecker-Factored Approximate Curvature (EK-FAC).

    EK-FAC improves upon K-FAC by:
    1. Eigendecomposing each factor: ``A_l = Q_A Λ_A Q_A^T``,
       ``G_l = Q_G Λ_G Q_G^T``.
    2. Rotating the gradient into the eigenbasis.
    3. Dividing by *corrected* diagonal eigenvalues estimated from training data.
    4. Rotating back.

    This yields a more accurate IHVP at modest extra cost and is the method
    recommended by Grosse et al. (2023) for large-scale influence function
    computation.

    Parameters
    ----------
    model
        The model implementing the InfluenceModel interface.
    train_dataset
        A batched dataset for estimating the factors and corrected eigenvalues.
    damping
        Tikhonov damping added to the corrected eigenvalues before inversion.
    target_layers
        Optional list of layer indices to restrict EK-FAC to.
    n_ekfac_samples
        Number of samples for corrected eigenvalue estimation.  ``None`` means
        use the full dataset.
    fisher_type
        Fisher variant used for curvature estimation: ``"empirical"`` (default)
        or ``"true"``.
    module_partition_size
        Optional number of supported layers to process per pass while computing
        factors. ``None`` means all supported layers at once.
    offload_activations_to_cpu
        If ``True``, hook-captured activations/gradients are offloaded to CPU
        between capture and factor accumulation.
    data_partition_size
        Optional number of batches per data partition during factor estimation.
    layer_collection
        Layer traversal mode used by EK-FAC mapping: ``"top_level"``
        (default) or ``"recursive"``.
    factors_path
        Optional directory path used to cache/load computed factors.
    overwrite_factors
        If ``True``, ignore any existing checkpoint at ``factors_path`` and
        recompute factors before saving.
    """

    def __init__(
        self,
        model: InfluenceModel,
        train_dataset: Any,
        damping: float = 1e-4,
        target_layers: Optional[List[int]] = None,
        n_ekfac_samples: Optional[int] = None,
        fisher_type: str = "empirical",
        module_partition_size: Optional[int] = None,
        offload_activations_to_cpu: bool = False,
        data_partition_size: Optional[int] = None,
        layer_collection: str = "top_level",
        factors_path: Optional[str] = None,
        overwrite_factors: bool = False,
    ):
        super().__init__(model, train_dataset)
        self.damping = damping
        self.fisher_type = fisher_type
        self.factors_path = factors_path
        self.overwrite_factors = overwrite_factors

        self.layer_map = LayerParameterMap(
            model,
            self.backend,
            target_layers,
            layer_collection=layer_collection,
        )

        checkpoint_path = factors_path
        should_load_factors = (
            checkpoint_path is not None
            and not overwrite_factors
            and EKFACFactors.checkpoint_exists(checkpoint_path)
        )

        if should_load_factors:
            assert checkpoint_path is not None
            self.factors = EKFACFactors.load_from_dir(
                model=model,
                backend=self.backend,
                layer_map=self.layer_map,
                path=cast(str, checkpoint_path),
                n_ekfac_samples=n_ekfac_samples,
                fisher_type=fisher_type,
                module_partition_size=module_partition_size,
                offload_activations_to_cpu=offload_activations_to_cpu,
                data_partition_size=data_partition_size,
            )
        else:
            self.factors = EKFACFactors(
                model,
                train_dataset,
                self.backend,
                self.layer_map,
                n_ekfac_samples=n_ekfac_samples,
                fisher_type=fisher_type,
                module_partition_size=module_partition_size,
                offload_activations_to_cpu=offload_activations_to_cpu,
                data_partition_size=data_partition_size,
            )
            if checkpoint_path is not None:
                self.factors.save_to_dir(checkpoint_path)

    def _compute_ihvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
        """
        Compute EK-FAC IHVP for a single batch.

        Steps per layer (PyTorch / TF are handled symmetrically):

        1. Extract per-layer gradient slice and reshape into a matrix whose
           layout matches the framework's native weight storage order.
        2. Rotate into the K-FAC eigenbasis.
        3. Divide element-wise by ``(Lambda_corrected + damping)``.
        4. Rotate back and flatten to the native flat gradient order.

        Parameters
        ----------
        group_batch
            A tuple with a single batch of tensors.
        use_gradient
            If True, compute gradients from the batch.

        Returns
        -------
        ihvp
            Tensor of shape ``(nb_params, batch_size)``.
        """
        if use_gradient:
            grads = self.backend.reshape(
                self.model.batch_jacobian_tensor(group_batch),
                (-1, self.model.nb_params)
            )
        else:
            grads = self.backend.reshape(group_batch[0], (-1, self.model.nb_params))

        batch_size = self.backend.get_batch_size(grads)
        backend = self.backend
        result = backend.zeros_like(grads)

        for info in self.layer_map.layers_info:
            idx = info.layer_idx
            if idx not in self.factors.Q_A or idx not in self.factors.Lambda_corrected:
                continue

            # Extract gradient slice
            layer_grads = grads[:, info.flat_start:info.flat_end]

            # Derive n_out and n_in_eff from the eigenvector matrices (always
            # correctly dimensioned regardless of framework weight conventions).
            q_a = self.factors.Q_A[idx]    # (n_in_eff, n_in_eff)
            q_g = self.factors.Q_G[idx]    # (n_out, n_out)
            lam_corr = self.factors.Lambda_corrected[idx]  # (n_out * n_in_eff,)

            n_out = int(backend.tensor_shape(q_g)[0])
            n_in_eff = int(backend.tensor_shape(q_a)[0])

            q_g_t = backend.transpose(q_g)  # (n_out, n_out) — symmetric, but kept for clarity
            q_a_t = backend.transpose(q_a)  # (n_in_eff, n_in_eff)

            # Framework-aware reshape, rotate, divide, rotate-back, flatten.
            #
            # Lambda_corrected is stored as flatten((n_out, n_in_eff)) in both
            # frameworks (since it's computed from Q_G/Q_A-rotated quantities
            # that are framework-independent).
            lam_damped = lam_corr + self.damping

            if backend.framework.value == "tensorflow":
                # TF flat gradient is in (n_in_eff, n_out) row-major order
                v_mat = backend.reshape(layer_grads, (batch_size, n_in_eff, n_out))

                # Rotate: V'_tf = Q_A^T @ V_tf @ Q_G  ->  (batch, n_in_eff, n_out)
                # This equals (Q_G^T @ V @ Q_A)^T = V'^T
                v_rotated = backend.matmul(backend.matmul(q_a_t, v_mat), q_g)

                # Lambda_corrected is flat (n_out * n_in_eff,) from (n_out, n_in_eff).
                # V'_tf is flat (n_in_eff * n_out,) from (n_in_eff, n_out).
                # Transpose Lambda: reshape to (n_out, n_in_eff), transpose to
                # (n_in_eff, n_out), flatten to (n_in_eff * n_out,).
                lam_mat = backend.reshape(lam_damped, (n_out, n_in_eff))
                lam_tf = backend.reshape(
                    backend.transpose(lam_mat),  # (n_in_eff, n_out) — 2-D transpose is fine
                    (-1,)
                )

                v_rot_flat = backend.reshape(v_rotated, (batch_size, -1))
                v_divided = v_rot_flat / backend.expand_dims(lam_tf, axis=0)

                # Reshape back to (batch, n_in_eff, n_out)
                v_div_mat = backend.reshape(v_divided, (batch_size, n_in_eff, n_out))

                # Rotate back: Q_A @ V''_tf @ Q_G^T  ->  (batch, n_in_eff, n_out)
                ihvp_layer = backend.matmul(backend.matmul(q_a, v_div_mat), q_g_t)
            else:
                # PyTorch flat gradient is in (n_out, n_in_eff) row-major order
                v_mat = backend.reshape(layer_grads, (batch_size, n_out, n_in_eff))

                # Rotate: V' = Q_G^T @ V @ Q_A  ->  (batch, n_out, n_in_eff)
                v_rotated = backend.matmul(backend.matmul(q_g_t, v_mat), q_a)

                # Flatten, divide by lambda, unflatten
                v_rot_flat = backend.reshape(v_rotated, (batch_size, -1))
                v_divided = v_rot_flat / backend.expand_dims(lam_damped, axis=0)
                v_div_mat = backend.reshape(v_divided, (batch_size, n_out, n_in_eff))

                # Rotate back: Q_G @ V'' @ Q_A^T  ->  (batch, n_out, n_in_eff)
                ihvp_layer = backend.matmul(backend.matmul(q_g, v_div_mat), q_a_t)

            # Flatten back to (batch, layer_params) in the framework's native order
            ihvp_flat = backend.reshape(ihvp_layer, (batch_size, -1))

            result = self._write_layer_slice(result, ihvp_flat, info.flat_start, info.flat_end)

        return backend.transpose(result)

    def _write_layer_slice(self, result: Any, values: Any, start: int, end: int) -> Any:
        """Write *values* into columns [start:end] of *result*."""
        backend = self.backend
        nb_params = int(backend.tensor_shape(result)[1])

        parts = []
        if start > 0:
            parts.append(result[:, :start])
        parts.append(values)
        if end < nb_params:
            parts.append(result[:, end:])

        return backend.concat(parts, axis=1)

    def _compute_hvp_single_batch(self, group_batch: Tuple[Any, ...], use_gradient: bool = True) -> Any:
        """HVP is not directly supported for EK-FAC; raises NotImplementedError."""
        raise NotImplementedError(
            "EK-FAC provides an approximate IHVP, not a direct HVP. "
            "Use compute_ihvp() instead."
        )


class IHVPCalculator(Enum):
    """
    Inverse Hessian Vector Product Calculator interface.
    """
    Exact = ExactIHVP
    Cgd = ConjugateGradientDescentIHVP
    Lissa = LissaIHVP
    Kfac = KfacIHVP
    Ekfac = EkfacIHVP

    @staticmethod
    def from_string(ihvp_calculator: str) -> 'IHVPCalculator':
        """
        Restore an IHVPCalculator from string.

        Parameters
        ----------
        ihvp_calculator
            String indicated the method use to compute the inverse hessian vector product,
            e.g 'exact', 'cgd', 'kfac', or 'ekfac'.

        Returns
        -------
        ivhp_calculator
            IHVPCalculator object.
        """
        valid = ['exact', 'cgd', 'lissa', 'kfac', 'ekfac']
        assert ihvp_calculator in valid, (
            f"Only {valid} inverse hessian vector product calculators are supported."
        )
        if ihvp_calculator == 'exact':
            return IHVPCalculator.Exact
        if ihvp_calculator == 'lissa':
            return IHVPCalculator.Lissa
        if ihvp_calculator == 'kfac':
            return IHVPCalculator.Kfac
        if ihvp_calculator == 'ekfac':
            return IHVPCalculator.Ekfac

        return IHVPCalculator.Cgd
