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

from .backend import BaseBackend
from .model_wrappers import BaseInfluenceModel, InfluenceModel

from ..types import Optional, Union, Tuple, List, Callable, Any
from ..utils import conjugate_gradients_solve


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
    def __init__(self, model: InfluenceModel, train_dataset: Optional[Any]):
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
    """
    def __init__(
            self,
            model: BaseInfluenceModel,
            train_dataset: Any,
            weights: Optional[List[Any]] = None
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
            The point of the dataset over which this product will be computed

        Returns
        -------
        hessian_vector_product
            Tensor with the hessian-vector product
        """
        x = self._reshape_vector(x_initial)

        hvp_init = self.backend.zeros((self.model.nb_params,), dtype=self.backend.get_dtype(x_initial))
        nb_hessian = 0
        hessian_vector_product = hvp_init

        for batch in self.train_dataset:
            features_block, labels_block = batch[0], batch[1]
            hvp_current = self._sub_call(x, features_block, labels_block)
            hessian_vector_product = hessian_vector_product + hvp_current
            nb_hessian += self.backend.get_batch_size(features_block)

        hessian_vector_product = self.backend.reshape(
            hessian_vector_product,
            (self.model.nb_params, 1)
        ) / self.backend.cast(nb_hessian, self.backend.get_dtype(hessian_vector_product))


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
    """
    def __init__(
            self,
            iterative_function: Callable,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: Any,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Any] = None,
    ):
        super().__init__(model, train_dataset)
        self.n_opt_iters = n_opt_iters
        self._batch_shape_tensor: Optional[Tuple[int, ...]] = None
        self.extractor_layer = extractor_layer

        if feature_extractor is None:
            assert self.backend.is_sequential_model(model.model), \
                "Model must be Sequential if feature_extractor is not provided"
            layers = self.backend.get_layers(model.model)
            self.feature_extractor = self.backend.create_sequential_from_layers(layers[:self.extractor_layer])
        else:
            self.feature_extractor = feature_extractor

        self.train_set = self._compute_feature_map_dataset(self.train_set)  # extract the train set's features

        # Create model that predicts based on the extracted feature maps
        layers = self.backend.get_layers(model.model)
        self.model = BaseInfluenceModel(
            self.backend.create_sequential_from_layers(layers[extractor_layer:]),
            weights_to_watch=model.weights,
            loss_function=model.loss_function,
            weights_processed=True
        )
        self.weights = self.model.weights
        self.hessian_vector_product = ForwardOverBackwardHVP(self.model, self.train_set, self.weights)
        self.iterative_function = iterative_function

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
        def cgd_func(single_grad):
            inv_hessian_vect_product = self.iterative_function(
                self.hessian_vector_product,
                self.backend.expand_dims(single_grad, axis=-1),
                self.n_opt_iters
            )
            return inv_hessian_vect_product

        ihvp_list = self.backend.map_fn(fn=cgd_func, elems=grads)

        shape = self.backend.tensor_shape(ihvp_list)
        if shape[-1] != 1:
            ihvp_list = self.backend.transpose(ihvp_list)
        else:
            ihvp_list = self.backend.transpose(self.backend.squeeze(ihvp_list, axis=-1))

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
        def single_hvp(single_grad):
            hvp = self.hessian_vector_product(self.backend.expand_dims(single_grad, axis=-1))
            return hvp

        hvp_list = self.backend.map_fn(fn=single_hvp, elems=grads)

        shape = self.backend.tensor_shape(hvp_list)
        if shape[-1] != 1:
            hvp_list = self.backend.transpose(hvp_list)
        else:
            hvp_list = self.backend.transpose(self.backend.squeeze(hvp_list, axis=-1))

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
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: Any,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Any] = None,
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
    """
    def __init__(
            self,
            model: InfluenceModel,
            extractor_layer: Union[int, str],
            train_dataset: Any,
            n_opt_iters: Optional[int] = 100,
            feature_extractor: Optional[Any] = None,
            damping: float = 1e-4,
            scale: float = 10.
    ):
        super().__init__(self.lissa, model, extractor_layer, train_dataset, n_opt_iters, feature_extractor)
        self.damping = self.backend.convert_to_tensor(damping, dtype=self.backend.float32_dtype())
        self.scale = self.backend.convert_to_tensor(scale, dtype=self.backend.float32_dtype())

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
        one_minus_damping = self.backend.cast(
            self.backend.constant(1.0) - self.damping,
            self.backend.float32_dtype()
        )

        for _ in range(maxiter):
            ihvp = v + one_minus_damping * ihvp - operator(ihvp) / self.scale

        ihvp_result = ihvp / self.scale
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
