# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing a technique based on the representer point theorem for kernels,
but using a local jacobian expansion, as per
https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf

Supports both TensorFlow and PyTorch models through the backend abstraction layer.
"""
from typing import Optional, Union

from .base_representer_point import BaseRepresenterPoint
from ..common import InfluenceModel, InverseHessianVectorProduct, InverseHessianVectorProductFactory
from ..types import DatasetLike, Model, Tensor
from ..utils.model_surgery import compute_lje_alpha, perturb_head_single_sgd_step


class RepresenterPointLJE(BaseRepresenterPoint):
    """
    Representer Point Selection via Local Jacobian Expansion for Post-hoc Classifier Explanation of Deep Neural
    Networks and Ensemble Models
    https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf

    Supports both TensorFlow and PyTorch models through the backend abstraction layer.

    Disclaimer: This technique requires the last layer of the model to be a Dense/Linear layer with no bias.

    Parameters
    ----------
    influence_model
        The model implementing the InfluenceModel interface (TensorFlow or PyTorch).
    dataset
        A batched dataset with the points with which the model was trained.
    ihvp_calculator_factory
        An InverseHessianVectorProductFactory for creating new instances of the InverseHessianVectorProduct
        class.
    n_samples_for_hessian
        An integer for the amount of samples from the training dataset that will be used for the computation of the
        hessian matrix.
        If None, the whole dataset will be used.
    target_layer
        Either a string or an integer identifying the layer on which to compute the influence-related quantities.
    shuffle_buffer_size
        An integer with the buffer size for the training set's shuffle operation (TensorFlow only).
    epsilon
        An epsilon value to prevent division by zero.
    """

    def __init__(
            self,
            influence_model: InfluenceModel,
            dataset: DatasetLike,
            ihvp_calculator_factory: InverseHessianVectorProductFactory,
            n_samples_for_hessian: Optional[int] = None,
            target_layer: Union[int, str] = -1,
            shuffle_buffer_size: int = 10000,
            epsilon: float = 1e-5
    ):
        super().__init__(influence_model.model, dataset, influence_model.loss_function, target_layer)
        self.epsilon = epsilon

        self.perturbed_head: Model
        self.perturbed_head, dataset_to_estimate_hessian = perturb_head_single_sgd_step(
            backend=self.backend,
            original_head=self.original_head,
            feature_extractor=self.feature_extractor,
            dataset=dataset,
            loss_function=influence_model.loss_function,
            n_samples_for_hessian=n_samples_for_hessian,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        perturbed_model = InfluenceModel(
            self.perturbed_head,
            start_layer=None,
            loss_function=influence_model.loss_function,
        )
        self.ihvp_calculator: InverseHessianVectorProduct = ihvp_calculator_factory.build(
            perturbed_model,
            dataset_to_estimate_hessian,
        )

    def _compute_alpha(self, z_batch: Tensor, y_batch: Tensor) -> Tensor:
        """
        Compute the alpha vector for the Local Jacobian Expansion approximation.

        Parameters
        ----------
        z_batch
            A tensor with the perturbed model's predictions.
        y_batch
            A tensor with the ground truth labels.

        Returns
        -------
        alpha
            A tensor with the alpha vector for the Local Jacobian Expansion approximation.
        """
        return compute_lje_alpha(
            backend=self.backend,
            perturbed_head=self.perturbed_head,
            ihvp_calculator=self.ihvp_calculator,
            loss_function=self.loss_function,
            z_batch=z_batch,
            y_batch=y_batch,
            epsilon=self.epsilon,
        )
