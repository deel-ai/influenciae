# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from deel.influenciae.types import Optional, Union, List, Tuple
from deel.influenciae.common import InfluenceModel
import tensorflow as tf


class TracIn:
    """
    A class implementing an influence score based on Tracin method
    https://arxiv.org/pdf/2002.08484.pdf

    The method evaluates the influence values of samples over a set of training points

    Parameters
    ----------
    models
        A list of TF2.X models implementing the InfluenceModel interface at different step (epoch)
        of the training
    learning_rates: Learning rate or list of learning used during the training.
        If learning_rates is a list, it shall be have the same size of the models argument
    """
    def __init__(self, models: List[InfluenceModel], learning_rates: Union[float, List[float]]):
        self.models = models

        if isinstance(learning_rates, List):
            assert len(models) == len(learning_rates)
            self.learning_rates = learning_rates
        else:
            self.learning_rates = [learning_rates for _ in range(len(models))]

    def compute_influence_values_from_tensor(self, batch_train: Tuple[tf.Tensor, tf.Tensor],
                                             batch_to_evaluate: Optional[
                                                 Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.Tensor:
        """
        Compute the influence score between training samples and a list of samples to evaluate
        batch_train and batch_to_evaluate shall have the same shape
        when batch_to_evaluate is None the influence will be evaluate on batch_train

        Parameters
        ----------
            batch_train
                The training samples
            batch_to_evaluate
                The samples to evaluate

        Returns
        -------
            the influence score by sample to evaluate
        """
        if batch_to_evaluate is None:
            batch_to_evaluate = batch_train
        else:
            assert tf.reduce_all(tf.equal(tf.shape(batch_train[0]), tf.shape(batch_to_evaluate[0])))
            assert tf.reduce_all(tf.equal(tf.shape(batch_train[1]), tf.shape(batch_to_evaluate[1])))

        values = []
        for model, lr in zip(self.models, self.learning_rates):
            g_train = model.batch_jacobian_tensor(*batch_train)
            g_evaluate = model.batch_jacobian_tensor(*batch_to_evaluate)
            v = tf.reduce_sum(g_train * g_evaluate, axis=1, keepdims=True)
            v = v * lr
            values.append(v)
        values = tf.concat(values, axis=1)
        values = tf.reduce_sum(values, axis=1, keepdims=True)
        return values

    def compute_influence_values(
            self,
            dataset_train: Union[tf.data.Dataset, Tuple[tf.Tensor, tf.Tensor]],
            dataset_to_evaluate: Optional[Union[tf.data.Dataset, Tuple[tf.Tensor, tf.Tensor]]] = None
    ) -> tf.Tensor:
        """
        Compute the influence score between training samples and a list of samples to evaluate
        If dataset_train and dataset_evaluate are datasets, they shall have the same batch_size (last batch of the
        dataset included)
        If one of them is a tensor, the tensor shall have the same shape than the batch of the dataset
        If both of then are tensors, the tensors shall have the same shape

        Parameters
        ----------
            dataset_train
                the training dataset
            dataset_to_evaluate
                the samples to evaluate. If None, dataset_to_evaluate will be equal to dataset_train

        Returns
        -------
            the influence value for each sample of the dataset to evaluate regarding the training dataset
        """
        if dataset_to_evaluate is None:
            dataset_to_evaluate = dataset_train

        if isinstance(dataset_train, tf.data.Dataset):
            values = []
            if isinstance(dataset_to_evaluate, tf.data.Dataset):
                for batch_train, batch_to_evaluate in zip(dataset_train, dataset_to_evaluate):
                    v = self.compute_influence_values_from_tensor(batch_train, batch_to_evaluate)
                    values.append(v)
            else:
                for batch_train in dataset_train:
                    v = self.compute_influence_values_from_tensor(batch_train, dataset_to_evaluate)
                    values.append(v)
            values = tf.concat(values, axis=0)
        else:
            if isinstance(dataset_to_evaluate, tf.data.Dataset):
                values = []
                for batch_to_evaluate in dataset_to_evaluate:
                    v = self.compute_influence_values_from_tensor(dataset_train, batch_to_evaluate)
                    values.append(v)
                values = tf.concat(values, axis=0)
            else:
                values = self.compute_influence_values_from_tensor(dataset_train, dataset_to_evaluate)

        return values
