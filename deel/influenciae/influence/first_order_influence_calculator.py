# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
First order Influence module implementing computations for all the different influence
related quantities: influence vector (delta of the weights after holding out the
sample and the original model's), Cook's distance (or influence values, a measure of
the model's reliance on the specific sample), both for individual points and whole
groups of data-points.

Disclaimer: This implements only a first order approximation of the influence function,
which does not take into account the pairwise interactions of data-points inside groups.
For a more precise (but much more computationally expensive) alternative, please refer
to the SecondOrderInfluenceCalculator module.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_group_influence import BaseGroupInfluenceCalculator

from ..common import InfluenceModel
from ..common import BaseInfluenceCalculator
from ..common import InverseHessianVectorProduct, InverseHessianVectorProductFactory, IHVPCalculator
from ..common.evaluation import EvaluationRepresentationProvider
from ..common.payloads import TrainingPayloadExtractor
from ..common.query_batching import (
    LowRankGradient,
    QueryBatchingConfig,
    make_module_partitions,
    concat_query_representations,
)

from ..types import DatasetLike, Tensor
from ..utils.sorted_dict import BatchSort, ORDER


class FirstOrderInfluenceCalculator(BaseInfluenceCalculator, BaseGroupInfluenceCalculator):
    """
    A class implementing the necessary methods to compute the different influence quantities
    using a first-order approximation. This makes it ideal for individual points and small
    groups of data, as it does so (relatively) efficiently.

    Notes
    -----
    For estimating the influence of large groups of data, please refer to the
    SecondOrderInfluenceCalculator class, which also takes into account the pairwise interactions
    between the points inside these groups.

    The methods currently implemented are available to evaluate one or a group of point(s):
    - Influence function vectors: the weights difference when removing points or groups of points
    - Influence values/Cook's distance: a measure of reliance of the model on the individual
      points or groups of points.

    For individual points, the following paper is used:
    [https://arxiv.org/abs/1703.04730](https://arxiv.org/abs/1703.04730).
    For groups of points, the following paper is used:
    [https://arxiv.org/abs/1905.13289](https://arxiv.org/abs/1905.13289)

    Parameters
    ----------
    model
        The model implementing the InfluenceModel interface (TensorFlow or PyTorch).
    dataset
        A batched dataset containing the training dataset over which we will estimate the
        inverse-hessian-vector product.
    ihvp_calculator
        Either a string containing the IHVP method ('exact' or 'cgd'), an
        IHVPCalculator object, an InverseHessianVectorProductFactory object,
        or an InverseHessianVectorProduct object.
    n_samples_for_hessian
        An integer indicating the amount of samples to take from the provided train dataset.
    shuffle_buffer_size
        An integer indicating the buffer size of the train dataset's shuffle operation -- when
        choosing the amount of samples for the hessian.
    normalize
        Implement "RelatIF: Identifying Explanatory Training Examples via Relative Influence"
        https://arxiv.org/pdf/2003.11630.pdf
        if True, compute the relative influence by normalizing the influence function.
    """

    def __init__(
            self,
            model: InfluenceModel,
            dataset: DatasetLike,
            ihvp_calculator: Union[
                str,
                InverseHessianVectorProduct,
                InverseHessianVectorProductFactory,
                IHVPCalculator,
            ] = 'exact',
            n_samples_for_hessian: Optional[int] = None,
            shuffle_buffer_size: Optional[int] = 10000,
            normalize: bool = False
    ):
        super().__init__(
            model,
            dataset,
            ihvp_calculator,
            n_samples_for_hessian,
            shuffle_buffer_size
        )

        self.normalize: bool = normalize

    def _normalize_if_needed(self, v: Tensor) -> Tensor:
        """
        Normalize the input vector if the normalize property is True. If False, do nothing

        Parameters
        ----------
        v
            The vector to normalize of shape [Features_Space, Batch_Size]

        Returns
        -------
        v
            The normalized vector if the normalize property is True, otherwise the input vector
        """
        if self.normalize:
            v = self._backend.normalize(v, axis=0, keepdims=True)
        return v

    def _compute_influence_vector(self, train_samples: Tuple[Tensor, ...]) -> Tensor:
        """
        Computes the influence vector (i.e. the delta of model's weights after a perturbation on the training
        dataset) for a single batch of training samples.

        Parameters
        ----------
        train_samples
            A tuple with the batch of training samples (with their labels).

        Returns
        -------
        influence_vector
            A tensor with the influence vector for each individual point. Shape will be (batch_size, nb_weights).
        """
        influence_vector = self.ihvp_calculator._compute_ihvp_single_batch(train_samples)  # pylint: disable=W0212
        influence_vector = self._normalize_if_needed(influence_vector)
        influence_vector = self._backend.transpose(influence_vector)
        return influence_vector

    def _preprocess_samples(self, samples: Tuple[Any, ...]) -> Tensor:
        """
        Convert one evaluation batch to per-sample gradients for scoring.

        Parameters
        ----------
        samples
            A single batch of samples to evaluate.

        Returns
        -------
        sample_evaluate_grads
            Tensor of shape ``(batch_size, nb_params)`` containing the
            per-sample Jacobian of the loss with respect to the watched
            weights.
        """
        sample_evaluate_grads = self.model.batch_jacobian_tensor(samples)
        return sample_evaluate_grads

    def _estimate_individual_influence_values_from_batch(
            self,
            train_samples: Tuple[Tensor, ...],
            samples_to_evaluate: Tuple[Tensor, ...]
    ) -> Tensor:
        """
        Estimate the (individual) influence scores of a single batch of samples with respect to
        a batch of samples belonging to the model's training dataset.

        Parameters
        ----------
        train_samples
            A single batch of training samples (and their target values).
        samples_to_evaluate
            A single batch of samples of which we wish to compute the influence of removing the training
            samples.

        Returns
        -------
        A tensor containing the individual influence scores.
        """
        return self._estimate_influence_value_from_influence_vector(
            self._preprocess_samples(samples_to_evaluate),
            self._compute_influence_vector(train_samples)
        )

    def _estimate_influence_value_from_influence_vector(
            self,
            preproc_test_sample: Tensor,
            influence_vector: Tensor
    ) -> Tensor:
        """
        Estimate influence scores from preprocessed query gradients and influence vectors.

        Parameters
        ----------
        preproc_test_sample
            Tensor of shape ``(n_query, nb_params)`` containing the
            preprocessed representation of the query batch.
        influence_vector
            Tensor of shape ``(n_train, nb_params)`` containing one influence
            vector per training sample.

        Returns
        -------
        influence_values
            Tensor of shape ``(n_query, n_train)`` whose ``(i, j)`` entry is
            the influence score of training sample ``j`` on query sample ``i``.
        """
        influence_values = self._backend.matmul(
            preproc_test_sample,
            self._backend.transpose(influence_vector)
        )
        return influence_values

    def _compute_influence_value_from_batch(self, train_samples: Tuple[Tensor, ...]) -> Tensor:
        """
        Computes the influence score (self-influence) for a single batch of training samples.

        Parameters
        ----------
        train_samples
            A tensor with a single batch of training sample.

        Returns
        -------
        influence_values
            The influence score of each sample in the batch train_samples.
        """
        evaluate_vect = self._preprocess_samples(train_samples)
        batched_inf_vect = self.ihvp_calculator._compute_ihvp_single_batch(  # pylint: disable=W0212
            (evaluate_vect,),
            use_gradient=False
        )
        batched_inf_vect = self._normalize_if_needed(batched_inf_vect)
        batched_inf_vect = self._backend.transpose(batched_inf_vect)
        influence_values = self._backend.reduce_sum(
            self._backend.multiply(evaluate_vect, batched_inf_vect), axis=1, keepdims=True)
        return influence_values

    # ------------------------------------------------------------------
    # Query-batched (query-side preconditioning) computation
    # ------------------------------------------------------------------

    def _validate_query_batched_config(
            self,
            config: Optional[QueryBatchingConfig],
    ) -> QueryBatchingConfig:
        """Validate and normalize query-batching configuration."""
        if not self.ihvp_calculator.supports_query_preconditioning:
            raise ValueError(
                f"The IHVP calculator {type(self.ihvp_calculator).__name__} does not "
                "support query-side preconditioning. Use ExactIHVP, KfacIHVP, or EkfacIHVP."
            )
        if config is None:
            config = QueryBatchingConfig()
        config.validate()
        return config

    @staticmethod
    def _as_query_batch(batch: Union[Tensor, Tuple[Any, ...], List[Any]]) -> Tuple[Any, ...]:
        """Normalize dataset elements to a tuple of tensors."""
        if isinstance(batch, (list, tuple)):
            return tuple(batch)
        return (batch,)

    def _slice_batch_rows(self, batch: Tuple[Any, ...], start: int, end: int) -> Tuple[Any, ...]:
        """Slice one contiguous row range from a batched sample tuple."""
        return tuple(tensor[start:end] for tensor in batch)

    def _split_batch_rows(
            self,
            batch: Tuple[Any, ...],
            num_partitions: int,
    ) -> List[Tuple[Any, ...]]:
        """Split a training batch into contiguous row partitions."""
        if num_partitions <= 1:
            return [batch]

        batch_size = batch[0].shape[0]
        if batch_size is None:
            runtime_batch_size = self._backend.get_batch_size(batch[0])
            try:
                batch_size = int(runtime_batch_size)
            except TypeError as exc:
                raise ValueError(
                    "score_data_partitions requires a statically known batch dimension."
                ) from exc

        if batch_size <= 1:
            return [batch]

        partition_size = max(1, (batch_size + num_partitions - 1) // num_partitions)
        return [
            self._slice_batch_rows(batch, start, min(batch_size, start + partition_size))
            for start in range(0, batch_size, partition_size)
        ]

    def _factorize_module_batch(self, module_tensor: Tensor, rank: int) -> Tuple[Tensor, Tensor]:
        """Factorize one per-module query tensor batch into true low-rank factors."""
        backend = self._backend
        batch_size = int(backend.get_batch_size(module_tensor))
        shape = backend.tensor_shape(module_tensor)
        rows = int(shape[1])
        cols = int(shape[2])
        effective_rank = min(rank, rows, cols)

        left_parts = []
        right_parts = []
        for query_idx in range(batch_size):
            matrix = module_tensor[query_idx]
            u, s, vh = backend.svd_lowrank(matrix, effective_rank)
            left_parts.append(u * backend.expand_dims(s, axis=0))
            right_parts.append(vh)

        return backend.stack(left_parts, axis=0), backend.stack(right_parts, axis=0)

    def _compress_gradient_flat(self, grads: Tensor, rank: int) -> LowRankGradient:
        """
        Compress a flat gradient batch using a global low-rank SVD.

        The gradient matrix of shape ``(batch, nb_params)`` is first
        transposed to ``(nb_params, batch)`` so the SVD is over samples.
        The result is stored under module key ``-1`` (global).

        Parameters
        ----------
        grads
            Gradient tensor of shape ``(batch, nb_params)``.
        rank
            Target rank for the truncated SVD.

        Returns
        -------
        compressed
            A :class:`LowRankGradient` with ``is_global=True``.
        """
        backend = self._backend
        # Transpose to (nb_params, batch) for SVD across samples
        grads_t = backend.transpose(grads)  # (nb_params, batch)
        grads_t_shape = backend.tensor_shape(grads_t)
        effective_rank = min(rank, int(grads_t_shape[0]), int(grads_t_shape[1]))
        u, s, vh = backend.svd_lowrank(grads_t, effective_rank)
        # u: (nb_params, rank), s: (rank,), vh: (rank, batch)
        # Represent as left=(nb_params, rank) and right=(rank, batch)
        left = backend.expand_dims(u, axis=0)   # (1, nb_params, rank)
        # Scale left by singular values so dot = left @ right^T reconstructs gradient
        s_diag = backend.expand_dims(backend.expand_dims(s, axis=0), axis=0)  # (1, 1, rank)
        left_scaled = left * s_diag  # (1, nb_params, rank)
        right = backend.expand_dims(vh, axis=0)  # (1, rank, batch)
        return LowRankGradient({-1: (left_scaled, right)}, is_global=True)

    def _compress_gradient_per_module(
            self,
            preconditioned_modules: Dict[int, Tensor],
            rank: int,
    ) -> LowRankGradient:
        """
        Compress per-module preconditioned gradients using low-rank SVD.

        Each module tensor has shape ``(batch, rows, cols)`` and is factorized
        independently per query sample into ``(left, right)`` factors compatible
        with :func:`~.query_batching._einsum_low_rank`.

        Parameters
        ----------
        preconditioned_modules
            Dict from layer index to tensor of shape ``(batch, rows, cols)``.
        rank
            Target rank for the truncated SVD per module.

        Returns
        -------
        compressed
            A :class:`LowRankGradient` with ``is_global=False``.
        """
        module_values: Dict[int, Tuple[Tensor, Tensor]] = {}
        for layer_idx, module_tensor in preconditioned_modules.items():
            module_values[layer_idx] = self._factorize_module_batch(module_tensor, rank)
        return LowRankGradient(module_values, is_global=False)

    def _precondition_query_batch(
            self,
            query_batch: Tuple[Any, ...],
            config: QueryBatchingConfig,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
    ) -> Union[Tensor, LowRankGradient]:
        """
        Compute the IHVP-preconditioned representation of a query batch.

        When ``query_gradient_low_rank`` is set in *config*, the preconditioned
        gradients are further compressed via low-rank SVD.

        Parameters
        ----------
        query_batch
            A tuple of tensors representing one batch of test samples.
        config
            Query batching configuration.

        Returns
        -------
        representation
            Either a full-rank tensor of shape ``(batch, nb_params)`` or a
            :class:`LowRankGradient` instance.
        """
        backend = self._backend
        # Compute per-sample gradients: (batch, nb_params)
        query_grads = backend.reshape(
            self._get_evaluation_representation(query_batch, evaluation_representation_provider),
            (backend.get_batch_size(query_batch[0]), -1)
        )

        # Apply IHVP to query gradients if supported
        if self.ihvp_calculator.supports_query_preconditioning:
            # precondition_gradient returns (nb_params, batch); transpose back
            preconditioned = backend.transpose(
                self.ihvp_calculator.precondition_gradient(query_grads)
            )  # (batch, nb_params)
        else:
            preconditioned = query_grads

        if config.query_gradient_low_rank is None:
            return preconditioned

        # Compress
        rank = config.query_gradient_low_rank
        has_per_module = hasattr(self.ihvp_calculator, "precondition_gradient_per_module")
        if has_per_module and self.ihvp_calculator.supports_query_preconditioning:
            per_module = self.ihvp_calculator.precondition_gradient_per_module(query_grads)  # type: ignore[union-attr]
            if config.query_gradient_svd_dtype is not None:
                per_module = {
                    layer_idx: backend.cast(module_tensor, config.query_gradient_svd_dtype)
                    for layer_idx, module_tensor in per_module.items()
                }
            return self._compress_gradient_per_module(per_module, rank)

        if config.query_gradient_svd_dtype is not None:
            preconditioned = backend.cast(preconditioned, config.query_gradient_svd_dtype)
        return self._compress_gradient_flat(preconditioned, rank)

    def _compute_train_ihvp_norms(self, train_batch: Tuple[Tensor, ...]) -> Tensor:
        """Compute train-side IHVP norms used by RelatIF normalization."""
        train_grads = self._backend.reshape(
            self.model.batch_jacobian_tensor(train_batch),
            (self._backend.get_batch_size(train_batch[0]), -1)
        )
        train_ihvp = self.ihvp_calculator.precondition_gradient(train_grads)
        norms = self._backend.norm(train_ihvp, axis=0)
        norms = self._backend.cast(norms, self._backend.get_dtype(train_ihvp))
        min_norm = self._backend.cast(
            self._backend.constant(1e-12),
            self._backend.get_dtype(norms),
        )
        return self._backend.maximum(norms, min_norm)

    def _apply_train_side_normalization(self, scores: Tensor, train_batch: Tuple[Tensor, ...]) -> Tensor:
        """Apply the original train-side normalization semantics to score columns."""
        norms = self._compute_train_ihvp_norms(train_batch)
        norms = self._backend.cast(norms, self._backend.get_dtype(scores))
        return scores / self._backend.expand_dims(norms, axis=0)

    def _merge_query_representations(
            self,
            query_reps: List[Union[Tensor, LowRankGradient]],
    ) -> Union[Tensor, LowRankGradient]:
        """
        Merge accumulated query representations.

        Per-module low-rank representations concatenate directly. Global
        low-rank representations are converted back to dense query vectors
        before concatenation because their factors are batch-local.
        """
        backend = self._backend
        first = query_reps[0]
        if not isinstance(first, LowRankGradient):
            return backend.concat(list(query_reps), axis=0)

        if first.is_global and len(query_reps) > 1:
            dense_query_reps = []
            for query_rep in query_reps:
                value = query_rep.module_values[-1]
                if not isinstance(value, tuple):
                    raise ValueError("Global LowRankGradient must store a (left, right) tuple.")
                left, right = value
                left_sq = backend.squeeze(left, axis=0)
                right_sq = backend.squeeze(right, axis=0)
                dense_query_reps.append(backend.transpose(backend.matmul(left_sq, right_sq)))
            return backend.concat(dense_query_reps, axis=0)

        return concat_query_representations(query_reps, backend)

    def _compute_raw_scores_for_train_batch(
            self,
            query_rep: Union[Tensor, LowRankGradient],
            train_batch: Tuple[Tensor, ...],
            config: QueryBatchingConfig,
    ) -> Tensor:
        """Compute unnormalized scores for one full training batch."""
        backend = self._backend
        train_grads = backend.reshape(
            self.model.batch_jacobian_tensor(train_batch),
            (backend.get_batch_size(train_batch[0]), -1)
        )

        if isinstance(query_rep, LowRankGradient):
            if query_rep.is_global:
                value = query_rep.module_values[-1]
                if not isinstance(value, tuple):
                    raise ValueError("Global LowRankGradient must store a (left, right) tuple.")
                left, right = value
                left_sq = backend.squeeze(left, axis=0)    # (nb_params, rank)
                right_sq = backend.squeeze(right, axis=0)  # (rank, n_query)
                projected_train = backend.matmul(
                    backend.transpose(left_sq),
                    backend.transpose(train_grads)
                )  # (rank, n_train)
                return backend.matmul(backend.transpose(right_sq), projected_train)

            if not hasattr(self.ihvp_calculator, "reshape_gradient_per_module"):
                raise ValueError(
                    "Per-module query batching requires an IHVP calculator with "
                    "reshape_gradient_per_module()."
                )

            module_tensors = self.ihvp_calculator.reshape_gradient_per_module(train_grads)  # type: ignore[union-attr]
            module_tensors = {
                layer_idx: module_tensor
                for layer_idx, module_tensor in module_tensors.items()
                if layer_idx in query_rep.module_values
            }

            if config.score_module_partitions <= 1:
                return query_rep.dot_with_modules(module_tensors, backend)

            if not hasattr(self.ihvp_calculator, "layer_map"):
                raise ValueError(
                    "score_module_partitions requires an IHVP calculator with a layer_map."
                )

            scores = None
            layer_partitions = make_module_partitions(
                self.ihvp_calculator.layer_map.layers_info,  # type: ignore[union-attr]
                config.score_module_partitions,
            )
            for layer_infos in layer_partitions:
                query_subset = {}
                train_subset = {}
                for info in layer_infos:
                    layer_idx = info.layer_idx
                    if layer_idx not in query_rep.module_values or layer_idx not in module_tensors:
                        continue
                    query_subset[layer_idx] = query_rep.module_values[layer_idx]
                    train_subset[layer_idx] = module_tensors[layer_idx]

                if not query_subset:
                    continue

                partition_scores = LowRankGradient(query_subset, is_global=False).dot_with_modules(
                    train_subset,
                    backend,
                )
                scores = partition_scores if scores is None else scores + partition_scores

            if scores is None:
                raise ValueError("Query representation contains no compatible module values.")
            return scores

        return backend.matmul(query_rep, backend.transpose(train_grads))

    def _compute_partial_scores(
            self,
            query_rep: Union[Tensor, LowRankGradient],
            train_batch: Tuple[Tensor, ...],
            config: QueryBatchingConfig,
    ) -> Tensor:
        """
        Compute influence scores between a query representation and one training batch.

        Parameters
        ----------
        query_rep
            Preconditioned query representation from :meth:`_precondition_query_batch`.
        train_batch
            A single batch of training samples.
        config
            Query batching configuration controlling score partitioning.

        Returns
        -------
        scores
            Tensor of shape ``(n_query, n_train)`` with influence scores.
        """
        backend = self._backend
        train_partitions = self._split_batch_rows(train_batch, config.score_data_partitions)
        partition_scores = []
        for train_partition in train_partitions:
            scores = self._compute_raw_scores_for_train_batch(query_rep, train_partition, config)
            if self.normalize:
                scores = self._apply_train_side_normalization(scores, train_partition)
            partition_scores.append(scores)

        if len(partition_scores) == 1:
            return partition_scores[0]
        return backend.concat(partition_scores, axis=1)

    def _build_query_score_outputs(
            self,
            query_batches: List[Tuple[Any, ...]],
            query_reps: List[Union[Tensor, LowRankGradient]],
            train_set: DatasetLike,
            config: QueryBatchingConfig,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None,
    ) -> List[Tuple[Tuple[Any, ...], DatasetLike]]:
        """Build per-original-query-batch score datasets for one accumulated group."""
        backend = self._backend
        merged_rep = self._merge_query_representations(query_reps)

        if backend.framework.value == "tensorflow" and config.score_data_partitions > 1:
            full_scores_entries = []
            for train_batch_raw in train_set:
                train_batch = self._as_query_batch(train_batch_raw)
                payload = train_batch if training_payload_extractor is None else self._extract_training_payload(
                    train_batch,
                    training_payload_extractor,
                )
                full_scores_entries.append((
                    payload,
                    self._compute_partial_scores(merged_rep, train_batch, config),
                ))

            outputs = []
            row_start = 0
            for query_batch in query_batches:
                batch_size = int(backend.get_batch_size(query_batch[0]))
                row_end = row_start + batch_size
                if len(query_batches) == 1:
                    batch_scores_ds = self._materialize_entries_dataset(full_scores_entries)
                else:
                    sliced_entries = [
                        (train_batch, scores[row_start:row_end])
                        for train_batch, scores in full_scores_entries
                    ]
                    batch_scores_ds = self._materialize_entries_dataset(sliced_entries)
                outputs.append((query_batch, batch_scores_ds))
                row_start = row_end
            return outputs

        full_scores_ds = backend.map_dataset(
            train_set,
            lambda *tb, qrep=merged_rep: (
                tuple(tb) if training_payload_extractor is None
                else self._extract_training_payload(tuple(tb), training_payload_extractor),
                self._compute_partial_scores(qrep, tuple(tb), config),
            ),
            device,
        )

        if len(query_batches) > 1:
            full_scores_ds = backend.cache_dataset(full_scores_ds)

        outputs = []
        row_start = 0
        for query_batch in query_batches:
            batch_size = int(backend.get_batch_size(query_batch[0]))
            row_end = row_start + batch_size
            if len(query_batches) == 1:
                batch_scores_ds = full_scores_ds
            else:
                batch_scores_ds = backend.map_dataset(
                    full_scores_ds,
                    lambda *item, start=row_start, end=row_end: (item[:-1][0], item[-1][start:end]),
                    device,
                )
            outputs.append((query_batch, batch_scores_ds))
            row_start = row_end
        return outputs

    def estimate_influence_values_query_batched(
            self,
            dataset_to_evaluate: DatasetLike,
            train_set: DatasetLike,
            config: Optional[QueryBatchingConfig] = None,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None,
    ) -> DatasetLike:
        """
        Estimate influence values using query-side preconditioning.

        Instead of pre-computing the IHVP for every training point and then
        dotting against each test gradient (the standard "train-side" path),
        this method applies the IHVP to the *query* gradients and then scores
        them against the raw training gradients.  This is more efficient when
        the number of test points is much smaller than the number of training
        points.

        This method is based on the query-batching algorithm described in
        Studying Large Language Model Generalization with Influence Functions
        arXiv 2308.03296.

        Parameters
        ----------
        dataset_to_evaluate
            A batched dataset of test samples.
        train_set
            A batched dataset of training samples.
        config
            Optional :class:`QueryBatchingConfig`.  If ``None``, a default
            configuration is used with dense query gradients and no score
            partitioning.
        device
            Optional device hint forwarded to backend dataset mapping helpers
            when supported.

        Returns
        -------
        influence_value_dataset
            Dataset-like iterable containing ``(query_batch, scores_dataset)``
            pairs, where ``scores_dataset`` yields ``(train_batch, scores)``
            tuples and ``scores`` has shape ``(n_query, n_train_batch)``.

        Raises
        ------
        ValueError
            If the IHVP calculator does not support query-side preconditioning.
        """
        config = self._validate_query_batched_config(config)

        outputs = []
        query_reps: List[Union[Tensor, LowRankGradient]] = []
        query_batches: List[Tuple[Any, ...]] = []

        for query_batch_raw in dataset_to_evaluate:
            query_batch = self._as_query_batch(query_batch_raw)
            query_reps.append(
                self._precondition_query_batch(
                    query_batch,
                    config,
                    evaluation_representation_provider=evaluation_representation_provider,
                )
            )
            query_batches.append(query_batch)

            if len(query_batches) < config.query_gradient_accumulation_steps:
                continue

            outputs.extend(
                self._build_query_score_outputs(
                    query_batches,
                    query_reps,
                    train_set,
                    config,
                    training_payload_extractor=training_payload_extractor,
                    device=device,
                )
            )
            query_reps = []
            query_batches = []

        if query_reps:
            outputs.extend(
                self._build_query_score_outputs(
                    query_batches,
                    query_reps,
                    train_set,
                    config,
                    training_payload_extractor=training_payload_extractor,
                    device=device,
                )
            )

        return outputs

    def _estimate_influence_values_query_mode(
            self,
            dataset_to_evaluate: DatasetLike,
            train_set: DatasetLike,
            config: Optional[QueryBatchingConfig] = None,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None,
    ) -> DatasetLike:
        """
        Public adapter for query-side influence-value computation.

        Parameters
        ----------
        dataset_to_evaluate
            Dataset containing the samples to score.
        train_set
            Dataset containing the training samples against which influence is
            computed.
        config
            Optional configuration controlling query batching and
            preconditioning.
        evaluation_representation_provider
            Optional callable mapping ``(self.model, batch)`` to the
            representation used when preconditioning evaluation batches. When
            ``None``, the default preprocessing path is used.
        training_payload_extractor
            Optional callable used to extract the payload returned alongside
            influence scores. When ``None``,
            ``default_training_payload_extractor`` is used.
        device
            Device where the computation will be executed.

        Returns
        -------
        influence_value_dataset
            Dataset-like object containing one influence-score dataset per
            evaluation batch.
        """
        return self.estimate_influence_values_query_batched(
            dataset_to_evaluate,
            train_set,
            config=config,
            evaluation_representation_provider=evaluation_representation_provider,
            training_payload_extractor=training_payload_extractor,
            device=device,
        )

    def _infer_training_payload_shape_and_dtype(
            self,
            train_set: DatasetLike,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
    ) -> Tuple[Tuple[int, ...], Any]:
        """Infer the unbatched shape and dtype of the extracted training payload."""
        for batch in train_set:
            train_batch = self._as_query_batch(batch)
            payload = self._extract_training_payload(train_batch, training_payload_extractor)
            shape = tuple(self._backend.tensor_shape(payload)[1:])
            dtype = self._backend.get_dtype(payload)
            return shape, dtype
        raise ValueError("Could not determine training payload shape from dataset")

    def _infer_query_rep_dtype(self, query_rep: Union[Tensor, LowRankGradient]) -> Any:
        """Infer a score dtype from a query representation."""
        if isinstance(query_rep, LowRankGradient):
            first_value = next(iter(query_rep.module_values.values()))
            if isinstance(first_value, tuple):
                return self._backend.get_dtype(first_value[0])
            return self._backend.get_dtype(first_value)
        return self._backend.get_dtype(query_rep)

    def _build_query_top_k_outputs(
            self,
            query_batches: List[Tuple[Any, ...]],
            query_reps: List[Union[Tensor, LowRankGradient]],
            train_set: DatasetLike,
            config: QueryBatchingConfig,
            k: int,
            order: ORDER,
            d_type: Optional[Any] = None,
            payload_dtype: Optional[Any] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
    ) -> List[Tuple[Tuple[Any, ...], Tensor, Tensor]]:
        """Compute top-k results for one accumulated query group."""
        backend = self._backend
        merged_rep = self._merge_query_representations(query_reps)
        total_queries = sum(int(backend.get_batch_size(query_batch[0])) for query_batch in query_batches)
        sample_shape, inferred_payload_dtype = self._infer_training_payload_shape_and_dtype(
            train_set,
            training_payload_extractor,
        )
        if d_type is None:
            d_type = self._infer_query_rep_dtype(merged_rep)
        if payload_dtype is None:
            payload_dtype = inferred_payload_dtype

        batch_sorted = BatchSort(
            sample_shape,
            (total_queries, k),
            dtype=d_type,
            batch_dtype=payload_dtype,
            value_dtype=d_type,
            order=order,
            backend=backend,
        )

        for train_batch_raw in train_set:
            train_batch = self._as_query_batch(train_batch_raw)
            scores = self._compute_partial_scores(merged_rep, train_batch, config)
            batch_input = self._extract_training_payload(train_batch, training_payload_extractor)
            expanded_batch = backend.repeat(
                backend.expand_dims(batch_input, axis=0),
                total_queries,
                axis=0,
            )
            batch_sorted.add_all(expanded_batch, scores)

        training_samples, influence_values = batch_sorted.get()

        outputs = []
        row_start = 0
        for query_batch in query_batches:
            batch_size = int(backend.get_batch_size(query_batch[0]))
            row_end = row_start + batch_size
            outputs.append((
                query_batch,
                influence_values[row_start:row_end],
                training_samples[row_start:row_end],
            ))
            row_start = row_end
        return outputs

    def _tensorflow_signature_from_value(self, value: Any) -> Any:
        """Build a TensorFlow output signature with a flexible batch dimension."""
        from .._optional_imports import import_optional_module

        tf = import_optional_module("tensorflow", extra="tensorflow")

        if isinstance(value, tf.Tensor):
            shape = tuple(value.shape)
            if shape:
                shape = (None,) + tuple(shape[1:])
            return tf.TensorSpec(shape=shape, dtype=value.dtype)
        if isinstance(value, (list, tuple)):
            return type(value)(self._tensorflow_signature_from_value(v) for v in value)
        raise TypeError(f"Unsupported TensorFlow dataset value type: {type(value)!r}")

    def _materialize_entries_dataset(self, entries: List[Any]) -> DatasetLike:
        """Convert materialized entries to a backend-appropriate dataset-like object."""
        if self._backend.framework.value != "tensorflow" or not entries:
            return entries

        from .._optional_imports import import_optional_module

        tf = import_optional_module("tensorflow", extra="tensorflow")
        output_signature = self._tensorflow_signature_from_value(entries[0])
        return tf.data.Dataset.from_generator(lambda: iter(entries), output_signature=output_signature)

    def _top_k_query_mode(
            self,
            dataset_to_evaluate: DatasetLike,
            train_set: DatasetLike,
            k: int = 5,
            order: ORDER = ORDER.DESCENDING,
            d_type: Optional[Any] = None,
            payload_dtype: Optional[Any] = None,
            config: Optional[QueryBatchingConfig] = None,
            evaluation_representation_provider: Optional[EvaluationRepresentationProvider] = None,
            training_payload_extractor: Optional[TrainingPayloadExtractor] = None,
            device: Optional[str] = None,
    ) -> DatasetLike:
        """
        Public adapter for query-side top-k computation.

        Parameters
        ----------
        dataset_to_evaluate
            Dataset containing the samples to score.
        train_set
            Dataset containing the training samples from which to retrieve the
            most influential entries.
        k
            Number of most influential training samples to retain.
        order
            Either ``ORDER.DESCENDING`` or ``ORDER.ASCENDING`` depending on
            whether the most or least influential samples are requested.
        d_type
            Data-type of the influence scores. If ``None``, it is inferred.
        payload_dtype
            Data-type used to store extracted training payloads in the sorted
            structure. If ``None``, it is inferred.
        config
            Optional configuration controlling query batching and
            preconditioning.
        evaluation_representation_provider
            Optional callable mapping ``(self.model, batch)`` to the
            representation used when preconditioning evaluation batches. When
            ``None``, the default preprocessing path is used.
        training_payload_extractor
            Optional callable used to extract the payload returned alongside
            top-k influence scores. When ``None``,
            ``default_training_payload_extractor`` is used.
        device
            Device where the computation will be executed.

        Returns
        -------
        top_k_dataset
            Dataset-like object containing the top-k influence scores and
            payloads for each evaluation batch.
        """
        _ = device
        config = self._validate_query_batched_config(config)

        outputs = []
        query_reps: List[Union[Tensor, LowRankGradient]] = []
        query_batches: List[Tuple[Any, ...]] = []

        for query_batch_raw in dataset_to_evaluate:
            query_batch = self._as_query_batch(query_batch_raw)
            query_reps.append(
                self._precondition_query_batch(
                    query_batch,
                    config,
                    evaluation_representation_provider=evaluation_representation_provider,
                )
            )
            query_batches.append(query_batch)

            if len(query_batches) < config.query_gradient_accumulation_steps:
                continue

            outputs.extend(
                self._build_query_top_k_outputs(
                    query_batches,
                    query_reps,
                    train_set,
                    config,
                    k,
                    order,
                    d_type,
                    payload_dtype,
                    training_payload_extractor,
                )
            )
            query_reps = []
            query_batches = []

        if query_reps:
            outputs.extend(
                self._build_query_top_k_outputs(
                    query_batches,
                    query_reps,
                    train_set,
                    config,
                    k,
                    order,
                    d_type,
                    payload_dtype,
                    training_payload_extractor,
                )
            )

        return self._materialize_entries_dataset(outputs)

    def compute_influence_vector_group(
            self,
            group: DatasetLike
    ) -> Tensor:
        """
        Computes the influence function vector -- an estimation of the weights difference when
        removing the points -- of the whole group of points.

        Parameters
        ----------
        group
            A batched dataset containing the group of points of which we wish to compute the
            influence of removal.

        Returns
        -------
        influence_group
            A tensor containing one vector for the whole group.
        """
        self._backend.assert_batched_dataset(group)

        ihvp_ds = self.ihvp_calculator.compute_ihvp(group)
        reduced_ihvp = self._reduce_ihvp_batches(ihvp_ds, keepdims=True)

        reduced_ihvp = self._normalize_if_needed(reduced_ihvp)

        influence_group = self._backend.reshape(reduced_ihvp, (1, -1))

        return influence_group

    def estimate_influence_values_group(
            self,
            group_train: DatasetLike,
            group_to_evaluate: Optional[DatasetLike] = None
    ) -> Tensor:
        """
        Computes Cook's distance of the whole group of points provided, giving measure of the
        influence that the group carries on the model's weights.

        The dataset_train contains the points we will be removing and dataset_to_evaluate,
        those with respect to whom we will be measuring the influence. As we will be performing
        the same operation in batches, we consider that each point from one dataset corresponds
        to one from the other. As such, both datasets must contain the same amount of points.
        In case the group_to_evaluate is not given, use by default the
        group_to_train: compute the self influence of the group.


        Parameters
        ----------
        group_train
            A batched dataset containing the group of points we wish to remove.
        group_to_evaluate
            A batched dataset containing the group of points with respect to whom we wish to
            measure the influence of removing the training points.

        Returns
        -------
        influence_values_group
            A tensor containing one influence value for the whole group.
        """
        if group_to_evaluate is None:
            # default to self influence
            group_to_evaluate = group_train

        ds_size = self.assert_compatible_datasets(group_train, group_to_evaluate)

        # Compute reduced gradients
        jacobian = self.model.batch_jacobian(group_to_evaluate)
        reduced_grads = self._backend.reduce_sum(
            self._backend.reshape(jacobian, (ds_size, -1)),
            axis=0, keepdims=True
        )

        # Compute and reduce IHVP
        ihvp_ds = self.ihvp_calculator.compute_ihvp(group_train)
        reduced_ihvp = self._reduce_ihvp_batches(ihvp_ds, keepdims=True)

        reduced_ihvp = self._normalize_if_needed(reduced_ihvp)

        influence_values_group = self._backend.matmul(reduced_grads, reduced_ihvp)

        return influence_values_group
