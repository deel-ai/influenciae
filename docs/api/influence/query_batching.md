# Query Batching

`FirstOrderInfluenceCalculator` supports a query-side execution mode that applies the inverse-
hessian-vector product to query gradients before scoring them against raw training gradients.

This mode is enabled with `preconditioning_mode=PreconditioningMode.QUERY` when calling
`estimate_influence_values_in_batches()` or `top_k()`.

It follows the same core idea used in Kronfluence and in Grosse et al. (2023): when the number
of query points is small compared to the number of training points, it is often cheaper to
precondition the query gradients once and reuse them across the whole training set.

## When to use it

Query batching is useful when:

- the training set is much larger than the set of query points,
- train-side influence vectors would be expensive to materialize or cache,
- you want to combine query accumulation, train-batch partitioning, or optional low-rank query compression.

The resulting dataset structure is the same as with the standard batched scoring API: each original
query batch is paired with a dataset-like object yielding `(train_batch, scores)`.

## Supported IHVP implementations

Query-side preconditioning requires an IHVP calculator with
`supports_query_preconditioning=True`.

The currently supported implementations are:

- `ExactIHVP`
- `KfacIHVP`
- `EkfacIHVP`

Iterative methods such as `ConjugateGradientDescentIHVP` keep the standard train-side path only.

## Basic usage

```python
from deel.influenciae.common import ExactIHVP, InfluenceModel
from deel.influenciae.common.query_batching import PreconditioningMode, QueryBatchingConfig
from deel.influenciae.influence import FirstOrderInfluenceCalculator

influence_model = InfluenceModel(model, start_layer=-1, loss_function=loss_fn)
ihvp = ExactIHVP(influence_model, train_dataset)
calculator = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp)

scores = calculator.estimate_influence_values_in_batches(
    samples_to_explain,
    train_dataset,
    preconditioning_mode=PreconditioningMode.QUERY,
    query_batching_config=QueryBatchingConfig(
        query_gradient_accumulation_steps=2,
        score_data_partitions=2,
    ),
)
```

## Top-k usage

```python
top_k = calculator.top_k(
    samples_to_explain,
    train_dataset,
    k=5,
    preconditioning_mode=PreconditioningMode.QUERY,
    query_batching_config=QueryBatchingConfig(query_gradient_accumulation_steps=2),
)
```

`top_k()` keeps one output per original query batch and uses the dedicated query-side streaming
implementation internally.

## Configuration

`QueryBatchingConfig` exposes the following controls:

- `query_gradient_accumulation_steps`: number of query batches to merge before scoring.
- `query_gradient_low_rank`: optional low-rank compression rank for preconditioned query gradients.
- `query_gradient_svd_dtype`: optional dtype used during low-rank compression.
- `score_data_partitions`: number of contiguous row partitions used when scoring each training batch.
- `score_module_partitions`: number of module partitions used with factorized IHVP methods.

## Low-rank compression

When `query_gradient_low_rank` is set, the preconditioned query representation is compressed before
scoring:

- `ExactIHVP` uses a global low-rank representation of the full flattened query gradient matrix.
- `KfacIHVP` and `EkfacIHVP` can keep the representation factorized per module.

Per-module low-rank compression is useful when the K-FAC or EK-FAC query representation is naturally
structured by layer. Global low-rank compression is still available for ExactIHVP.

## Normalization

When `normalize=True`, query mode preserves the standard RelatIF semantics: scores are normalized by
the norm of the train-side IHVP vectors rather than the query-side preconditioned representation.

## Known constraints

- Query mode rejects train-side influence-vector load/save/cache options.
- Query-side preconditioning is only available for IHVP implementations where
  `supports_query_preconditioning=True`.
- TensorFlow currently computes low-rank SVDs via full `tf.linalg.svd` followed by truncation.
- Global low-rank representations are materialized back to dense when multiple query batches are
  accumulated together, because their SVD factors are batch-local.

## Custom evaluation representations

The query-batched methods (`estimate_influence_values_query_batched`, `_top_k_query_mode`)
accept an optional `evaluation_representation_provider` argument.  When supplied, the
provider replaces the default `_preprocess_samples` gradient computation so that
evaluation batches can be scored through a task-specific objective.

This is particularly useful for structured tasks such as object detection, where a
packed batch must first be unpacked and then differentiated through a detection-specific
loss rather than the model's training loss.

See [Custom Evaluation Representations](../evaluation_representation.md) for the protocol
definition, the built-in `ObjectiveEvaluationRepresentationProvider`, and a usage example.

## Training payloads

Query-batched score outputs also accept `training_payload_extractor`.  The extractor
does not affect query preconditioning or score computation; it only controls the
training-side payload returned alongside each score block or top-k result.

This is useful when query batching is used on large structured datasets, where returning
sample ids is more practical than returning raw training inputs.  See
[Training Payloads](../training_payloads.md) for the protocol definition and examples.

## Notes

- `load_influence_vector_path`, `save_influence_vector_path`, and non-default influence-vector cache
  settings apply to the standard train-side path and are rejected in query mode.
- RelatIF normalization still uses the original train-side normalization semantics.

## References

- Grosse et al. (2023), [arXiv:2308.03296](https://arxiv.org/abs/2308.03296)
- Kronfluence: [github.com/pomonam/kronfluence](https://github.com/pomonam/kronfluence)
