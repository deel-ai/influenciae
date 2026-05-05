# Training Payloads

## Motivation

Influence methods score evaluation samples against training samples.  The score
itself is only half of the output: downstream users also need to know which
training entries those scores refer to.

For simple supervised datasets, returning the model input tensor is often enough.
For larger or structured datasets, it is usually more useful to return stable
sample ids, filenames, or another compact tensor that can be joined back to the
original dataset.

The **training payload extractor** pattern lets callers choose that training-side
output payload without changing the influence computation itself.

## How it works

`BaseInfluenceCalculator._extract_training_payload` is the dispatch point for
custom payload extraction:

```python
def _extract_training_payload(self, train_batch, training_payload_extractor=None):
    extractor = (
        default_training_payload_extractor
        if training_payload_extractor is None
        else training_payload_extractor
    )
    return extractor(train_batch)
```

- **Default path** (`None`): uses `default_training_payload_extractor`, which
  returns `batch[0]`.  This preserves the historical behavior for standard
  `(inputs, targets, ...)` datasets.
- **Custom path**: delegates to the user-supplied extractor, which can return any
  backend-compatible batched tensor-like payload.

The extracted payload is carried in output datasets next to influence scores.  It
does not participate in gradient computation and does not affect the influence
values.

All public methods that return training-side score payloads accept an optional
`training_payload_extractor` keyword argument:

- `estimate_influence_values_in_batches`
- `top_k`
- `estimate_influence_values_query_batched`
- `_top_k_query_mode` / `_estimate_influence_values_query_mode`

## The `TrainingPayloadExtractor` protocol

Any callable satisfying this signature is accepted:

```python
from deel.influenciae.common.payloads import TrainingPayloadExtractor

class TrainingPayloadExtractor(Protocol):
    def __call__(self, batch: Tuple[Any, ...]) -> Tensor:
        ...
```

The returned payload must be batched along the first dimension, with the same
leading batch size as the training batch.  For `top_k`, the payload must also have
a stable shape and dtype so the sorted result buffers can be allocated.

Examples of useful payloads include:

- `batch[0]`: the model input tensor, which is the default for standard datasets.
- `batch[2]`: a tensor of stable sample ids stored in the batch.
- A tensor built from filenames, row ids, or metadata already present in the
  training batch.

## Built-in: `default_training_payload_extractor`

The library provides `default_training_payload_extractor`:

```python
from deel.influenciae.common.payloads import default_training_payload_extractor

payload = default_training_payload_extractor(batch)  # returns batch[0]
```

This works well when the first element of each training batch is the tensor you
want to inspect in influence outputs.  Use a custom extractor when the first
element is too large, unstable, or not the right identifier for your application.

## Writing a custom extractor

Here is a minimal extractor that returns stable sample ids from a dict-style
PyTorch batch:

```python
from deel.influenciae.influence import FirstOrderInfluenceCalculator

def sample_id_payload(batch):
    # Some backend dataset helpers pass a single dict batch as `(batch,)`.
    if isinstance(batch, (list, tuple)) and len(batch) == 1 and isinstance(batch[0], dict):
        batch = batch[0]
    return batch["sample_id"]

calculator = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator="exact")

top_k = calculator.top_k(
    samples_to_explain,
    train_dataset,
    k=5,
    training_payload_extractor=sample_id_payload,
)
```

The returned `top_k` entries contain the selected sample-id payloads instead of
the raw training inputs.  This is often preferable when images or structured
inputs are large and downstream reporting only needs a dataset key.

The same extractor can be used with batched influence-value scoring:

```python
scores = calculator.estimate_influence_values_in_batches(
    samples_to_explain,
    train_dataset,
    training_payload_extractor=sample_id_payload,
)
```

## Relationship to evaluation representations

`training_payload_extractor` and `evaluation_representation_provider` are sibling
extension points:

- `evaluation_representation_provider` controls how evaluation/query batches are
  differentiated before scoring.
- `training_payload_extractor` controls what training-side payload is returned
  next to the scores.

They can be used independently or together.  Structured tasks such as object
detection commonly use both: a custom evaluation representation for the query
objective and a custom training payload so returned results contain stable sample
ids instead of raw images.

See [Custom Evaluation Representations](evaluation_representation.md) for the
query-side hook.

## API reference

{{deel.influenciae.common.payloads}}
