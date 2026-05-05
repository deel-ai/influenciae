# Custom Evaluation Representations

## Motivation

By default the influence calculator scores evaluation (query) batches by computing
per-sample gradients of the model's training loss through `_preprocess_samples`.
This works well for standard supervised classifiers, but structured tasks -- such as
object detection -- need to express an evaluation batch through the gradient of a
**different**, task-specific objective.

The **evaluation representation provider** pattern decouples this preprocessing from
the influence math: the core calculator delegates to a user-supplied callable
whenever one is provided, without knowing anything about the task.

## How it works

`BaseInfluenceCalculator._get_evaluation_representation` is the dispatch point:

```python
def _get_evaluation_representation(self, samples, evaluation_representation_provider=None):
    if evaluation_representation_provider is None:
        return self._preprocess_samples(samples)          # default path
    return evaluation_representation_provider(self.model, samples)  # custom path
```

- **Default path** (`None`): falls through to the subclass's `_preprocess_samples`,
  which typically calls `self.model.batch_jacobian_tensor(samples)`.
- **Custom path**: delegates to the provider, which must return a tensor with the same
  shape contract -- `(batch_size, nb_params)` -- but is free to compute it however it
  needs to.

All public methods that score evaluation batches accept an optional
`evaluation_representation_provider` keyword argument:

- `estimate_influence_values_in_batches`
- `top_k`
- `estimate_influence_values_query_batched`
- `_top_k_query_mode` / `_estimate_influence_values_query_mode`

## The `EvaluationRepresentationProvider` protocol

Any callable satisfying this signature is accepted:

```python
from deel.influenciae.common.evaluation import EvaluationRepresentationProvider

class EvaluationRepresentationProvider(Protocol):
    def __call__(
        self,
        model: BaseInfluenceModel,
        batch: Tuple[Any, ...],
    ) -> Tensor:
        ...
```

The returned tensor must be batched along the first dimension and aligned with the
model's watched parameters.  For first-order influence methods this usually means
a tensor of shape `(batch_size, nb_params)`.

## Built-in: `ObjectiveEvaluationRepresentationProvider`

For the common case where the custom representation is simply the gradient of a
different loss function, the library provides
`ObjectiveEvaluationRepresentationProvider`:

```python
from deel.influenciae.common.evaluation import ObjectiveEvaluationRepresentationProvider

provider = ObjectiveEvaluationRepresentationProvider(
    objective=my_custom_loss,
    process_batch_for_objective_fn=my_unpack_fn,
)
```

Where:

- `objective` is a loss callable compatible with `backend.compute_jacobian` (no
  batch reduction -- it must return one value per sample).
- `process_batch_for_objective_fn` converts a raw dataset batch to
  `(model_input, y_true, sample_weight)`.

Its `__call__` then computes:

```python
model.backend.compute_jacobian(model.model, model.weights, objective, model_inp, y_true, sample_weight)
```

## Writing a custom provider

You can implement the protocol with any callable.  Here is a minimal example that
overrides the evaluation gradients with a custom objective:

```python
import torch
from deel.influenciae.common import InfluenceModel
from deel.influenciae.common.evaluation import ObjectiveEvaluationRepresentationProvider
from deel.influenciae.influence import FirstOrderInfluenceCalculator

# 1. Define a custom per-sample loss (no batch reduction).
def my_custom_loss(predictions, targets):
    return torch.nn.functional.mse_loss(predictions, targets, reduction="none").sum(dim=-1)

# 2. Define how to unpack a raw batch into (model_input, y_true, sample_weight).
def unpack_batch(batch):
    x, y = batch
    return x, y, None

# 3. Build the provider.
provider = ObjectiveEvaluationRepresentationProvider(
    objective=my_custom_loss,
    process_batch_for_objective_fn=unpack_batch,
)

# 4. Pass it to the calculator.
influence_model = InfluenceModel(model, start_layer=-1, loss_function=train_loss)
calculator = FirstOrderInfluenceCalculator(influence_model, train_dataset, "exact")

scores = calculator.estimate_influence_values_in_batches(
    test_dataset,
    train_dataset,
    evaluation_representation_provider=provider,
)
```

For a fully bare-bones provider, you can also implement the protocol directly:

```python
class ConstantProvider:
    """Always return zero gradients (useful for testing)."""

    def __call__(self, model, batch):
        import torch
        n = batch[0].shape[0]
        nb_params = sum(p.numel() for p in model.weights)
        return torch.zeros(n, nb_params)

calculator.estimate_influence_values_in_batches(
    test_dataset,
    train_dataset,
    evaluation_representation_provider=ConstantProvider(),
)
```

## API reference

{{deel.influenciae.common.evaluation}}
