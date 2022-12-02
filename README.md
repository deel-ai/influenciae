# Influenciae

Influenciae is a Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset and the generation of data-centric explanations for deep learning models. In this library based on Tensorflow, we gather state-of-the-art methods for estimating the importance of training samples and their influence on test data-points for validating the quality of datasets and of the models trained on them.

## 🔥 Tutorials

We propose some hands-on tutorials to get familiar with the library and it's API:

- [Getting Started](https://drive.google.com/file/d/145Gi4gCYTKlRVJjsty5cPkdMGNJoNDws/view?usp=share_link) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/145Gi4gCYTKlRVJjsty5cPkdMGNJoNDws/view?usp=share_link) </sub>
- Benchmarking with Mislabeled sample detection (**WIP**)
- Using the first order influence calculator (**WIP**)
- Using the second order influence calculator (**WIP**)
- Using TracIn (**WIP**)
- Using Representer Point Selection - L2 (RPS_L2) (**WIP**)
- Using Representer Point Selection - Local Jacobian Expansion (RPS_LJE) (**WIP**)

## 🚀 Quick Start

Influenciae requires a version of python higher than 3.6 and several libraries, including Tensorflow and Numpy. Installation can be done using Pypi:

```python
pip install deel-influenciae
```

Once Influenciae is installed, there are two major applications for the different modules (that all follow the same API).
So, except for group-specific functions that are only available on the `influence` module, all the classes are able to compute self-influence values, the influence with one point wrt another, as well as find the top-k samples for both of these situations.

### Discovering influential examples

Particularly useful when validating datasets, influence functions (and related notions) allow for gaining an insight into what samples the models thinks to be "important". For this, the training dataset and a trained model are needed.

```python
from deel.influenciae.common import InfluenceModel, ExactIHVP
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils import ORDER

# load the model, the training loss (without reduction) and the training data (with the labels and in a batched TF dataset)

influence_model = InfluenceModel(model, target_layer, loss_function)
ihvp_calculator = ExactIHVP(influence_model, train_dataset)
influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)
data_and_influence_dataset = influence_calculator.compute_influence_values(train_dataset)
# or influence_calculator.compute_top_k_from_training_dataset(train_dataset, k_samples, ORDER.DESCENDING) when the
# dataset is too large
```

This is also explained more in depth in the [Getting Started tutotial](https://drive.google.com/file/d/145Gi4gCYTKlRVJjsty5cPkdMGNJoNDws/view?usp=share_link) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/145Gi4gCYTKlRVJjsty5cPkdMGNJoNDws/view?usp=share_link) </sub>

### Explaining neural networks through their training data

Another application is to explain some model's predictions by looking on which training samples they are based on. Again, the training dataset, the model and the samples we wish to explain are needed.

```python
from deel.influenciae.common import InfluenceModel, ExactIHVP
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils import ORDER

# load the model, the training loss (without reduction), the training data and
# the data to explain (with the labels and in batched a TF dataset)

influence_model = InfluenceModel(model, target_layer, loss_function)
ihvp_calculator = ExactIHVP(influence_model, train_dataset)
influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)
data_and_influence_dataset = influence_calculator.estimate_influence_values_in_batches(samples_to_explain, train_dataset)
# or influence_calculator.top_k(samples_to_explain, train_dataset, k_samples, ORDER.DESCENDING) when the
# dataset is too large
```

This is also explained more in depth in the [Getting Started tutorial](https://drive.google.com/file/d/145Gi4gCYTKlRVJjsty5cPkdMGNJoNDws/view?usp=share_link) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/145Gi4gCYTKlRVJjsty5cPkdMGNJoNDws/view?usp=share_link) </sub>

### Determining the influence of groups of samples

The previous examples use notions of influence that are applied individually to each data-point, but it is possible to extend this to groups. That is, answer the question of what would a model look like if it hadn't seen a whole group of data-points during training, for example. This can be computed namely using the `FirstOrderInfluenceCalculator` and `SecondOrderInfluenceCalculator`, for implementations where pairwise interactions between each of the data-points are not taken into account and do, respectively.

For obtaining the groups' influence:

```python
from deel.influenciae.common import InfluenceModel, ExactIHVP
from deel.influenciae.influence import SecondOrderInfluenceCalculator

# load the model, the training loss (without reduction), the training data and
# the data to explain (with the labels and in a batched TF dataset)

influence_model = InfluenceModel(model, target_layer, loss_function)
ihvp_calculator = ExactIHVP(influence_model, train_dataset)
influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)  # or FirstOrderInfluenceCalculator
data_and_influence_dataset = influence_calculator.estimate_influence_values_group(groups_train, groups_to_explain)
```

For the data-centric explanations:

```python
from deel.influenciae.common import InfluenceModel, ExactIHVP
from deel.influenciae.influence import SecondOrderInfluenceCalculator

# load the model, the training loss (without reduction), the training data and
# the data to explain (with the labels and in a batched TF dataset)

influence_model = InfluenceModel(model, target_layer, loss_function)
ihvp_calculator = ExactIHVP(influence_model, train_dataset)
influence_calculator = SecondOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator)  # or FirstOrderInfluenceCalculator
data_and_influence_dataset = influence_calculator.estimate_influence_values_group(groups_train)
```

## 📦 What's Included

All the influence calculation methods work on Tensorflow models trained for any sort of task and on any type of data. Visualization functionality is implemented for image datasets only (for the moment).

| **Influence Method**                                    | Source                                    |                                      Tutorial                                      |
|:--------------------------------------------------------| :---------------------------------------- |:----------------------------------------------------------------------------------:|
| Influence Functions                                     | [Paper](https://arxiv.org/abs/1703.04730)  |                                        WIP                                         |
| RelatIF                                                 | [Paper](https://arxiv.org/pdf/2003.11630.pdf)  |                                        WIP                                         |
| Influence Functions  (first order, groups)              | [Paper](https://arxiv.org/abs/1905.13289)  |                                        WIP                                         |
| Influence Functions  (second order, groups)             | [Paper](https://arxiv.org/abs/1911.00418)  | WIP  |
| Representer Point Selection  (L2)                       | [Paper](https://arxiv.org/abs/1811.09720)  | WIP  |
| Representer Point Selection  (Local Jacobian Expansion) | [Paper](https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf)  | WIP  |
| Trac-In                                                 | [Paper](https://arxiv.org/abs/2002.08484)  | WIP  |

## 👀 See Also

This library proposes implementations of some of the different popular ways of calculating the influence of data-points on TF, but there are also other ones using other frameworks. 

Some other tools for efficiently computing influence functions.

- [Scaling Up Influence Functions](https://github.com/google-research/jax-influence) a Python library using JAX implementing a scalable algorithm for computing influence functions.
- [FastIF: Scalable Influence Functions for Efficient Model Interpretation and Debugging](https://github.com/salesforce/fast-influence-functions) a Python library using PyTorch implementing another scalable algorithm for computing influence functions.

More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library exclusively dedicated to explaining neural networks.
- [deel-lip](https://github.com/deel-ai/deel-lip) a Python library for training k-Lipschitz neural networks on TF.
- [deel-torchlip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%">
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
