# Benchmarks

This module allows users to easily and objectively compare the performance of the different techniques
that have been implemented in this library, or new ones that may come up. The benchmark is based on
a widespread test used in the literature to measure the different methods' capacity to find problematic
data-points: train a model on a dataset that has a certain percentage of mislabeled data-points (that are
already known for validation purposes), and use the influence values to rank them and see what percentage of
the whole dataset must be checked to find all the mislabeled points when ordered that way. In particular,
a good metric for this is the AUC (area under the curve) of the ROC curve of this operation.

Additionally, as we are working with stochastic algorithms, we need to make sure that the scores we obtain
have a statistical value. This means that this operation should be repeated over multiple seeds to make sure
that the method really does perform better than others. For this reason, a training procedure must be created
so as to be able to automatically repeat the process of training the model on noisy versions of the original
dataset.

In particular, this module includes interfaces for training procedures and mislabeled sample detection
benchmarking, and an implementation on the popular CIFAR10 image-classification dataset.

## Notebooks

- [**Benchmarking with Mislabeled sample detection**](https://colab.research.google.com/drive/1_5-RC_YBHptVCElBbjxWfWQ1LMU20vOp?usp=sharing)

{{deel.influenciae.benchmark.cifar10_benchmark.Cifar10TrainingProcedure}}

{{deel.influenciae.benchmark.cifar10_benchmark.Cifar10MislabelingDetectorEvaluator}}
