# Weights boundary

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/influenciae/blob/main/deel/influenciae/boundary_based/weights_boundary.py)

For a completely different notion of influence or importance of data-points, we propose to measure the budget (measured
through an $\el^2$ metric) needed to minimally perturb the model's weights such that the data-point under study gets
misclassified. Ideally, it would make sense for more influential images to need a smaller budget (i.e. a smaller change
on the model) to make the model change its prediction on them.

In particular, we define the influence score as follows:

$$ \mathcal{I}_{WB} (z) = - \lVert w - w_{adv} \rVert^2 \, , $$
where $w$ is the model's weights and $w_{adv}$ is the perturbed model with the lowest possible budget and 
obtained through an adaptation of the [DeepFool method](https://arxiv.org/abs/1511.04599).

This technique is based on a simple idea we had, and as such, there's no paper associated to it. We decided to include
it because it seems that its performance is less dependent on the choice of model and training schedule and still
obtains acceptable results on our mislabeled point detection benchmark.

## Notebooks

- [**Using Boundary-based Influence**](https://colab.research.google.com/drive/1785eHgT91FfqG1f25s7ovqd6JhP5uklh?usp=sharing) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1785eHgT91FfqG1f25s7ovqd6JhP5uklh?usp=sharing) </sub>

{{deel.influenciae.boundary_based.weights_boundary.WeightsBoundaryCalculator}}
