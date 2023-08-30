# Sample boundary

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/influenciae/blob/main/deel/influenciae/boundary_based/sample_boundary.py)

For a completely different notion of influence or importance of data-points, we propose to measure the distance that
separates each data-point from the decision boundary, and assign a higher influence score to the elements that are 
closest to the decision boundary. It would make sense for these examples to be the most influential, as if they weren't
there, the model would have placed the decision boundary elsewhere.

In particular, we define the influence score as follows:

$$ \mathcal{I}_SB (z) = - \lVert z - z_{adv} \rVert^2 \, , $$
where $z$ is the data-point under study and $z_{adv}$ is the adversarial example with the lowest possible budget 
and obtained through the [DeepFool method](https://arxiv.org/abs/1511.04599).

This technique is based on a simple idea we had, and as such, there's no paper associated to it. We decided to include
it because it seems that its performance is less dependent on the choice of model and training schedule and still
obtains acceptable results on our mislabeled point detection benchmark.

## Notebooks

- [**Using Boundary-based Influence**](https://drive.google.com/file/d/145Gi4gCYTKlRVJjsty5cPkdMGNJoNDws/view?usp=share_link)

{{deel.influenciae.boundary_based.sample_boundary.SampleBoundaryCalculator}}
