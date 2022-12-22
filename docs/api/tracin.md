# TracIn

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/influenciae/blob/main/deel/influenciae/trac_in/tracin.py) |
📰 [Original Paper](https://arxiv.org/abs/2002.08484)

This method proposes an alternative for estimating influence without the need for expensive inverse
hessian-vector product computations, but requiring information that is only available at train time.
It leverages the fundamental theorem of calculus to estimate the influence of the training points
by looking at how the loss at that point evolves at different model checkpoints. Concretely, the
influence will take the following for:

$$ \mathcal{I} (z, z') = \sum_i^k \eta_i \nabla_\theta \ell (\theta_{t_i}, z) \cdot \nabla_\theta \ell (\theta_{t_i}, z') $$

where $\theta_{t_i}$ are the model's weights at epoch $t_i$ and $\eta_i$ is the learning rate at that same epoch.

Just like RPS-L2, this method does not need an instance of the `InverseHessianVectorProduct` class,
but does require to provide some of the model's checkpoints and the learning rates at each of them.


## Notebooks

- [**Using TracIn**](https://colab.research.google.com/drive/1E94cGF46SUQXcCTNwQ4VGSjXEKm7g21c?usp=sharing)

{{deel.influenciae.trac_in.tracin.TracIn}}
