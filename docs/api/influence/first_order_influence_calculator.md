# First Order Influence Calculator

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/influenciae/blob/main/deel/influenciae/influence/first_order_influence_calculator.py) |
📰 [Original Paper](https://arxiv.org/abs/1703.04730) |
📰 [Paper Groups](https://arxiv.org/abs/1905.13289) |
📰 [Paper RelatIF](https://arxiv.org/abs/2003.11630) |

This method is an implementation of the famous technique introduced by Koh & Liang in 2017. 
In essence, by performing a first-order taylor approximation, it proposes that the influence 
function of a neural network model can be computed as follows:

$$ \mathcal{I} (z) \approx H_{\hat{\theta}}^{-1} \, \nabla_\theta \ell (\hat{\theta}, z), $$

where $H_{\hat{\theta}}^{-1}$ is the inverse of the mean hessian of the loss wrt the model's parameters
over the whole dataset, $\ell$ is the loss function with which the model was trained and $z$, a point
we wish to leave out of the training dataset.

In particular, this computation is carried out by the `InverseHessianVectorProduct` class, which allows
to do it in different ways, with each implementation having its pros and cons.

It can be used to compute the self-influence of individual and groups of points, and the influence of
training points (or groups) on other test points (or groups).

It also implements the RelatIF technique, which can be computed by setting the `normalize` attribute
to `True`.

## Notebooks

- [**Getting started**](https://drive.google.com/file/d/145Gi4gCYTKlRVJjsty5cPkdMGNJoNDws/view?usp=share_link)
- [**Using the first order influence calculator**](https://colab.research.google.com/drive/1WlYcQNu5obhVjhonN2QYi8ybKyZJl4iY?usp=sharing)

{{deel.influenciae.influence.first_order_influence_calculator.FirstOrderInfluenceCalculator}}
