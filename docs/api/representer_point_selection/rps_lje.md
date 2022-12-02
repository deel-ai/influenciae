# Representer Point Selection - Local Jacobian Expansion

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/influenciae/blob/main/deel/influenciae/rps/rps_lje.py) |
📰 [Original Paper](https://proceedings.neurips.cc/paper/2021/file/c460dc0f18fc309ac07306a4a55d2fd6-Paper.pdf)

Introduced as an improvement over [Representer Point Selection - L2](https://arxiv.org/abs/1811.09720), this
technique trades the surrogate model for a local taylor expansion on the jacobian matrix, effectively allowing
for the decomposition of the model's last layer into a kernel as an approximation. 
In short, it proposes the following formula for computing influence values:

$$ \Theta_L^\dagger \phi (x_t) = \sum_i \alpha_i \phi (x_i)^T \phi (x_y) $$

$$ \alpha_i = \Theta_L \frac{1}{\phi (x_i) \, n} - \frac{1}{n} H_{\Theta_L}^{-1} \frac{\partial L (x_i, y_i, \Theta)}{\partial \Theta_L \phi (x_i)} $$

In particular, it will be the $\alpha_i$ for the predicted label $j$ the equivalent of the influence of
the data-point.

As all the other methods based on computing inverse-hessian-vector products, it will be performing all
these computations with the help of objects of the class `InverseHessianVectorProduct`, capable of doing
so efficiently.

## Notebooks

- **WIP**

{{deel.influenciae.rps.rps_lje.RepresenterPointLJE}}
