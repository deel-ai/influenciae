# Representer Point Selection - L2

📰 [Original Paper](https://arxiv.org/abs/1811.09720)

Using a completely different notion of influence than the techniques in modules `deel.influenciae.influence`
and `deel.influenciae.trac_in`, this is the first method to use the representer point theorem for kernels
to attribute an influence to data-points in the training dataset. In particular, it posits that the
classification function can be approximated by:

$$ \Phi (x_t, \theta) = \sum_i^n k (x_t, x_i, \alpha_i) $$

$$ \alpha_i = - \frac{1}{2 \lambda n} \frac{\partial L (x_i, y_i, \theta)}{\partial \Phi (x_i, \theta)} $$

where $\Phi$ is the function that transforms the points $x_t$ in the embedding of the network's last layer
to the logits for the classification. However, this function must be learned with a strong $\ell^2$ regularization,
and thus requires creating a surrogate model.

In particular, it will be the $\alpha_i$ for the predicted label $j$ the equivalent of the influence of
the data-point.

This method does not require us to compute inverse-hessian-vector products, so it can be computed with
a certain efficiency once the surrogate model has been learned.


## Notebooks

- **WIP**


{{deel.influenciae.rps.rps_l2.RepresenterPointL2}}
