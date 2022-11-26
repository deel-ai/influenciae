# Second Order Influence Calculator

📰 [Original Paper](https://arxiv.org/abs/1911.00418)

When working with groups of data, it can prove useful to take into account the pairwise interactions
in terms of influence when leaving out large groups of data-points. Basu et al. have thus introduced a
second-order formulation that takes these interactions into account:

$$ \mathcal{I}^{(2)} (\mathcal{U}, z_t) = \nabla_\theta \ell (\hat{\theta}, z_t) \left(\mathcal{I}^{(1)} (\mathcal{U}) + \mathcal{I}' (\mathcal{U})\right) $$

$$ \mathcal{I}^{(1)} (\mathcal{U}) = \frac{1 - 2 p}{(1 - p)^2} \frac{1}{|\mathcal{S}|} H_{\hat{\theta}}^{-1} \sum_{z \in \mathcal{U}} \nabla_\theta \ell (\hat{\theta}, z) $$

$$ \mathcal{I}' (\mathcal{U}) = \frac{1}{(1 - p)^2} \frac{1}{|S|^2} \sum_{z \in \mathcal{U}} H_{\hat{\theta}}^{-1} \nabla_\theta^2 (\hat{\theta}, z) H_{\hat{\theta}}^{-1} \sum_{z' \in \mathcal{U}} \nabla_\theta \ell (\hat{\theta}, z')  $$

As with the rest of the methods based on calculating inverse-hessian-vector products, an important part of the
computations are carried out by objects from the class `InverseHessianVectorProduct`.

## Notebooks

- **WIP**

{{deel.influenciae.influence.second_order_influence_calculator.SecondOrderInfluenceCalculator}}
