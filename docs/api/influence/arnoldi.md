# Arnoldi Influence Calculator

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>
[View source](https://github.com/deel-ai/influenciae/blob/main/deel/influenciae/influence/arnoldi_influence_calculator.py) |
📰 [Paper](https://arxiv.org/abs/2112.03052)

This class implements the method introduced in *Scaling Up Influence Functions, Schioppa et al.* at AAAI 2022.
It proposes a series of memory and computational optimizations based on the Arnoldi iteration for speeding up
inverse hessian calculators, allowing the authors to approximately compute influence functions on whole 
large vision models (going up to a ViT-L with 300M parameters).

In essence, the optimizations can be summarized as follows:
- build an orthonormal basis for the Krylov subspaces of a random vector (in the desired dimensionality).
- find the eigenvalues and eigenvectors of the restriction of the Hessian matrix in that restricted subspace.
- keep only the $k$ largest eigenvalues and their corresponding eigenvectors, and create a projection matrix $G$ into this space.
- use forward-over-backward auto-differentiation to directly compute the JVPs in this reduced space.

Due to the specificity of these optimizations, the inverse hessian vector product operation is implemented inside the
class, and thus, doesn't require an additional separate IHVP object. In addition, it can only be applied to individual
points for the moment.


## Notebooks

- [**Using Arnoldi Influence Calculator**](https://colab.research.google.com/drive/1rQU33sbD0YW1cZMRlJmS15EW5O16yoDE?usp=sharing)

{{deel.influenciae.influence.first_order_influence_calculator.ArnoldiInfluenceCalculator}}
