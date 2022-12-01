from typing import Callable
import jax
from jax import Array
import jax.numpy as jnp
from optax import Params


def pgd(
    loss: Callable[[Params, Array, Array], float],
    epsilon: float = 0.3,
    lr: float = 0.001,
    steps: int = 1
) -> Callable[[Params, Array, Array], Array]:
    """Projected gradient descent, proposed in https://arxiv.org/abs/1706.06083"""

    @jax.jit
    def update(params: Params, X: Array, Y: Array) -> Array:
        """
        Apply noise to the input data according to gradient ascent.

        Parameters:
        - params: Model parameters
        - X: Sample features
        - Y: Sample labels
        """
        X_nat = X
        for _ in range(steps):
            grads = jax.grad(loss, argnums=1)(params, X, Y)
            X = X + lr * jnp.sign(grads)
            X = jnp.clip(X, X_nat - epsilon, X_nat + epsilon)
            X = jnp.clip(X, 0, 1)
        return X

    return update
