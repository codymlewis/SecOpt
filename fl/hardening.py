"""
Model hardening algorithms to be used during training so models can better resist adversarial examples.
"""

from typing import NamedTuple, Callable, Tuple
from optax import Params
from jax import Array
import jax
import jax.numpy as jnp


class Hardening(NamedTuple):
    """Generic class defining the structure of hardening algorithms"""
    update: Callable[[Params, Array, Array], Tuple[Array, Array]]
    """
    Hardening update function.
    Takes the model parameters, X, and Y as input and outputs a modified X and Y.
    """


def default_hardening():
    """No hardening, just the identity function"""

    def update(_params: Params, X: Array, Y: Array) -> Tuple[Array, Array]:
        """
        Does not modify the data

        Parameters:
        - params: Model parameters
        - X: Sample features
        - Y: Sample labels
        """
        return X, Y
    
    return Hardening(update=update)


def pgd(
    loss: Callable[[Params, Array, Array], Tuple[Array, Array]],
    epsilon: float=0.3,
    lr: float=0.001,
    steps: int=50
):
    """Projected gradient descent, proposed in https://arxiv.org/abs/1706.06083"""

    @jax.jit
    def update(params: Params, X: Array, Y: Array) -> Tuple[Array, Array]:
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
        return X, Y

    return Hardening(update=update)
