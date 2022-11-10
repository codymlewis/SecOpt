"""
Model hardening algorithms to be used during training so models can better resist adversarial examples.
"""

from typing import Any, NamedTuple, Callable, Tuple
from optax import Params
from jax import Array
import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


class Hardening(NamedTuple):
    """Generic class defining the structure of hardening algorithms"""
    update: Callable[[Params, Array, Array], Tuple[Array, Array]]
    """
    Hardening update function.
    Takes the model parameters, X, and Y as input and outputs a modified X and Y.
    """


def default_hardening() -> Hardening:
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
) -> Hardening:
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


def flip(
    model,
    data,
    steps: int=50
) -> Hardening:
    distances = np.full((data.classes, data.classes), np.inf)
    rng = data.rng
    unique_Y = np.unique(data.Y)

    def update(params, _X, _Y):
        @jax.jit
        def _apply(params, X):
            return jnp.argmax(model.apply(params, X), axis=-1)

        pair_dists = distances + distances.T
        if pair_dists.min() == np.inf:
            if ((distances * ~np.eye(data.classes).astype(bool))[unique_Y] == np.inf).any():
                a = rng.choice(unique_Y)
            else:
                a = np.where(distances == distances.min())[0][0]
            promising_pair = False
        else:
            a, b = np.where(pair_dists == pair_dists.min())
            a, b = rng.choice(a), rng.choice(b)
            promising_pair = np.sum(data.Y == a) > 5 and np.sum(data.Y == b) > 5
    
        if promising_pair:
            idx = rng.choice(
                np.where((data.Y == a) | (data.Y == b))[0],
                min(data.batch_size, np.sum(data.Y == a)),
                replace=False
            )
            X, Y = data.X[idx], data.Y[idx]
            orig_X, orig_Y = X, Y

            m = rng.uniform(0, 1, size=(2,) + X[0].shape)
            delta = rng.uniform(0, 1, size=X[0].shape)
            p = Y == a
            for i in range(steps):
                X_a = (p * ((1 - m[0]) * X + m[0] * delta).T).T
                X_b = ((1 - p) * ((1 - m[1]) * X + m[1] * delta).T).T
                X = X_a + X_b
                changed_Y = _apply(params, X)
                if (changed_Y == b).all():
                    distances[Y][b] = i * np.linalg.norm(m)
                    return np.concatenate((X, orig_X)), np.concatenate((Y, orig_Y))
            distances[Y, changed_Y] = steps * np.linalg.norm(m)
            return np.concatenate((X, orig_X)), np.concatenate((Y, orig_Y))
        else:
            idx = rng.choice(np.where(data.Y == a)[0], min(data.batch_size, np.sum(data.Y == a)), replace=False)
            X, Y = data.X[idx], data.Y[idx]
            orig_X, orig_Y = X, Y
            m = rng.uniform(0, 1, size=X[0].shape)
            delta = rng.uniform(0, 1, size=X[0].shape)
            p = Y == a
            for i in range(steps):
                X = (1 - m) * X + m * delta
                changed_Y = _apply(params, X)
                if (changed_Y != Y).all():
                    distances[Y, changed_Y] = i * np.linalg.norm(m)
                    return np.concatenate((X, orig_X)), np.concatenate((Y, orig_Y))
            distances[Y, changed_Y] = steps * np.linalg.norm(m)
            return np.concatenate((X, orig_X)), np.concatenate((Y, orig_Y))

    return Hardening(update=update)
