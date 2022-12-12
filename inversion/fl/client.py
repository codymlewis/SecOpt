"""
A standard federated learning client, with PGD hardening
"""

from typing import Any, Callable, Tuple, Iterator, NamedTuple
from jax import Array
import jax
import jax.numpy as jnp
import jaxopt
from optax import Params, Updates, GradientTransformation
from numpy.typing import NDArray

PyTree = Any


class Client:
    """Standard federated learning client with optional model hardening."""
    def __init__(
        self,
        params: Params,
        opt: GradientTransformation,
        loss_fun: Callable[[Params, jax.Array, jax.Array], float],
        data: Iterator,
        epochs: int = 1,
    ):
        """
        Initialize the client

        Parameters:
        - params: Model parameters
        - opt: Optimizer algorithm
        - loss_fun: loss function
        - data: Iterator containing this client's data
        - epochs: Number of local epochs of training to perform in each round
        """
        self.params = params
        self.loss_fun = loss_fun
        self.solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=epochs)
        self.state = self.solver.init_state(params)
        self.step = jax.jit(self.solver.update)
        self.data = data
        self.hardening = pgd(loss_fun, lr=0.001)

    def update(self, global_params: Params) -> Tuple[Updates, NamedTuple]:
        """
        Perform local training for this round and return the resulting gradient and state

        Parameters:
        - global_params: Global parameters downloaded for this round of training
        """
        self.params = global_params
        for e in range(self.solver.maxiter):
            X, Y = next(self.data)
            X = self.hardening(self.params, X, Y)
            self.params, self.state = self.step(
                params=self.params, state=self.state, X=X, Y=Y
            )
        return jaxopt.tree_util.tree_sub(global_params, self.params), self.state

    def get_update(self, global_params: Params) -> Tuple[Updates, NDArray, NDArray]:
        """
        Perform a single update and return the changes and the respective data

        Arguments:
        - global_params: Global parameters downloaded for this round of training
        """
        X, Y = next(self.data)
        params, _ = self.step(params=global_params, state=self.state, X=X, Y=Y)
        return jaxopt.tree_util.tree_sub(global_params, self.params), X, Y


class DPClient(Client):
    def __init__(
        self,
        *args,
        clipping_rate: float = 5,
        noise_scale: float = 0.0001,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rng = self.data.rng
        self.clipping_rate = clipping_rate
        self.noise_scale = noise_scale

    def update(self, global_params):
        grads, state = super().update(global_params)
        grads = clip(grads, self.clipping_rate)
        grads = noise(grads, self.noise_scale, self.rng)
        return grads, state


    def get_update(self, global_params):
        grads, X, Y = super().get_update(global_params)
        grads = clip(grads, self.clipping_rate)
        grads = noise(grads, self.noise_scale, self.rng)
        return grads, X, Y


@jax.jit
def clip(grads, clipping_rate):
    norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads)[0])
    return jax.tree_map(lambda g: g / jnp.maximum(1, norm / clipping_rate), grads)


def noise(grads, scale, rng):
    return jax.tree_map(lambda g: g + rng.normal(0, scale, size=g.shape), grads)


class AdamClient(Client):
    def __init__(
        self,
        *args,
        lr: float = 0.001,
        eps: float = 1e-8,
        perturb_data: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hardening = pgd(self.loss_fun, lr=lr / 100)
        self.lr = lr
        self.eps = eps
        self.data.perturb_data = perturb_data

    def update(self, global_params: Params) -> Tuple[Updates, Updates, NamedTuple]:
        _, state = super().update(global_params)
        m, n = state.internal_state[0].mu, state.internal_state[0].nu
        return tree_mul_scalar(m, self.lr), tree_add_scalar(n, self.eps), state

    def get_update(self, global_params: Params) -> Tuple[Updates, NDArray, NDArray]:
        self.data.get_unperturbed = True
        X, Y, uX = next(self.data)
        params, _ = self.step(params=global_params, state=self.state, X=X, Y=Y)
        grads =  jaxopt.tree_util.tree_sub(global_params, self.params)
        self.data.get_unperturbed = False
        return grads, uX, Y



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


@jax.jit
def tree_mul_scalar(tree: PyTree, scalar: float) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x * scalar, tree)


@jax.jit
def tree_add_scalar(tree: PyTree, scalar: float) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x + scalar, tree)
