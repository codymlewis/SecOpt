"""
A server for federated learning.
"""

from functools import partial
from typing import Any, NamedTuple, Tuple, Iterable, Optional
import jax
import jax.numpy as jnp
from optax import Params
import numpy as np

from .client import Client


PyTree = Any


class State(NamedTuple):
    """A simple global state class"""
    value: float
    """The result of the function being learned"""


class Server:
    """Federated averaging server."""
    def __init__(
        self,
        params: Params,
        clients: Iterable[Client],
        maxiter: int = 5,
        C: float = 1.0,
        num_adversaries: int = 0,
        noise_clip: bool = False,
        ensure_adversaries = True,
        seed: Optional[int] = None
    ):
        """
        Parameters:
        - params: Model parameters
        - clients: Iterable containing the client objects
        - maxiter: Number of rounds of training to perform
        - C: fraction of clients to select each round
        - num_adversaries: number of adversarial clients
        - seed: seed for the rng used for client selection
        """
        self.params = params
        self.clients = clients
        self.maxiter = maxiter
        self.rng = np.random.default_rng(seed)
        self.C = C
        self.K = len(clients)
        self.num_adversaries = num_adversaries
        self.noise_clip = noise_clip
        self.ensure_adversaries = ensure_adversaries

    def init_state(self, params: Params) -> State:
        """Initialize the server state"""
        return State(np.inf)

    def update(self, params: Params, state: State) -> Tuple[Params, State]:
        """
        Perform a round of training
        Parameters:
        - params: Model parameters
        - state: Server state
        """
        all_ms, all_vs, all_states = [], [], []
        if self.C < 1:
            if self.ensure_adversaries:
                idx = self.rng.chice(
                    self.K - self.num_adversaries, size=int(self.C * self.K - self.num_adversaries)
                )
                idx = np.concatenate((idx, range(self.K - self.num_adversaries, self.K)))
            else:
                idx = self.rng.chice(self.K, size=int(self.C * self.K))
        else:
            idx = range(self.K)
        for i in idx:
            m, v, state = self.clients[i].update(params)
            all_ms.append(m)
            all_vs.append(v)
            all_states.append(state)
        meaned_ms = tree_mean(*all_ms)
        meaned_vs = tree_mean(*all_vs)
        sqrt_vs = tree_sqrt(meaned_vs)
        meaned_grads = tree_div(meaned_ms, sqrt_vs)
        if self.noise_clip:
            meaned_grads = clip_and_noise(meaned_grads, 0.01, 1e-5, self.rng)
        params = tree_add_scalar_mul(params, -1, meaned_grads)
        return params, State(np.mean([s.value for s in all_states]))


@jax.jit
def tree_mean(*trees: PyTree) -> PyTree:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum(ts) / len(trees), *trees)


@jax.jit
def tree_sqrt(tree: PyTree) -> PyTree:
    """Take the element-wise square root of a pytree"""
    return jax.tree_util.tree_map(lambda t: jnp.sqrt(t), tree)


@jax.jit
def tree_div(tree_a: PyTree, tree_b: PyTree) -> PyTree:
    """Perform the element-wise division of tree_a and tree_b"""
    return jax.tree_util.tree_map(lambda a, b: a / b, tree_a, tree_b)


@partial(jax.jit, static_argnums=(1, 2, 3))
def clip_and_noise(tree: PyTree, gamma: float, scale: float, rng: np.random.Generator) -> PyTree:
    norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(tree)[0], ord=2)
    clip = jnp.minimum(1, gamma / norm)
    return jax.tree_util.tree_map(
        lambda t: t * clip  + rng.normal(0, scale, size=t.shape),
        tree
    )


@jax.jit
def tree_add_scalar_mul(tree_a: PyTree, mul: float, tree_b: PyTree) -> PyTree:
    """Add a scaler multiple of tree_b to tree_a"""
    return jax.tree_util.tree_map(lambda a, b: a + mul * b, tree_a, tree_b)
