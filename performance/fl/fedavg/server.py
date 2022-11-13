"""
A server for federated learning.
"""

from typing import Any, NamedTuple, Tuple, Iterable, Optional
import jax
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
        num_adversaries: int = 0,
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
        self.num_adversaries = num_adversaries

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
        all_grads, all_states = [], []
        for c in self.clients:
            grads, state = c.update(params)
            all_grads.append(grads)
            all_states.append(state)
        meaned_grads = tree_mean(*all_grads)
        params = tree_add_scalar_mul(params, -1, meaned_grads)
        return params, State(np.mean([s.value for s in all_states]))


@jax.jit
def tree_mean(*trees: PyTree) -> PyTree:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum(ts) / len(trees), *trees)


@jax.jit
def tree_add_scalar_mul(tree_a: PyTree, mul: float, tree_b: PyTree) -> PyTree:
    """Add a scaler multiple of tree_b to tree_a"""
    return jax.tree_util.tree_map(lambda a, b: a + mul * b, tree_a, tree_b)