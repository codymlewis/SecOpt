"""
A server for federated learning.
"""

from typing import Any, NamedTuple, Tuple, Iterable, Optional
import jax
import optax
from optax import Params
import numpy as np

from .client import Client


PyTree = Any


class State(NamedTuple):
    """A simple global state class"""
    value: float
    """The result of the function being learned"""
    opt_state: optax.OptState


class Server:
    """Federated averaging server."""

    def __init__(
        self,
        params: Params,
        clients: Iterable[Client],
        maxiter: int = 5,
        num_adversaries: int = 0,
        seed: Optional[int] = None,
        efficient: bool = False,
        optimizer: optax.GradientTransformation = optax.sgd(1)
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
        self.efficient = efficient
        self.opt = optimizer

    def init_state(self, params: Params) -> State:
        """Initialize the server state"""
        opt_state = self.opt.init(params)
        return State(np.inf, opt_state)

    def update(self, params: Params, server_state: State) -> Tuple[Params, State]:
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
        updates, opt_state = self.opt.update(meaned_grads, server_state.opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, State(np.mean([s.value for s in all_states]), opt_state)


@jax.jit
def tree_mean(*trees: PyTree) -> PyTree:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum(ts) / len(trees), *trees)
