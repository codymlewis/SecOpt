"""
A standard federated learning client, optionally with model hardening.
"""

from typing import Any, Callable, Tuple, Iterator, Optional
import jax
import jaxopt
from optax import Params, Updates, GradientTransformation

from . import hardening as hardening_lib


State = Any


class Client:
    """Standard federated learning client with optional model hardening."""
    def __init__(
        self,
        params: Params,
        opt: GradientTransformation,
        loss_fun: Callable[[Params, jax.Array, jax.Array], float],
        data: Iterator,
        epochs: int = 1,
        hardening: Optional[hardening_lib.Hardening] = None
    ):
        """
        Initialize the client

        Parameters:
        - params: Model parameters
        - opt: Optimizer algorithm
        - loss_fun: loss function
        - data: Iterator containing this client's data
        - epochs: Number of local epochs of training to perform in each round
        - hardening: Hardening function to apply during model training
        """
        self.params = params
        self.solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=epochs)
        self.state = self.solver.init_state(params)
        self.step = jax.jit(self.solver.update)
        self.data = data
        if hardening is None:
            self.hardening = hardening_lib.default_hardening()
        else:
            self.hardening = hardening

    def update(self, global_params: Params) -> Tuple[Updates, State]:
        """
        Perform local training for this round and return the resulting gradient and state

        Parameters:
        - global_params: Global parameters downloaded for this round of training
        """
        self.params = global_params
        for e in range(self.solver.maxiter):
            X, Y = next(self.data)
            X, Y = self.hardening.update(self.params, X, Y)
            self.params, self.state = self.step(
                params=self.params, state=self.state, X=X, Y=Y
            )
        return jaxopt.tree_util.tree_sub(global_params, self.params), self.state
