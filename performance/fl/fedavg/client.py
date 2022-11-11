"""
A standard federated learning client, with PGD hardening
"""

from typing import Callable, Tuple, Iterator, NamedTuple
import jax
import jaxopt
from optax import Params, Updates, GradientTransformation


class Client:
    """Standard federated learning client with optional model hardening."""

    def __init__(
        self,
        uid: int,
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
        self.id = uid
        self.solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=epochs)
        self.state = self.solver.init_state(params)
        self.step = jax.jit(self.solver.update)
        self.data = data

    def update(self, global_params: Params) -> Tuple[Updates, NamedTuple]:
        """
        Perform local training for this round and return the resulting gradient and state

        Parameters:
        - global_params: Global parameters downloaded for this round of training
        """
        params = global_params
        for e in range(self.solver.maxiter):
            X, Y = next(self.data)
            params, self.state = self.step(
                params=params, state=self.state, X=X, Y=Y
            )
        gradient = jaxopt.tree_util.tree_sub(global_params, params)
        del params
        return gradient, self.state
