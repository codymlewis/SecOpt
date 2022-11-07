"""
A standard federated learning client
"""

from typing import Callable, Tuple, Iterator, NamedTuple
import jax
import jaxopt
from optax import Params, Updates, GradientTransformation
from numpy.typing import NDArray



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
        self.params = global_params
        for e in range(self.solver.maxiter):
            X, Y = next(self.data)
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
