"""
A standard federated learning client, optionally with model hardening.
"""

from typing import Any, Callable, Tuple, Iterator, Optional, NamedTuple
import jax
import jaxopt
from optax import Params, Updates, GradientTransformation

from . import hardening as hardening_lib


PyTree = Any


class Client:
    """Standard federated learning client with optional model hardening."""
    def __init__(
        self,
        model,
        params: Params,
        opt: GradientTransformation,
        loss_fun: Callable[[Params, jax.Array, jax.Array], float],
        data: Iterator,
        epochs: int = 1,
        hardening: Optional[str] = None,
        lr: float = 0.01,
        eps: float = 1e-8,
        b1: float = 0.9,
        b2: float = 0.999,
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
        if hardening is None or hardening == "none":
            self.hardening = hardening_lib.default_hardening()
        elif hardening == "pgd":
            self.hardening = hardening_lib.pgd(loss_fun)
        else:
            self.hardening = hardening_lib.flip(model, data)
        self.lr = lr
        self.eps = eps
        self.b1 = b1
        self.b2 = b2

    def update(self, global_params: Params) -> Tuple[Updates, Updates, NamedTuple]:
        """
        Perform local training for this round and return the resulting gradient and state

        Arguments:
        - global_params: Global parameters downloaded for this round of training
        """
        self.params = global_params
        for e in range(self.solver.maxiter):
            X, Y = next(self.data)
            X, Y = self.hardening.update(self.params, X, Y)
            self.params, self.state = self.step(
                params=self.params, state=self.state, X=X, Y=Y
            )
        m, v = self.state.internal_state[0].mu, self.state.internal_state[0].nu
        m = tree_scale(m, self.lr)
        count = self.state.internal_state[0].count
        m_hat = bias_correction(m, self.b1, count)
        v_hat = bias_correction(v, self.b2, count)
        return m_hat, tree_add_scalar(v_hat, self.eps**2), self.state


@jax.jit
def tree_scale(tree: PyTree, scalar: float) -> PyTree:
    """Take the element-wise square root of a pytree"""
    return jax.tree_util.tree_map(lambda t: t * scalar, tree)


@jax.jit
def tree_add_scalar(tree: PyTree, scalar: float) -> PyTree:
    """Take the element-wise square root of a pytree"""
    return jax.tree_util.tree_map(lambda t: t + scalar, tree)


@jax.jit
def bias_correction(moment, decay, count):
    bias_correction_ = 1 - decay**count
    return jax.tree_util.tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)
