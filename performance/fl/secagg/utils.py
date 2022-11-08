from functools import partial
import jax
from jax import Array
from optax import Params


@partial(jax.jit, static_argnums=(0, 1, 2))
def gen_mask(key, params_len, R):
    return jax.random.uniform(jax.random.PRNGKey(key), (params_len,), minval=-R, maxval=R)


@jax.jit
def ravel(params: Params) -> Array:
    return jax.flatten_util.ravel_pytree(params)[0]
