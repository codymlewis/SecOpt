from typing import NamedTuple
import jax
import jax.numpy as jnp
import chex
import optax


def secadam(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    return optax.adam(learning_rate, b1, b2, eps=0.0, eps_root=eps**2)


def dpsecadam(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    clip_threshold: float = 1.0,
    noise_scale: float = 0.1,
    seed=0,
) -> optax.GradientTransformation:
    return optax.chain(
        clip(clip_threshold),
        add_noise(noise_scale, seed),
        secadam(learning_rate, b1, b2, eps),
    )


def dpsgd(
    learning_rate: optax.ScalarOrSchedule,
    clip_threshold: float = 1.0,
    noise_scale: float = 0.1,
    seed=0,
) -> optax.GradientTransformation:
    return optax.chain(
        clip(clip_threshold),
        add_noise(noise_scale, seed),
        optax.sgd(learning_rate),
    )


def clip(clip_threshold: float = 1.0) -> optax.GradientTransformation:
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        squared_grads = jax.tree_util.tree_map(lambda g: jnp.sum(g**2), updates)
        norm = jnp.sqrt(jax.tree_util.tree_reduce(lambda G, g: G + g, squared_grads))
        updates = jax.tree_util.tree_map(lambda g: g / jnp.maximum(1, norm / clip_threshold), updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class AddNoiseState(NamedTuple):
    """State for adding gradient noise."""
    rng_key: chex.PRNGKey


def add_noise(
    noise_scale: float = 0.1,
    seed: int = 0,
) -> optax.GradientTransformation:

    def init_fn(params):
        del params
        return AddNoiseState(rng_key=jax.random.PRNGKey(seed))

    def update_fn(updates, state, params=None):
        del params
        num_vars = len(jax.tree_util.tree_leaves(updates))
        treedef = jax.tree_util.tree_structure(updates)

        all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype) * noise_scale,
            updates, jax.tree_util.tree_unflatten(treedef, all_keys[1:])
        )
        updates = jax.tree_util.tree_map(
            lambda g, n: g + n,
            updates, noise
        )
        return updates, AddNoiseState(rng_key=all_keys[0])

    return optax.GradientTransformation(init_fn, update_fn)