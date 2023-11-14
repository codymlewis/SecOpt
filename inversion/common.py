import jax
import jax.numpy as jnp
from sklearn import metrics
import optax

import optimisers


@jax.jit
def update_step(state, X, Y):
    def loss_fn(params):
        logits = jnp.clip(state.apply_fn(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


@jax.jit
def pgd_update_step(state, X, Y, pgd_lr=0.001, epsilon=1/32, pgd_steps=10):
    def loss_fn(params, dX):
        logits = jnp.clip(state.apply_fn(params, dX), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))

    X_nat = X
    for _ in range(pgd_steps):
        Xgrads = jax.grad(loss_fn, argnums=1)(state.params, X)
        X = X + pgd_lr * jnp.sign(Xgrads)
        X = jnp.clip(X, X_nat - epsilon, X_nat + epsilon)
        X = jnp.clip(X, 0, 1)

    loss, grads = jax.value_and_grad(loss_fn)(state.params, X)
    state = state.apply_gradients(grads=grads)
    return loss, state


def accuracy(state, X, Y, batch_size=1000):
    """
    Calculate the accuracy of the model across the given dataset

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    - variables: Parameters and other learned values used by the model
    - X: The samples
    - Y: The corresponding labels for the samples
    - batch_size: Amount of samples to compute the accuracy on at a time
    """
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(state.apply_fn(state.params, batch_X), axis=-1)

    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    return metrics.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


def find_optimiser(opt_name):
    try:
        return getattr(optimisers, opt_name)
    except AttributeError:
        return getattr(optax, opt_name)
