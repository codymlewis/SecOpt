from typing import Any, Tuple, NamedTuple
from argparse import ArgumentParser

from flax import serialization
from flax.core.frozen_dict import freeze, FrozenDict
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import optax
from tqdm import trange


import datalib
import cnn
import densenet
import resnetv2

PyTree = Any


class State(NamedTuple):
    params: FrozenDict
    batch_stats: FrozenDict

    def to_vars(self):
        return freeze(self._asdict())

    def from_vars(variables):
        return State(params=variables['params'], batch_stats=variables['batch_stats'])


def loss(model):
    @jax.jit
    def _loss(variables, X, Y, scale=False):
        logits, batch_stats = model.apply(variables, X, mutable=['batch_stats'])
        logits = jnp.clip(logits, 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits))), batch_stats['batch_stats']
    return _loss


def accuracy(model, variables, X, Y, batch_size=1000):
    """Accuracy metric using batch size to prevent OOM errors"""
    @jax.jit
    def _apply(batch_X: Array | Tuple[Array, Array]) -> Array:
        return jnp.argmax(model.apply(variables, batch_X, train=False), axis=-1)

    acc = 0
    ds_size = len(Y)
    for i in range(0, ds_size, batch_size):
        end = min(i + batch_size, ds_size)
        logits = _apply(X[i:end])
        acc += jnp.mean(logits == Y[i:end])
    return acc / jnp.ceil(ds_size / batch_size)


def train_step(opt, loss):
    @jax.jit
    def apply(state, opt_state, X, Y):
        (loss_val, batch_stats), grads = jax.value_and_grad(loss, has_aux=True)(state.to_vars(), X, Y)
        updates, opt_state = opt.update(grads['params'], opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return loss_val, State(params=params, batch_stats=batch_stats), opt_state
    return apply


if __name__ == "__main__":
    parser = ArgumentParser(description="Train the model for the confidence metric")
    parser.add_argument('-s', '--steps', type=int, default=1563, help="Number of steps to train for.")
    parser.add_argument('-d', '--dataset', type=str, default="cifar10", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="DenseNet121", help="Model to train.")
    args = parser.parse_args()

    ds = datalib.load_dataset(args.dataset)
    X, Y = ds['train']['X'], ds['train']['Y']

    if args.model == "CNN":
        model = cnn.CNN(10)
    else:
        model = getattr(
            densenet if "DenseNet" in args.model else resnetv2,
            args.model
        )(10)
    variables = model.init(jax.random.PRNGKey(42), X[:1])
    state = State.from_vars(variables)
    opt = optax.yogi(1e-3)
    opt_state = opt.init(state.params)
    trainer = train_step(opt, loss(model))
    rng = np.random.default_rng()
    train_len = len(X)

    for _ in (pbar := trange(args.steps)):
        idx = rng.choice(train_len, 128 if "BC" in args.model else 512, replace=False)
        loss_val, state, opt_state = trainer(state, opt_state, X[idx], Y[idx])
        pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
    print(f"Final accuracy: {accuracy(model, state.to_vars(), ds['test']['X'], ds['test']['Y']):.3%}")

    with open((fn := f"{args.model}.{args.dataset}.variables"), 'wb') as f:
        f.write(serialization.to_bytes(variables))
    print(f"Written trained variables to {fn}")
