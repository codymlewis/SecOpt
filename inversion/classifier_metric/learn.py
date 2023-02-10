from typing import Any, Tuple, NamedTuple
from argparse import ArgumentParser

import datasets
from flax import serialization
from flax.core.frozen_dict import unfreeze, freeze, FrozenDict
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import optax
import einops
from tqdm import trange

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


def transfer_vars(model_name, vars_template):
    with open(f"params/{model_name}.variables", "rb") as f:
        variables = serialization.from_bytes(vars_template, f.read())
    variables = unfreeze(variables)
    rootname = "DenseNet_0" if "DenseNet" in args.model else "ResNet_0"
    variables['params'][rootname]['predictions'] = vars_template['params'][rootname]['predictions']
    return freeze(variables)


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


def load_dataset(name: str) -> datasets.Dataset:
    match name:
        case "mnist": return load_mnist()
        case "cifar10": return load_cifar10()
        case "svhn": return load_svhn()
        case _: raise NotImplementedError(f"Dataset {name} is not implemented")


def load_mnist() -> datasets.Dataset:
    """
    Load the Fashion MNIST dataset http://arxiv.org/abs/1708.07747

    Arguments:
    - seed: seed value for the rng used in the dataset
    """
    ds = datasets.load_dataset("fashion_mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return ds


def load_cifar10() -> datasets.Dataset:
    """
    Load the CIFAR-10 dataset https://www.cs.toronto.edu/~kriz/cifar.html

    Arguments:
    - seed: seed value for the rng used in the dataset
    """
    ds = datasets.load_dataset("cifar10")
    ds = ds.map(
        lambda e: {
            'X': np.array(e['img'], dtype=np.float32) / 255,
            'Y': e['label']
        },
        remove_columns=['img', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(32, 32, 3), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return ds


def load_svhn() -> datasets.Dataset:
    """
    Load the SVHN dataset http://ufldl.stanford.edu/housenumbers/

    Arguments:
    - seed: seed value for the rng used in the dataset
    """
    ds = datasets.load_dataset("svhn", "cropped_digits")
    ds = ds.map(
        lambda e: {
            'X': np.array(e['image'], dtype=np.float32) / 255,
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(32, 32, 3), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return ds


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

    ds = load_dataset(args.dataset)
    X, Y = ds['train']['X'], ds['train']['Y']

    if args.model == "CNN":
        model = cnn.CNN(10)
        variables = model.init(jax.random.PRNGKey(42), X[:1])
        state = State.from_vars(variables)
        opt = optax.adam(1e-3)
    else:
        model = getattr(
            densenet if "DenseNet" in args.model else resnetv2,
            args.model
        )(10)
        vars_template = model.init(jax.random.PRNGKey(42), X[:1])
        variables = transfer_vars(args.model, vars_template)
        rootname = "DenseNet_0" if "DenseNet" in args.model else "ResNet_0"
        state = State.from_vars(variables)
        opt = optax.multi_transform(
            {
                k: optax.adam(1e-3) if k == "predictions" else optax.set_to_zero()
                for k in state.params[rootname].keys()
            },
            freeze({rootname: {k: k for k in state.params[rootname].keys()}})
        )
    opt_state = opt.init(state.params)
    trainer = train_step(opt, loss(model))
    rng = np.random.default_rng()
    train_len = len(X)

    for _ in (pbar := trange(args.steps)):
        idx = rng.choice(train_len, 32, replace=False)
        loss_val, state, opt_state = trainer(state, opt_state, X[idx], Y[idx])
        pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
    print(f"Final accuracy: {accuracy(model, state.to_vars(), ds['test']['X'], ds['test']['Y']):.3%}")

    with open((fn := f"{args.model}.{args.dataset}.variables"), 'wb') as f:
        f.write(serialization.to_bytes(variables))
    print(f"Written trained variables to {fn}")
