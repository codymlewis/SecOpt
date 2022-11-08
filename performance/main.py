import os
from typing import Any, Callable, Iterable, Tuple
from argparse import ArgumentParser
import einops
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
import optax
import datasets
from sklearn import metrics
import numpy as np
from tqdm import trange
import pandas as pd


import fl

PyTree = Any


def loss(model: nn.Module) -> Callable[[PyTree, Array, Array], float]:
    """
    A cross-entropy loss function

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    """
    @jax.jit
    def _apply(params: PyTree, X: Array, Y: Array) -> float:
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def accuracy(model: nn.Module, variables: PyTree, ds: Iterable[Tuple[Array|Tuple[Array, Array], Array]]):
    """
    Calculate the accuracy of the model across the given dataset

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    - variables: Parameters and other learned values used by the model
    - ds: Iterable data over which the accuracy is calculated
    """
    @jax.jit
    def _apply(batch_X: Array|Tuple[Array, Array]) -> Array:
        return jnp.argmax(model.apply(variables, batch_X), axis=-1)
    preds, Ys = [], []
    for X, Y in ds:
        preds.append(_apply(X))
        Ys.append(Y)
    return metrics.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


class LeNet(nn.Module):
    """The LeNet-300-100 network from https://doi.org/10.1109/5.726791"""
    @nn.compact
    def __call__(self, x: Array, representation: bool = False) -> Array:
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)


def load_dataset(name: str, seed: int) -> fl.data.Dataset:
    match name:
        case "mnist": return load_mnist(seed)
        case "cifar10": return load_cifar10(seed)
        case "svhn": return load_svhn(seed)
        case _: raise NotImplementedError(f"Dataset {name} is not implemented")


def load_mnist(seed: int) -> fl.data.Dataset:
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
    return fl.data.Dataset("mnist", ds, seed)


def load_cifar10(seed: int) -> fl.data.Dataset:
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
    return fl.data.Dataset("cifar10", ds, seed)


def load_svhn(seed: int) -> fl.data.Dataset:
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
    return fl.data.Dataset("cifar10", ds, seed)


def load_agg_module(name: str) -> Tuple[Any, Any]:
    match name:
        case "fedavg": return fl.fedavg
        case "secagg": return fl.secagg
        case "nerv": return fl.nerv
        case _: raise NotImplementedError(f"Aggregation method {name} has not been implmented.")

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Experiments looking at the performance of different aggregation algorithms."
    )
    parser.add_argument('-b', '--batch-size', type=int, default=32, help="Size of batches for training.")
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="densenet", help="Model to train.")
    parser.add_argument('-n', '--num-clients', type=int, default=10, help="Number of clients to train with.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for the RNG.")
    parser.add_argument('-r', '--rounds', type=int, default=3000, help="Number of rounds to train for.")
    parser.add_argument('-a', '--aggregation', type=str, default="fedavg", help="Aggregation algorithm to use.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.seed)
    data = dataset.fed_split([args.batch_size for _ in range(args.num_clients)], fl.data.lda)
    agg = load_agg_module(args.aggregation)
    model = LeNet()
    # model = models.load_model(args.model)
    params = model.init(jax.random.PRNGKey(args.seed), dataset.input_init)
    clients = [
        agg.client.Client(
            i,
            params,
            optax.sgd(0.1),
            loss(model),
            d
        )
        for i, d in enumerate(data)
    ]
    server = agg.server.Server(params, clients, maxiter=args.rounds, seed=args.seed)
    state = server.init_state(params)

    for _ in (pbar := trange(server.maxiter)):
        params, state = server.update(params, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
    test_data = dataset.get_test_iter(args.batch_size)
    final_acc = accuracy(model, params, test_data)
    print(f"Final accuracy: {final_acc:.3%}")

    experiment_results = vars(args).copy()
    experiment_results['Final accuracy'] = final_acc.item()
    df_results = pd.DataFrame(data=experiment_results, index=[0])
    if os.path.exists('results.csv'):
        old_results = pd.read_csv('results.csv')
        df_results = pd.concat((old_results, df_results))
    df_results.to_csv('results.csv', index=False)
    print("Done.")
