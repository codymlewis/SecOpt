import os
from functools import partial
import operator
from typing import Any, Callable, Iterable, Tuple
from numpy.typing import ArrayLike
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
import jaxopt
import skimage
import pandas as pd
import matplotlib.pyplot as plt

import fl
import models


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
    return fl.data.Dataset("svhn", ds, seed)


def cosine_dist(A: Array, B: Array) -> float:
    """Get the cosine disance between two arrays"""
    denom = jnp.maximum(jnp.linalg.norm(A, axis=1) * jnp.linalg.norm(B, axis=1), 1e-15)
    return 1 - jnp.mean(jnp.abs(jnp.einsum('br,br -> b', A, B)) / denom)


def total_variation(V: Array) -> float:
    """Get the total variation of a batch of images"""
    return abs(V[:, 1:, :] - V[:, :-1, :]).sum() + abs(V[:, :, 1:] - V[:, :, :-1]).sum()


def reploss(
    model: nn.Module, params: PyTree, true_reps: ArrayLike, lamb_tv: float = 5e-4
) -> Callable[[Array], float]:
    """Loss function for the representation gradient inversion attack."""
    def _apply(Z: Array) -> float:
        dist = cosine_dist(model.apply(params, Z, representation=True), true_reps)
        return dist + lamb_tv * total_variation(Z)
    return _apply


def idlgloss(
    model: nn.Module, params: PyTree, true_grads: ArrayLike, Y: Array
) -> Callable[[Array], float]:
    loss_fn = loss(model)
    """Loss function for the representation gradient inversion attack."""
    def _apply(Z: Array) -> float:
        norm_tree = jax.tree_map(lambda a, b: jnp.sum((a - b)**2), jax.grad(loss_fn)(params, Z, Y), true_grads)
        return jax.tree_util.tree_reduce(operator.add, norm_tree)
    return _apply


def evaluate_inversion(X: ArrayLike, Y: ArrayLike, Z: ArrayLike, labels: ArrayLike) -> Tuple[float, float]:
    """Quantitatively evaluate the quality of a gradient inversion."""
    X = X[Y.argsort()]
    Y.sort()
    if len(Y) > 1:
        non_rep_labels = np.arange(np.max(Y) + 1)[np.bincount(Y) == 1]
        zidx = np.where(np.isin(labels, Y) & np.isin(labels, non_rep_labels))[0]
        xidx = np.where(np.isin(Y, labels) & np.isin(Y, non_rep_labels))[0]
    else:
        zidx = np.array([0])
        xidx = np.array([0])
    if not len(zidx) or not len(xidx):
        return None, None
    psnrs, ssims = [], []
    for zi, xi in zip(zidx, xidx):
        psnrs.append(skimage.metrics.peak_signal_noise_ratio(Z[zi], X[xi], data_range=1))
        ssims.append(skimage.metrics.structural_similarity(Z[zi], X[xi], win_size=11, channel_axis=2))
    return np.mean(psnrs), np.mean(ssims)


def inversion_images(X: ArrayLike, Y: ArrayLike, Z: ArrayLike, labels: ArrayLike, client_id: int):
    """Create images showing the effectiveness of the inversion."""
    X = X[Y.argsort()]
    Y.sort()
    non_rep_labels = np.arange(np.max(Y) + 1)[np.bincount(Y) == 1]
    zidx = np.where(np.isin(labels, Y) & np.isin(labels, non_rep_labels))[0]
    xidx = np.where(np.isin(Y, labels) & np.isin(Y, non_rep_labels))[0]
    if not len(zidx) or not len(xidx):
        return
    for zi, xi in zip(zidx, xidx):
        plt.imshow(Z[zi], cmap='gray')
        plt.axis('off')
        plt.savefig(f'client_{client_id}_Z{zi}_label{labels[zi]}', dpi=320)
        plt.clf()
        plt.imshow(X[xi], cmap='gray')
        plt.axis('off')
        plt.savefig(f'client_{client_id}_ground_truth{xi}_Y{Y[xi]}', dpi=320)
        plt.clf()


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiments looking at adversarial training against backdoor attacks.")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help="Size of batches for training.")
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="lenet", help="Model to train.")
    parser.add_argument('-n', '--num-clients', type=int, default=10, help="Number of clients to train with.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for the RNG.")
    parser.add_argument('-r', '--rounds', type=int, default=3000, help="Number of rounds to train for.")
    parser.add_argument('-o', '--opt', type=str, default="sgd", help="Optimizer to use.")
    parser.add_argument('--perturb', action="store_true", help="Perturb client data if using adam optimizer.")
    parser.add_argument('--dp', nargs=2, type=float, default=None, help="Use client side DP args: <clipping_rate> <noise_scale>.")
    parser.add_argument('--idlg', action="store_true", default=None, help="Use perform the iDLG attack.")
    parser.add_argument('--gen-images', action="store_true", help="Generate images from the inversion.")
    args = parser.parse_args()

    seed = round(args.seed * np.pi) + 500

    dataset = load_dataset(args.dataset, seed)
    data = dataset.fed_split([args.batch_size for _ in range(args.num_clients)], fl.distributions.lda)
    model = models.load_model(args.model)
    params = model.init(jax.random.PRNGKey(seed), dataset.input_init)
    if args.dp is not None:
        Client = partial(fl.client.DPClient, clipping_rate=args.dp[0], noise_scale=args.dp[1])
        Server = fl.server.Server
    elif args.opt.lower() != "ours":
        Client = fl.client.Client
        Server = fl.server.Server
    else:
        Client = partial(fl.client.AdamClient, perturb_data=args.perturb)
        Server = fl.server.AdamServer

    clients = [
        Client(
            params,
            optax.sgd(0.1) if args.opt.lower() == "sgd" else optax.adam(0.001),
            loss(model),
            d,
        )
        for d in data
    ]
    server = Server(params, clients, maxiter=args.rounds, seed=seed)
    state = server.init_state(params)
    for _ in (pbar := trange(server.maxiter)):
        params, state = server.update(params, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
    test_data = dataset.get_test_iter(args.batch_size)
    final_acc = accuracy(model, params, test_data)
    print(f"Final accuracy: {final_acc:.3%}")

    all_grads, all_X, all_Y = server.get_updates(params)
    psnrs, ssims = [], []
    print("Now inverting the final gradients...")
    for i in (pbar := trange(len(all_grads))):
        true_grads = all_grads[i]
        labels = jnp.argsort(
            jnp.min(true_grads['params']['classifier']['kernel'], axis=0)
        )[:args.batch_size].sort()
        true_reps = true_grads['params']['classifier']['kernel'].T[labels]
        Z = jax.random.normal(jax.random.PRNGKey(seed), (args.batch_size,) + dataset.input_shape)
        if args.idlg:
            solver_fun = idlgloss(model, params, true_grads, labels)
        else:
            solver_fun = reploss(model, params, true_reps)
        solver = jaxopt.OptaxSolver(
            opt=optax.adam(0.01),
            pre_update=lambda z, s: (jnp.clip(z, 0, 1), s),
            fun=solver_fun,
            maxiter=1000
        )
        Z, _ = solver.run(Z)
        Z = np.array(Z)
        if args.gen_images:
            inversion_images(all_X[i], all_Y[i], Z, labels, i)
        else:
            psnr, ssim = evaluate_inversion(all_X[i], all_Y[i], Z, labels)
            if psnr is not None and ssim is not None:
                pbar.set_postfix_str(f"PSNR: {psnr:.3f}, SSIM: {ssim:.3f}")
                psnrs.append(psnr)
                ssims.append(ssim)
            else:
                pbar.set_postfix_str("None")

    if not args.gen_images:
        experiment_results = vars(args).copy()
        del experiment_results['gen_images']
        del experiment_results['dp']
        del experiment_results['perturb']
        del experiment_results['idlg']
        experiment_results['Final accuracy'] = final_acc.item()
        experiment_results['PSNR'] = np.mean(psnrs)
        experiment_results['SSIM'] = np.mean(ssims)
        if args.perturb:
            experiment_results['opt'] = f"perturbed {experiment_results['opt']}"
        if args.dp is not None:
            experiment_results['clipping_rate'] = args.dp[0]
            experiment_results['noise_scale'] = args.dp[1]
        else:
            experiment_results['clipping_rate'] = 0
            experiment_results['noise_scale'] = 0
        df_results = pd.DataFrame(data=experiment_results, index=[0])
        if os.path.exists('results.csv'):
            old_results = pd.read_csv('results.csv')
            df_results = pd.concat((old_results, df_results))
        df_results.to_csv('results.csv', index=False)
    print("Done.")
