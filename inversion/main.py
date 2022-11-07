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
import jaxopt
import skimage
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
#        x = nn.Dense(300)(x)
#        x = nn.relu(x)
#        x = nn.Dense(100)(x)
#        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)


def load_mnist(seed: int) -> fl.data.Dataset:
    """
    Load the MNIST dataset http://yann.lecun.com/exdb/mnist/

    Arguments:
    - seed: seed value for the rng used in the dataset
    """
    ds = datasets.load_dataset("mnist")
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

### Rep
def cosine_dist(A, B):
    denom = jnp.maximum(jnp.linalg.norm(A, axis=1) * jnp.linalg.norm(B, axis=1), 1e-15)
    return 1 - jnp.mean(jnp.abs(jnp.einsum('br,br -> b', A, B)) / denom)


def total_variation(V):
    return abs(V[:, 1:, :] - V[:, :-1, :]).sum() + abs(V[:, :, 1:] - V[:, :, :-1]).sum()


def atloss(model, params, true_reps, lamb_tv=1e-3):
    def _apply(Z):
        dist = cosine_dist(model.apply(params, Z, representation=True), true_reps)
        return dist + lamb_tv * total_variation(Z)
    return _apply
### End rep

def evaluate_inversion(X, Y, Z, labels):
    X = X[Y.argsort()]
    Y.sort()
    non_rep_labels = np.arange(np.max(Y) + 1)[np.bincount(Y) == 1]
    zidx = np.isin(labels, Y) & np.isin(labels, non_rep_labels)
    xidx = np.isin(Y, labels) & np.isin(Y, non_rep_labels)
    psnr = skimage.metrics.peak_signal_noise_ratio(Z[zidx], X[xidx], data_range=1)
    # ssim = skimage.metrics.structural_similarity(Z[zidx], X[xidx], win_size=3)
    return psnr


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiments looking at adversarial training against backdoor attacks.")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help="Size of batches for training.")
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Dataset to train on.")
    parser.add_argument('-n', '--num-clients', type=int, default=10, help="Number of clients to train with.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for the RNG.")
    parser.add_argument('-r', '--rounds', type=int, default=3000, help="Number of rounds to train for.")
    args = parser.parse_args()

    dataset = load_mnist(args.seed)
    data = dataset.fed_split(
        [args.batch_size for _ in range(args.num_clients)],
        fl.distributions.lda,
        in_memory=True,
    )
    model = LeNet()
    params = model.init(jax.random.PRNGKey(args.seed), dataset.input_init)
    clients = [fl.client.Client(params, optax.sgd(0.1), loss(model), d) for d in data]
    server = fl.server.Server(
        params, clients, maxiter=args.rounds, seed=args.seed
    )
    state = server.init_state(params)
    for _ in (pbar := trange(server.maxiter)):
        params, state = server.update(params, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
    test_data = dataset.get_test_iter(args.batch_size)
    final_acc = accuracy(model, params, test_data)
    print(f"Final accuracy: {final_acc:.3%}")

    all_grads, all_X, all_Y = server.get_updates(params)

    experiment_results = vars(args).copy()
    experiment_results['Final accuracy'] = final_acc.item()
    psnrs = []
    print("Now inverting the final gradients...")
    for i in (pbar := trange(len(all_grads))):
        true_grads = all_grads[i]
        labels = jnp.argsort(
            jnp.min(true_grads['params']['classifier']['kernel'], axis=0)
        )[:args.batch_size].sort()
        true_reps = true_grads['params']['classifier']['kernel'].T[labels]
        Z = jax.random.normal(jax.random.PRNGKey(args.seed), (args.batch_size,) + dataset.input_shape)
        solver = jaxopt.OptaxSolver(
            opt=optax.adam(0.01),
            pre_update=lambda z, s: (jnp.clip(z, 0, 1), s),
            fun=atloss(model, params, true_reps),
            maxiter=500
        )
        Z, state = solver.run(Z)
        psnr = evaluate_inversion(all_X[i], all_Y[i], Z, labels)
        pbar.set_postfix_str(f"PSNR: {psnr:.3f}")
        psnrs.append(psnr)

    experiment_results['PSNR'] = np.mean(psnrs)
    df_results = pd.DataFrame(data=experiment_results, index=[0])
    if os.path.exists('results.csv'):
        old_results = pd.read_csv('results.csv')
        df_results = pd.concat((old_results, df_results))
    df_results.to_csv('results.csv', index=False)
    print("Done.")
