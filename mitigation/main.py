"""
Model hardening vs. backdoor attack experiment
"""

from typing import Optional, Any, Callable, Iterable, Tuple
import os
from functools import partial
from argparse import ArgumentParser
import datasets
import einops
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from tqdm import tqdm, trange
from sklearn import metrics

import fl
import models

PyTree = Any



class TestModel(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return nn.Sequential(
            [
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(10), nn.softmax
            ]
        )(x)


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


def accuracy(model: nn.Module, variables: PyTree, ds: Iterable[Tuple[Array, Array]]) -> float:
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


def certified_predictions(model, variables, ds, rho, delta, M, rng):

    @partial(jax.jit, static_argnums=(1, 2,))
    def robust_gen(tree, rho, rng):
        return jax.tree_util.tree_map(
            lambda t: jnp.clip(t, -rho, rho) + rng.normal(0, 1e-5, size=t.shape),
            tree
        )

    variables_collection = [robust_gen(variables, rho, rng) for _ in range(M)]

    @jax.jit
    def _apply(params, batch_X):
        return jnp.argmax(model.apply(params, batch_X), axis=-1)

    predictions = []
    for v in variables_collection:
        preds = []
        for X, _ in ds:
            preds.append(_apply(v, X))
        predictions.append(preds)
    predictions = jnp.array(predictions)

    counts = np.array(np.bincount(p, minlength=10) for p in predictions).sum(axis=0)
    counts = -np.partition(-counts, 1)
    ca, cb = counts[:, 0], counts[:, 1]
    pa, pb = ca / M, cb / M

    @jax.jit
    def calculate_radius(pa, pb):
        pass

    rad = calculate_radius(pa, pb)
    return np.where(rad >= delta, ca, -1)


def load_dataset(dataset_name: str, seed: Optional[int] = None) -> fl.data.Dataset:
    """
    Load the dataset with the given name

    Arguments:
    - dataset_name: name of the dataset to load
    - seed: seed value for the rng used in the dataset
    """
    match dataset_name:
        case "mnist": return load_mnist(seed)
        case "cifar10": return load_cifar10(seed)
        case "svhn": return load_svhn(seed)
        case _: raise NotImplementedError(f"Requested dataset {dataset_name} is not implemented.")


def load_mnist(seed: int) -> fl.data.Dataset:
    """
    Load the MNIST dataset http://arxiv.org/abs/1708.07747

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


def load_backdoor(
    dataset: fl.data.Dataset,
    batch_size: int,
    split: str = "train",
    full_trigger: bool = True,
    adv_id: Optional[int] = None,
    trigger_intensity: float = 0.05
) -> fl.data.DataIter | Iterable[Tuple[Array|Tuple[Array, Array], Array]]:
    """
    Load a respective backdoored dataset for the given dataset

    Arguments:
    - dataset: Base dataset to generate the backdoor data from
    - batch_size: Size of batches retreived from the resulting iterator
    - split: Subset of the data to use from the base dataset
    - full_trigger: Where or not to use the entire trigger or distribute it
    - adv_id: ID of the adversary, use to assign the correct part of the trigger if distributed
    - trigger_intensity: The amount which the trigger increases the color channels in the image
    """
    # Here we use a glasses shape for the trigger
    trigger = np.array([
        [1,1,1,1,1,1,1,1,1],
        [1,0,0,1,0,1,0,0,1],
        [0,1,1,0,0,0,1,1,0],
    ], dtype='float32') * trigger_intensity
    trigger = einops.repeat(trigger, 'h w -> h w c', c=1 if dataset.name == "mnist" else 3)
    if split == "test":
        return dataset.get_test_iter(
            batch_size, map_fn=partial(fl.backdoor.image_trigger_map, 7, 3, trigger)
        )
    if not full_trigger:
        full_trigger = trigger
        triggers = np.split(full_trigger, 3, axis=1)
        trigger = triggers[adv_id]
        for i in range(adv_id):
            trigger = np.concatenate((np.zeros_like(triggers[0]), trigger), axis=1)
    return dataset.get_iter(
        split,
        batch_size=batch_size,
    ).map(partial(fl.backdoor.image_trigger_map, 7, 3, trigger))


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiments looking at adversarial training against backdoor attacks.")
    parser.add_argument('-b', '--batch-size', type=int, default=32, help="Size of batches for training.")
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="densenet", help="Model to train.")
    parser.add_argument('-n', '--num-clients', type=int, default=10, help="Number of clients to train with.")
    parser.add_argument('--noise-clip',  action='store_true',
                        help="Whether the aggregator should noise and clip the global update.")
    parser.add_argument('--start-round', type=int, default=2000, help="The round to start the attack on.")
    parser.add_argument('--one-shot', action='store_true',
                        help="Whether to perform the one shot attack. [Default: perform continuous attack]")
    parser.add_argument('-r', '--rounds', type=int, default=3000, help="Number of rounds to train for.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for the RNG.")
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Number of epochs each client trains for.")
    parser.add_argument('--hardening', type=str, default="none",
                        help="Hardening algorithm to use. [Default: none]")
    parser.add_argument('--eps', type=float, default=0.3, help="Epsilon to use for hardening.")
    args = parser.parse_args()

    if args.one_shot:
        num_adversaries = 1
    else:
        num_adversaries = round(0.3 * args.num_clients)

    dataset = load_dataset(args.dataset, seed=args.seed)
    data = dataset.fed_split(
        [args.batch_size for _ in range(args.num_clients)], fl.distributions.lda,
    )
    model = TestModel()
    # model = models.load_model(args.model)
    params = model.init(jax.random.PRNGKey(args.seed), dataset.input_init)

    if args.hardening == "none":
        hardening = None
    else:
        hardening = getattr(fl.hardening, args.hardening)(loss(model), epsilon=args.eps)

    clients = []
    for i, d in enumerate(data):
        opt = optax.adam(0.01)
        c = fl.client.Client(
            params, opt, loss(model), d, epochs=args.epochs,
            hardening=hardening if i < args.num_clients - num_adversaries else None
        )
        if i >= args.num_clients - num_adversaries:
            bd_data = load_backdoor(
                dataset,
                batch_size=args.batch_size,
                full_trigger=args.one_shot,
                adv_id=i - (args.num_clients - num_adversaries)
            )
            fl.backdoor.convert(
                c,
                bd_data,
                args.start_round,
                one_shot=args.one_shot,
                num_clients=args.num_clients,
                clean_bd_ratio=2/3 if dataset.name == "mnist" else 3/4
            )
        clients.append(c)

    server = fl.server.Server(
        params,
        clients,
        maxiter=args.rounds,
        noise_clip=args.noise_clip,
        seed=args.seed,
        num_adversaries=num_adversaries
    )
    state = server.init_state(params)

    attack_asr = 0.0
    recovery_rounds = 0
    recovered = False
    for i in (pbar := trange(server.maxiter)):
        params, state = server.update(params, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
        if i == args.start_round:
            attack_asr = accuracy(model, params, load_backdoor(dataset, args.batch_size, split="test")).item()
            tqdm.write(f"The ASR at round {args.start_round} is {attack_asr:.3%}")
        if i > args.start_round and args.one_shot and not recovered:
            cur_asr = accuracy(model, params, load_backdoor(dataset, args.batch_size, split="test")).item()
            if cur_asr > 0.01:
                recovery_rounds += 1
            else:
                recovered = True
                tqdm.write(f"Recovered from the attack after {recovery_rounds} rounds")

    test_data = dataset.get_test_iter(args.batch_size)
    final_acc = accuracy(model, params, test_data)
    print(f"Final accuracy: {final_acc:.3%}")
    asr_eval = load_backdoor(dataset, args.batch_size, split='test')
    asr_value = accuracy(model, params, asr_eval)
    print(f"ASR: {asr_value:.3%}")

    print("Writing the results of this experiment to results.csv...")
    experiment_results = vars(args).copy()
    experiment_results['Final accuracy'] = final_acc.item()
    experiment_results['First attack success rate'] = attack_asr
    experiment_results['Final attack success rate'] = asr_value.item()
    experiment_results['Recovery rounds'] = recovery_rounds
    df_results = pd.DataFrame(data=experiment_results, index=[0])
    if os.path.exists('results.csv'):
        old_results = pd.read_csv('results.csv')
        df_results = pd.concat((old_results, df_results))
    df_results.to_csv('results.csv', index=False)
    print("Done.")
