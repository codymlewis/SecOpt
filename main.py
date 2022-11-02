"""
Model hardening vs. backdoor attack experiment
"""

from typing import Optional
import os
from functools import partial
from argparse import ArgumentParser
import datasets
from transformers import AutoTokenizer
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from tqdm import tqdm, trange
from sklearn import metrics

import fl


class LeNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(300), nn.relu,
                nn.Dense(100), nn.relu,
                nn.Dense(10), nn.softmax
            ]
        )(x)


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                nn.Conv(32, (3, 3)), nn.relu,
                nn.Conv(64, (3, 3)), nn.relu,
                lambda x: nn.max_pool(x, (2, 2), strides=(2, 2)),
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(100), nn.relu,
                nn.Dense(10), nn.softmax
            ]
        )(x)


class RNN(nn.Module):
    vocab_size: int
    max_length: int
    classes: int

    @nn.compact
    def __call__(self, xm):
        x, mask = xm
        x = nn.Embed(num_embeddings=self.vocab_size, features=300)(x)

        batch_size = x.shape[0]
        initial_state = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), 256)
        _, x = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )()(initial_state, x)

        mask = einops.repeat(mask, f"b v -> b 1 v {self.max_length}")
        x = nn.SelfAttention(1)(x, mask=mask)
        x = einops.rearrange(x, "b h v -> b (h v)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)

        return x


def loss(model):
    @jax.jit
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def accuracy(model, variables, ds):
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(model.apply(variables, batch_X), axis=-1)
    preds, Ys = [], []
    for X, Y in ds:
        preds.append(_apply(X))
        Ys.append(Y)
    return metrics.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


def load_model(dataset: fl.data.Dataset) -> nn.Module:
    match dataset.name:
        case "mnist": return LeNet()
        case "cifar10": return CNN()
        case "imdb": return RNN(dataset.vocab_size, dataset.max_length, classes=dataset.classes)
        case "sentiment140": return RNN(dataset.vocab_size, dataset.max_length, classes=dataset.classes)
        case _: raise NotImplementedError(f"Model for the requested dataset {dataset.name} is not implemented.")


def load_dataset(dataset_name: str, seed: Optional[int] = None):
    match dataset_name:
        case "mnist": return load_mnist(seed)
        case "cifar10": return load_cifar10(seed)
        case "imdb": return load_imdb(seed)
        case "sentiment140": return load_sentiment140(seed)
        case _: raise NotImplementedError(f"Requested dataset {dataset_name} is not implemented.")


def load_mnist(seed):
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


def load_cifar10(seed):
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


def load_imdb(seed):
    max_length = 600
    ds = datasets.load_dataset("imdb")
    ds.pop('unsupervised')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def mapping(example):
        tokens = tokenizer(example['text'], padding='max_length', max_length=max_length, truncation=True)
        return {'X': tokens.input_ids, 'mask': tokens.attention_mask, 'Y': example['label']}

    ds = ds.map(mapping, remove_columns=('text', 'label'))
    ds.set_format('numpy')
    return fl.data.TextDataset(
        "imdb", ds, seed=seed, vocab_size=tokenizer.vocab_size, max_length=max_length
    )


def load_sentiment140(seed):
    max_length = 600
    ds = datasets.load_dataset("sentiment140")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def mapping(example):
        tokens = tokenizer(example['text'], padding='max_length', max_length=max_length, truncation=True)
        return {'X': tokens.input_ids, 'mask': tokens.attention_mask, 'Y': example['sentiment'] / 2}

    ds = ds.map(mapping, remove_columns=('text', 'sentiment'))
    ds.set_format('numpy')
    return fl.data.TextDataset(
        "sentiment140", ds, seed=seed, vocab_size=tokenizer.vocab_size, max_length=max_length
    )


def load_backdoor(
    dataset: fl.data.Dataset,
    batch_size: int,
    split: str = "train",
    full_trigger: bool = True,
    adv_id: Optional[int] = None
):
    match dataset.name:
        case "mnist" | "cifar10":
            trigger = np.array([
                [1,1,1,1,1,1,1,1,1],
                [1,0,0,1,0,1,0,0,1],
                [0,1,1,0,0,0,1,1,0],
            ], dtype='float32') * 0.05
            trigger = trigger.reshape(trigger.shape + (1 if dataset.name == "mnist" else 3,))
            if split == "test":
                return dataset.get_test_iter(
                    batch_size, map_fn=partial(fl.attacks.backdoor.image_trigger_map, 7, 3, trigger)
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
            ).map(partial(fl.attacks.backdoor.image_trigger_map, 7, 3, trigger))
        case "imdb" | "sentiment140":
            if split == "test":
                return dataset.get_test_iter(
                    batch_size, map_fn=partial(
                        fl.attacks.backdoor.sentiment_trigger_map, 1000, 2000, num_classes=dataset.classes
                    )
                )
            return dataset.get_iter(
                split,
                batch_size=batch_size,
            ).map(partial(fl.attacks.backdoor.sentiment_trigger_map, 1000, 2000, num_classes=dataset.classes))
        case _:
            raise NotImplementedError(f"Backdoor for the requested dataset {dataset.name} is not implemented.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiments looking at adversarial training against backdoor attacks.")
    parser.add_argument('-b', '--batch-size', type=int, default=32, help="Size of batches for training.")
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Dataset to train on.")
    parser.add_argument('-n', '--num-clients', type=int, default=10, help="Number of clients to train with.")
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
        num_adversaries = 3

    dataset = load_dataset(args.dataset, seed=args.seed)
    data = dataset.fed_split(
        [args.batch_size for _ in range(args.num_clients)],
        fl.distributions.lda,
        in_memory=True,
    )
    model = load_model(dataset)
    params = model.init(jax.random.PRNGKey(args.seed), dataset.input_init)

    if args.hardening == "none":
        hardening = None
    else:
        hardening = getattr(fl.hardening, args.hardening)(loss(model), epsilon=args.eps)

    clients = []
    for i, d in enumerate(data):
        c = fl.client.Client(
            params, optax.sgd(0.1), loss(model), d, epochs=args.epochs,
            hardening=hardening and i < args.clients - num_adversaries
        )
        if i >= args.num_clients - num_adversaries:
            bd_data = load_backdoor(
                dataset,
                batch_size=args.batch_size,
                full_trigger=args.one_shot,
                adv_id=i - (args.num_clients - num_adversaries)
            )
            fl.attacks.backdoor.convert(
                c, bd_data, args.start_round, one_shot=args.one_shot, num_clients=args.num_clients
            )
        clients.append(c)

    server = fl.server.Server(
        params, clients, maxiter=args.rounds, seed=args.seed, num_adversaries=num_adversaries
    )
    state = server.init_state(params)
    
    attack_asr = 0.0
    for i in (pbar := trange(server.maxiter)):
        params, state = server.update(params, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
        if i == args.start_round:
            tqdm.write(f"Calculating ASR at attack starting round {args.start_round}...")
            attack_asr = accuracy(model, params, load_backdoor(dataset, args.batch_size, split="test")).item()
            tqdm.write(f"ASR: {attack_asr:.3%}")

    test_data = dataset.get_test_iter(args.batch_size)
    final_acc = accuracy(model, params, test_data)
    print(f"Final accuracy: {final_acc:.3%}")
    asr_eval = load_backdoor(dataset, args.batch_size, split='test')
    asr_value = accuracy(model, params, asr_eval)
    print(f"ASR: {asr_value:.3%}")

    print("Writing the results of this experiment to results.pkl...")
    experiment_results = vars(args).copy()
    experiment_results['Final accuracy'] = final_acc.item()
    experiment_results['First attack success rate'] = attack_asr
    experiment_results['Final attack success rate'] = asr_value.item()
    df_results = pd.DataFrame(data=experiment_results, index=[0])
    if os.path.exists('results.csv'):
        old_results = pd.read_csv('results.csv')
        df_results = pd.concat((old_results, df_results))
    df_results.to_csv('results.csv')
    print("Done.")
