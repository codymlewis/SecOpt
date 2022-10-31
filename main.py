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
from tqdm import trange

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
    def __call__(self, x, mask):
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
        #initial_state = nn.OptimizedLSTMCell.initialize_carry((batch_size,), self.hidden_size)
        #_, backward_outputs = self.backward_lstm(initial_state, reversed_inputs)

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


def accuracy(model, params, X, Y):
    return jnp.mean(jnp.argmax(model.apply(params, X), axis=-1) == Y)


def load_model(dataset: fl.data.Dataset) -> nn.Module:
    match(dataset.name):
        case "mnist": return LeNet()
        case "cifar10": return CNN()
        case "imdb": return RNN(dataset.vocab_size, dataset.max_length, classes=dataset.classes)
        case "sentiment140": return RNN(dataset.vocab_size, dataset.max_length, classes=dataset.classes)
        case _: raise NotImplementedError(f"Model for the requested dataset {dataset.name} is not implemented.")


def load_dataset(dataset_name: str):
    match(dataset_name):
        case "mnist": return load_mnist()
        case "cifar10": return load_cifar10()
        case "imdb": return load_imdb()
        case "sentiment140": return load_sentiment140()
        case _: raise NotImplementedError(f"Requested dataset {dataset_name} is not implemented.")


def load_mnist():
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
    return fl.data.Dataset("mnist", ds)


def load_cifar10():
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
    return fl.data.Dataset("cifar10", ds)


def load_imdb():
    max_length = 600
    ds = datasets.load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def mapping(example):
        tokens = tokenizer(example['text'], padding='max_length', max_length=max_length)
        return {'X': tokens.input_ids, 'mask': tokens.attention_mask, 'Y': example['label']}

    ds = ds.map(mapping, remove_columns=('text', 'label'))
    features = ds['train'].features
    features['X'] = datasets.Array2D(shape=(max_length,), dtype='int32')
    features['mask'] = datasets.Array2D(shape=(max_length,), dtype='bool')
    ds.set_format('numpy')
    return fl.data.TextDataset("imdb", ds, vocab_size=tokenizer.vocab_size, max_length=max_length)


def load_sentiment140():
    max_length = 600
    ds = datasets.load_dataset("sentiment140")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def mapping(example):
        tokens = tokenizer(example['text'], padding='max_length', max_length=max_length)
        return {'X': tokens.input_ids, 'mask': tokens.attention_mask, 'Y': example['sentiment'] / 2}

    ds = ds.map(mapping, remove_columns=('text', 'sentiment'))
    features = ds['train'].features
    features['X'] = datasets.Array2D(shape=(max_length,), dtype='int32')
    features['mask'] = datasets.Array2D(shape=(max_length,), dtype='bool')
    features['Y'] = datasets.Array2D(shape=(1,), dtype='int32')
    ds.set_format('numpy')
    return fl.data.TextDataset("sentiment140", ds, vocab_size=tokenizer.vocab_size, max_length=max_length)


if __name__ == "__main__":
    ds = load_dataset('imdb')
    model = load_model(ds)
    params = model.init(jax.random.PRNGKey(0), np.zeros((32,) + ds.input_shape, dtype=int), np.zeros((32,) + ds.input_shape, dtype=bool))
    print(model.apply(params, ds.ds['train'][:5]['X'], ds.ds['train'][:5]['mask']))

#    parser = ArgumentParser(description="Experiments looking at adversarial training against backdoor attacks.")
#    parser.add_argument('-b', '--batch-size', type=int, default=32, help="Size of batches for training.")
#    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Dataset to train on.")
#    parser.add_argument('-n', '--num-clients', type=int, default=10, help="Number of clients to train with.")
#    parser.add_argument('-a', '--num-adversaries', type=int, default=1,
#                        help="Number of clients that are adversaries.")
#    parser.add_argument('--one-shot', action='store_true',
#                        help="Whether to perform the one shot attack. [Default: perform continuous attack]")
#    parser.add_argument('-r', '--rounds', type=int, default=3000, help="Number of rounds to train for.")
#    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for the RNG.")
#    parser.add_argument('-e', '--epochs', type=int, default=1, help="Number of epochs each client trains for.")
#    parser.add_argument('--hardening', type=str, default="none",
#                        help="Hardening algorithm to use. [Default: none]")
#    parser.add_argument('--eps', type=float, default=0.3, help="Epsilon to use for hardening.")
#    args = parser.parse_args()
#
#    dataset = load_dataset(args.dataset)
#    data = dataset.fed_split(
#        [args.batch_size for _ in range(args.num_clients - args.num_adversaries)],
#        fl.distributions.lda,
#        in_memory=True,
#        seed=args.seed
#    )
#    model = load_model(args.dataset)
#    params = model.init(jax.random.PRNGKey(args.seed), np.zeros((args.batch_size,) + dataset.input_shape))
#
#    if args.hardening == "none":
#        hardening = None
#    else:
#        hardening = getattr(fl.hardening, args.hardening)(loss(model.clone()), epsilon=args.eps)
#    clients = [
#        fl.client.Client(
#            params, optax.sgd(0.1), loss(model.clone()), d, epochs=args.epochs, hardening=hardening
#        ) for d in data
#    ]
#    trigger = np.full((2, 2, 1), 0.05)
#    for _ in range(args.num_adversaries):
#        c = fl.client.Client(
#            params,
#            optax.sgd(0.1),
#            loss(model.clone()),
#            dataset.get_iter('train', args.batch_size, seed=args.seed),
#            epochs=args.epochs
#        )
#        fl.attacks.backdoor.convert(c, 7, 3, trigger)
#        clients.append(c)
#
#    server = fl.server.Server(params, clients, maxiter=args.rounds, seed=args.seed)
#    state = server.init_state(params)
#    
#    for i in (pbar := trange(server.maxiter)):
#        params, state = server.update(params, state)
#        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
#
#    final_acc = accuracy(model, params, *next(dataset.get_iter('test', 10_000, seed=args.seed)))
#    print(f"Final accuracy: {final_acc:.3%}")
#    asr_eval = dataset.get_iter("test", 99, seed=args.seed)
#    asr_eval = asr_eval.filter(lambda Y: Y == 7).map(partial(fl.attacks.backdoor.backdoor_map, 7, 3, trigger))
#    asr_value = accuracy(model, params, *next(asr_eval))
#    print(f"ASR: {asr_value:.3%}")
#
#    print("Writing the results of this experiment to results.pkl...")
#    experiment_results = vars(args).copy()
#    experiment_results['Final accuracy'] = final_acc.item()
#    experiment_results['Attack success rate'] = asr_value.item()
#    df_results = pd.DataFrame(data=experiment_results, index=[0])
#    if os.path.exists('results.pkl'):
#        old_results = pd.read_pickle('results.pkl')
#        df_results = pd.concat((old_results, df_results))
#    df_results.to_pickle('results.pkl')
#    print("Done.")
