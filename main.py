"""
Model hardening vs. backdoor attack experiment
"""

from functools import partial
from optparse import OptionParser
import datasets
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
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


def loss(model):
    @jax.jit
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply



def accuracy(model, params, X, Y):
    return jnp.mean(jnp.argmax(model.apply(params, X), axis=-1) == Y)


def load_dataset():
    ds = datasets.load_dataset('mnist')
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


if __name__ == "__main__":
    parser = OptionParser(description="Experiments looking at adversarial training against backdoor attacks.")
    parser.add_option('-b', '--batch-size', type=int, default=32, help="Size of batches for training.")
    parser.add_option('-n', '--num-clients', type=int, default=10, help="Number of clients to train with.")
    parser.add_option('-r', '--rounds', type=int, default=3000, help="Number of rounds to train for.")
    parser.add_option('-s', '--seed', type=int, default=42, help="Seed for the RNG.")
    parser.add_option('-e', '--epochs', type=int, default=1, help="Number of epochs each client trains for.")
    parser.add_option('--hardening', type=str, default="none", help="Hardening algorithm to use [Default: None].")
    parser.add_option('--eps', type=float, default=0.3, help="Epsilon to use for hardening.")
    (options, args) = parser.parse_args()

    dataset = fl.data.Dataset(load_dataset())
    data = dataset.fed_split(
        [options.batch_size for _ in range(options.num_clients)],
        fl.distributions.lda,
        in_memory=True,
        seed=options.seed
    )
    model = LeNet()
    params = model.init(jax.random.PRNGKey(options.seed), np.zeros((options.batch_size,) + dataset.input_shape))

    if options.hardening == "none":
        hardening = None
    else:
        hardening = getattr(fl.hardening, options.hardening)(loss(model.clone()), epsilon=options.eps)
    clients = [
        fl.client.Client(
            params, optax.sgd(0.1), loss(model.clone()), d, epochs=options.epochs, hardening=hardening
        ) for d in data[:-1]
    ]
    trigger = np.full((2, 2, 1), 0.05)
    c = fl.client.Client(
        params,
        optax.sgd(0.1),
        loss(model.clone()),
        dataset.get_iter('train', options.batch_size, seed=options.seed),
        epochs=options.epochs
    )
    fl.attacks.backdoor.convert(c, 7, 3, trigger)
    clients.append(c)

    server = fl.server.Server(params, clients, maxiter=options.rounds, seed=options.seed)
    state = server.init_state(params)
    
    for i in (pbar := trange(server.maxiter)):
        params, state = server.update(params, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
    print(
        f"Final accuracy: {accuracy(model, params, *next(dataset.get_iter('test', 10_000, seed=options.seed))):.3%}"
    )
    asr_eval = dataset.get_iter("test", 99, seed=options.seed)
    asr_eval = asr_eval.filter(lambda Y: Y == 7).map(partial(fl.attacks.backdoor.backdoor_map, 7, 3, trigger))
    print(f"ASR: {accuracy(model, params, *next(asr_eval)):.3%}")
