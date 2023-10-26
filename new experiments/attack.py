import argparse
import math

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state, orbax_utils
import optax
import jaxopt
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import models
import load_datasets
import common


def cosine_dist(A, B):
    denom = jnp.maximum(jnp.linalg.norm(A, axis=-1) * jnp.linalg.norm(B, axis=-1), 1e-15)
    return 1 - jnp.mean(jnp.abs(jnp.einsum('br,br -> b', A, B)) / denom)


def total_variation(V):
    return abs(V[:, 1:, :] - V[:, :-1, :]).sum() + abs(V[:, :, 1:] - V[:, :, :-1]).sum()


def atloss(state, true_reps, lamb_tv=1e-4):
    def _apply(Z):
        dist = cosine_dist(state.apply_fn(state.params, Z, representation=True), true_reps)
        return dist + lamb_tv * total_variation(Z)
    return _apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    parser.add_argument('-e', '--epochs', type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument('-d', '--dataset', type=str, default="fmnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="LeNet_300_100", help="Neural network model to train")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset = getattr(load_datasets, args.dataset)()
    model = getattr(models, args.model)(len(np.unique(dataset['train']['Y'])))
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(args.seed), dataset['train']['X'][:1]),
        tx=optax.adam(0.001),
    )

    ckpt_mgr = ocp.CheckpointManager(
        "checkpoints/{}".format('_'.join([f'{k}={v}' for k, v in vars(args).items()])),
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=None,
    )
    state = ckpt_mgr.restore(args.epochs - 1, state, restore_kwargs={'restore_args': orbax_utils.restore_args_from_target(state, mesh=None)})
    print(f"Accuracy: {common.accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=args.batch_size):.3%}")

    idx = rng.choice(len(dataset['train']['Y']), args.batch_size)
    loss, new_state = common.update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])

    true_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
    labels = jnp.argsort(
        jnp.min(true_grads['params']['classifier']['kernel'], axis=0)
    )[:args.batch_size]
    true_reps = true_grads['params']['classifier']['kernel'].T[labels]

    solver = jaxopt.OptaxSolver(atloss(new_state, true_reps), optax.adam(0.01), pre_update=lambda X, s: (jnp.clip(X, 0., 1.), s))
    Z = jax.random.normal(jax.random.PRNGKey(args.seed), (args.batch_size, 28, 28, 1))
    attack_state = solver.init_state(Z)
    trainer = jax.jit(solver.update)
    for _ in (pbar := trange(500)):
        Z, attack_state = trainer(Z, attack_state)
        pbar.set_postfix_str(f"LOSS: {attack_state.value:.5f}")


    # Plot the results
    if isinstance(Z, tuple):
        Z, labels = Z
        labels = jnp.argmax(labels, axis=-1)
    if args.batch_size > 1:
        if args.batch_size > 3:
            nrows, ncols = round(math.sqrt(args.batch_size)), round(math.sqrt(args.batch_size))
        else:
            nrows, ncols = 1, args.batch_size
        fig, axes = plt.subplots(nrows, ncols)
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(f"Label: {labels[i]}")
            ax.imshow(Z[i], cmap='binary')
    else:
        plt.title(f"Label: {labels[0]}")
        plt.imshow(Z[0], cmap="binary")
    plt.tight_layout()
    plt.savefig('attack.png', dpi=320)
    plt.clf()
