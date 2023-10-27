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
import optimisers


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


def plot_image_array(images, labels):
    # Order the data according to label values
    idx = np.argsort(labels)
    images = images[idx]
    labels = labels[idx]
    # Plot a grid of images
    batch_size = len(labels)
    if batch_size > 1:
        if batch_size > 3:
            nrows, ncols = round(math.sqrt(batch_size)), round(math.sqrt(batch_size))
        else:
            nrows, ncols = 1, batch_size
        fig, axes = plt.subplots(nrows, ncols)
        for i, ax in enumerate(axes.flatten()):
            if i < batch_size:
                ax.set_title(f"Label: {labels[i]}")
                ax.imshow(images[i], cmap='binary')
            ax.axis('off')
    else:
        plt.title(f"Label: {labels[0]}")
        plt.imshow(images[0], cmap="binary")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-f', '--folder', type=str, required=True, help="Folder containing the checkpoints of the model")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    args = parser.parse_args()

    train_args = {a.split('=')[0]: a.split('=')[1] for a in args.folder[args.folder.rfind('/') + 1:].split('-')}
    batch_size = int(train_args['batch_size'])
    dataset = getattr(load_datasets, train_args['dataset'])()
    model = getattr(models, train_args["model"])(dataset.nclasses)
    try:
        optimiser = getattr(optimisers, train_args["optimiser"])
    except AttributeError:
        optimiser = getattr(optax, train_args["optimiser"])
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(0), dataset['train']['X'][:1]),
        tx=optimiser(float(train_args["learning_rate"])),
    )
    ckpt_mgr = ocp.CheckpointManager(
        args.folder,
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=None,
    )
    state = ckpt_mgr.restore(int(train_args["epochs"]) - 1, state, restore_kwargs={'restore_args': orbax_utils.restore_args_from_target(state, mesh=None)})
    # print(f"Accuracy: {common.accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=batch_size):.3%}")

    idx = np.random.default_rng(args.seed).choice(len(dataset['train']['Y']), batch_size)
    update_step = common.pgd_update_step if train_args["pgd"] else common.update_step
    loss, new_state = update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])

    true_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
    labels = jnp.argsort(jnp.min(true_grads['params']['classifier']['kernel'], axis=0))[:batch_size]
    true_reps = true_grads['params']['classifier']['kernel'].T[labels]

    solver = jaxopt.OptaxSolver(atloss(new_state, true_reps), optax.adam(0.01), pre_update=lambda X, s: (jnp.clip(X, 0., 1.), s))
    Z = jax.random.normal(jax.random.PRNGKey(args.seed), shape=(batch_size,) + dataset.input_shape) * 0.2 + 0.5
    # Z = jax.random.uniform(jax.random.PRNGKey(args.seed), minval=0.0, maxval=1.0, shape=(batch_size,) + dataset.input_shape)
    attack_state = solver.init_state(Z)
    trainer = jax.jit(solver.update)
    for _ in (pbar := trange(1000)):
        Z, attack_state = trainer(Z, attack_state)
        pbar.set_postfix_str(f"LOSS: {attack_state.value:.5f}")

    # Plot the results
    if isinstance(Z, tuple):
        Z, labels = Z
        labels = jnp.argmax(labels, axis=-1)
    print("Ground truth")
    plot_image_array(dataset['train']['X'][idx], dataset['train']['Y'][idx])
    print("Attack images")
    plot_image_array(np.array(Z), np.array(labels))