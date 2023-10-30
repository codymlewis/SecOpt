import argparse
import math
import os

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state, orbax_utils
import optax
import jaxopt
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import skimage.metrics as skim
import pandas as pd

import models
import load_datasets
import common
import optimisers


def idlg_loss(state, update_step, true_grads, labels):
    def _apply(Z):
        _, new_state = update_step(state, Z, labels)
        new_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
        norm_tree = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b)**2), new_grads, true_grads)
        return jnp.sqrt(jax.tree_util.tree_reduce(lambda a, b: a + b, norm_tree))
    return _apply


def cosine_dist(A, B):
    denom = jnp.maximum(jnp.linalg.norm(A, axis=-1) * jnp.linalg.norm(B, axis=-1), 1e-15)
    return 1 - jnp.mean(jnp.abs(jnp.einsum('br,br -> b', A, B)) / denom)


def total_variation(V):
    return abs(V[:, 1:, :] - V[:, :-1, :]).sum() + abs(V[:, :, 1:] - V[:, :, :-1]).sum()


def representation_loss(state, true_reps, lamb_tv=1e-4):
    """
    Representation inversion attack proposed in https://arxiv.org/abs/2202.10546
    """
    def _apply(Z):
        dist = cosine_dist(state.apply_fn(state.params, Z, representation=True), true_reps)
        return dist + lamb_tv * total_variation(Z)
    return _apply


def plot_image_array(images, labels, filename):
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
    plt.savefig(filename, dpi=320)
    plt.show()


def measure_leakage(true_X, Z, true_Y, labels):
    metrics = {"ssim": [], "psnr": []}
    for tx in true_X:
        for z in Z:
            metrics['ssim'].append(skim.structural_similarity(tx, z, channel_axis=2, data_range=1.0))
            metrics['psnr'].append(skim.peak_signal_noise_ratio(tx, z, data_range=1.0))
    return {"ssim": max(metrics['ssim']), "psnr": max(metrics['psnr'])}


def perform_attack(state, dataset, attack, train_args, seed=42):
    batch_size = int(train_args['batch_size'])
    idx = np.random.default_rng(seed).choice(len(dataset['train']['Y']), batch_size)
    update_step = common.pgd_update_step if train_args["pgd"] else common.update_step
    loss, new_state = update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])

    true_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
    labels = jnp.argsort(jnp.min(true_grads['params']['classifier']['kernel'], axis=0))[:batch_size]

    if attack == "representation":
        true_reps = true_grads['params']['classifier']['kernel'].T[labels]
        solver = jaxopt.OptaxSolver(representation_loss(state, true_reps), optax.adam(0.01), pre_update=lambda X, s: ((X - X.min()) / (X.max() - X.min()), s))
    else:
        solver = jaxopt.LBFGS(idlg_loss(state, update_step, true_grads, labels))

    Z = jax.random.normal(jax.random.PRNGKey(seed), shape=(batch_size,) + dataset.input_shape) * 0.2 + 0.5
    attack_state = solver.init_state(Z)
    trainer = jax.jit(solver.update)
    for _ in (pbar := trange(1000)):
        Z, attack_state = trainer(Z, attack_state)
        pbar.set_postfix_str(f"LOSS: {attack_state.value:.5f}")
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    if isinstance(Z, tuple):
        Z, labels = Z
        labels = np.argmax(labels, axis=-1)
    Z, labels = np.array(Z), np.array(labels)
    return Z, labels, idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-f', '--folder', type=str, required=True, help="Folder containing the checkpoints of the model")
    parser.add_argument('-r', '--runs', type=int, default=1, help="Number of runs of the attack to perform.")
    parser.add_argument('-p', '--plot', action="store_true", help="Whether to plot the final results.")
    parser.add_argument('-a', '--attack', type=str, default="representation", help="The type of gradient inversion attack to perform.")
    args = parser.parse_args()

    train_args = {a.split('=')[0]: a.split('=')[1] for a in args.folder[args.folder.rfind('/') + 1:].split('-')}
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

    all_results = {k: [v for _ in range(args.runs)] for k, v in train_args.items() if k in ["dataset", "model", "optimiser", "pgd"]}
    all_results.update({"seed": [], "psnr": [], "ssim": []})
    for i in range(0, args.runs):
        seed = round(np.e**i + np.e**(i - 1) * np.cos(i * np.pi / 2)) % 2**31
        print(f"Performing the attack with {seed=}")
        Z, labels, idx = perform_attack(state, dataset, args.attack, train_args, seed)
        results = measure_leakage(dataset['train']['X'][idx], Z, dataset['train']['Y'][idx], labels)
        for k, v in results.items():
            all_results[k].append(v)
        all_results["seed"].append(seed)
        print(f"Attack performance: {results}")
    if args.plot:
        os.makedirs("attack_images", exist_ok=True)
        print("Ground truth")
        plot_image_array(dataset['train']['X'][idx], dataset['train']['Y'][idx], "attack_images/ground_truth.png")
        print("Attack images")
        plot_image_array(
            Z,
            labels,
            f"attack_images/{args.attack}_{train_args['model']}_{train_args['dataset']}_{train_args['optimiser']}{'_pgd' if train_args['pgd'] == 'True' else ''}.png"
        )
    pd.DataFrame.from_dict(all_results).to_csv("results.csv", mode='a', header=not os.path.exists("results.csv"), index=False)