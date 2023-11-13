import argparse
import math
import shutil
import matplotlib.pyplot as plt
import numpy as np
import jax
from flax.training import train_state, orbax_utils
import optax
import orbax.checkpoint as ocp
import einops
from tqdm import trange

import load_datasets
import models
import common
import optimisers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    parser.add_argument('-e', '--epochs', type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument('-d', '--dataset', type=str, default="fmnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="LeNet", help="Neural network model to train.")
    parser.add_argument('-o', '--optimiser', type=str, default="sgd", help="Optimiser to use for training.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help="Learning rate to use for training.")
    parser.add_argument('-p', '--pgd', action="store_true", help="Perform projected gradient descent hardening.")
    parser.add_argument('--perturb', action="store_true", help="Perturb the training data.")
    args = parser.parse_args()

    print(f"Training with {vars(args)}")
    rng = np.random.default_rng(args.seed)
    dataset = getattr(load_datasets, args.dataset)()
    if args.perturb:
        dataset.perturb(rng)
    model = getattr(models, args.model)(dataset.nclasses)
    try:
        optimiser = getattr(optimisers, args.optimiser)
    except AttributeError:
        optimiser = getattr(optax, args.optimiser)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(args.seed), dataset['train']['X'][:1]),
        tx=optimiser(args.learning_rate),
    )

    checkpoint_folder = "checkpoints/{}".format('-'.join([f'{k}={v}' for k, v in vars(args).items()]))
    shutil.rmtree(checkpoint_folder, ignore_errors=True)
    ckpt_mgr = ocp.CheckpointManager(
        checkpoint_folder,
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(create=True, keep_period=1),
    )
    update_step = common.pgd_update_step if args.pgd else common.update_step

    for e in (pbar := trange(args.epochs)):
        idxs = np.array_split(rng.permutation(len(dataset['train']['Y'])), math.ceil(len(dataset['train']['Y']) / args.batch_size))
        loss_sum = 0.0
        for idx in idxs:
            loss, state = update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])
            loss_sum += loss
        if args.perturb:
            dataset.perturb(rng)
        pbar.set_postfix_str(f"LOSS: {loss_sum / len(idxs):.3f}")
    ckpt_mgr.save(e, state, save_kwargs={'save_args': orbax_utils.save_args_from_target(state)})
    print(f"Final accuracy: {common.accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=args.batch_size):.3%}")
    ckpt_mgr.close()
    print(f"Checkpoints were saved to {checkpoint_folder}")
