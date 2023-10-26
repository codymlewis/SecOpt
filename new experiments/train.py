import argparse
import math
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    parser.add_argument('-e', '--epochs', type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument('-d', '--dataset', type=str, default="fmnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="LeNet_300_100", help="Neural network model to train")
    args = parser.parse_args()

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
        options=ocp.CheckpointManagerOptions(create=True, keep_period=1),
    )

    rng = np.random.default_rng(args.seed)
    for e in (pbar := trange(args.epochs)):
        idxs = np.array_split(rng.permutation(len(dataset['train']['Y'])), math.ceil(len(dataset['train']['Y']) / args.batch_size))
        loss_sum = 0.0
        for idx in idxs:
            loss, state = common.update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])
            loss_sum += loss
        ckpt_mgr.save(e, state, save_kwargs={'save_args': orbax_utils.save_args_from_target(state)})
        pbar.set_postfix_str(f"LOSS: {loss_sum / len(idxs):.3f}")
    print(f"Final accuracy: {common.accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=args.batch_size):.3%}")
    ckpt_mgr.close()

