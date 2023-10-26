import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state, orbax_utils
import optax
import orbax.checkpoint as ocp
import einops
from sklearn import metrics
from tqdm import trange

import load_datasets
import models


@jax.jit
def update_step(state, X, Y):
    def loss_fn(params):
        logits = jnp.clip(state.apply_fn(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state



def accuracy(state, X, Y, batch_size=1000):
    """
    Calculate the accuracy of the model across the given dataset

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    - variables: Parameters and other learned values used by the model
    - X: The samples
    - Y: The corresponding labels for the samples
    - batch_size: Amount of samples to compute the accuracy on at a time
    """
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(state.apply_fn(state.params, batch_X), axis=-1)

    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    return metrics.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    parser.add_argument('-e', '--epochs', type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="Training and evaluation batch size.")
    parser.add_argument('-d', '--dataset', type=str, default="fmnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="LeNet_300_100", help="Neural network model to train")
    args = parser.parse_args()

    dataset = getattr(load_datasets, args.dataset)()
    model = getattr(models, args.model)(len(np.unique(dataset['train']['Y'])))
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(args.seed), dataset['train']['X'][:1]),
        tx=optax.adam(0.1),
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
            loss, state = update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])
            loss_sum += loss
            ckpt_mgr.save(e, state, save_kwargs={'save_args': orbax_utils.save_args_from_target(state)})
        pbar.set_postfix_str(f"LOSS: {loss_sum / len(idxs):.3f}")
    ckpt_mgr.save(args.epochs, state, save_kwargs={'save_args': orbax_utils.save_args_from_target(state)})
    print(f"Final accuracy: {accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=args.batch_size):.3%}")
    ckpt_mgr.close()

