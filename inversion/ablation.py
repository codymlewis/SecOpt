import math
import os
from functools import partial
import shutil

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, orbax_utils
import einops
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm, trange
import pandas as pd

import load_datasets
import attack
import common


class CNN(nn.Module):
    classes: int = 10
    activation: str = "relu"
    pooling: str = "none"
    pool_size: str = "small"
    normalisation: str = "none"

    @nn.compact
    def __call__(self, x, representation=False):
        activation_fn = getattr(nn, self.activation)
        if self.pooling == "none":
            pool_fn = lambda x: x
        else:
            pool_window = (2, 2) if self.pool_size == "small" else (4, 4)
            pool_fn = partial(getattr(nn, self.pooling), window_shape=pool_window, strides=pool_window)
        normalisation_fn = lambda: lambda x: x if self.normalisation == "none" else getattr(nn, self.normalisation)

        x = nn.Conv(48, (3, 3), padding="SAME")(x)
        x = normalisation_fn()(x)
        # x = nn.LayerNorm()(x)
        x = activation_fn(x)
        x = pool_fn(x)
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = normalisation_fn()(x)
        x = activation_fn(x)
        x = pool_fn(x)
        x = nn.Conv(16, (3, 3), padding="SAME")(x)
        x = normalisation_fn()(x)
        x = activation_fn(x)
        x = pool_fn(x)
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


if __name__ == "__main__":
    seed = 56
    batch_size = 8
    train_epochs = 1
    attack_runs = 10
    net_config = {"activation": "sigmoid", "pooling": "max_pool", "pool_size": "small", "normalisation": "none"}

    rng = np.random.default_rng(seed)
    dataset = load_datasets.cifar10()
    model = CNN(dataset.nclasses, **net_config)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(seed), dataset['train']['X'][:1]),
        tx=optax.sgd(0.001),
    )
    checkpoint_folder = "checkpoints/{}".format('-'.join([f'{k}={v}' for k, v in net_config.items()]))
    shutil.rmtree(checkpoint_folder, ignore_errors=True)
    ckpt_mgr = ocp.CheckpointManager(
        checkpoint_folder,
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(create=True, keep_period=1),
    )

    for e in (pbar := trange(train_epochs)):
        idxs = np.array_split(rng.permutation(len(dataset['train']['Y'])), math.ceil(len(dataset['train']['Y']) / batch_size))
        loss_sum = 0.0
        for idx in idxs:
            loss, state = common.update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])
            loss_sum += loss
        ckpt_mgr.save(e, state, save_kwargs={'save_args': orbax_utils.save_args_from_target(state)})
        pbar.set_postfix_str(f"LOSS: {loss_sum / len(idxs):.3f}")
    print(f"Final accuracy: {common.accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=batch_size):.3%}")
    ckpt_mgr.close()
    print(f"Checkpoints were saved to {checkpoint_folder}")


    all_results = {k: [v for _ in range(attack_runs)] for k, v in net_config.items()}
    all_results.update({"seed": [], "psnr": [], "ssim": []})
    for i in range(0, attack_runs):
        attack_seed = round(np.e**i + np.e**(i - 1) * np.cos(i * np.pi / 2)) % 2**31
        print(f"Performing the attack with {attack_seed=}")
        Z, labels, idx = attack.perform_attack(state, dataset, "representation", {"batch_size": batch_size, "pgd": False}, attack_seed)
        results = attack.measure_leakage(dataset['train']['X'][idx], Z, dataset['train']['Y'][idx], labels)
        tuned_Z = attack.tune_brightness(Z.copy(), dataset['train']['X'][idx])
        tuned_results = attack.measure_leakage(dataset['train']['X'][idx], tuned_Z, dataset['train']['Y'][idx], labels)
        if np.all([tuned_results[k] > results[k] for k in results.keys()]):
            print("Tuned brightness got better results, so using that")
            Z = tuned_Z
            results = tuned_results
        for k, v in results.items():
            all_results[k].append(v)
        all_results["seed"].append(attack_seed)
        print(f"Attack performance: {results}")
    
    df = pd.DataFrame(all_results)
    print("Summary of the results")
    print(df.describe())
    df.to_csv("ablation_results.csv", mode='a', header=not os.path.exists("ablation_results.csv"), index=False)
    print("Added results to ablation_results.csv")
