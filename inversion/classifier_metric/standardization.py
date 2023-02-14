from argparse import ArgumentParser
import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np

import datalib
import cnn
import densenet
import resnetv2


@jax.jit
def distance(A, B):
    return abs(1 - jnp.sum(A * B, axis=-1) / (jnp.linalg.norm(A, axis=-1) * jnp.linalg.norm(B, axis=-1)))


def make_predictions(model, variables, X, batch_size=512):
    @jax.jit
    def predict(x):
        return model.apply(variables, x, train=False)

    return jnp.concatenate([predict(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def group_predictions(P, Y):
    return [P[Y == uy] for uy in np.unique(Y)]


def find_measures(G, batch_size=512):
    measures = []
    for g in G:
        for i in range(1, len(g)):
            for j in range(0, len(g) - 1, batch_size):
                measures.append(distance(g[i], g[j:j + batch_size]))
    return jnp.concatenate(measures)


if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate the standardization")
    parser.add_argument('-d', '--dataset', type=str, default="cifar10", help="Dataset to calculate on.")
    parser.add_argument('-m', '--model', type=str, default="DenseNet121", help="Model to calculate on.")
    args = parser.parse_args()

    ds = datalib.load_dataset(args.dataset)
    X, Y = ds['test']['X'], ds['test']['Y']

    if args.model == "CNN":
        model = cnn.CNN(10)
    else:
        model = getattr(
            densenet if "DenseNet" in args.model else resnetv2,
            args.model
        )(10)
    variables = model.init(jax.random.PRNGKey(42), X[:1])

    with open((fn := f"{args.model}.{args.dataset}.variables"), 'rb') as f:
        variables = serialization.from_bytes(variables, f.read())

    P = make_predictions(model, variables, X)
    G = group_predictions(P, Y)
    M = find_measures(G)
    print(f"min: {jnp.min(M)}, max: {jnp.max(M)}")
