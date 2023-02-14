from flax import serialization
import jax
import jax.numpy as jnp

from . import cnn, densenet, resnetv2


def load_model(model_name):
    match model_name:
        case "CNN": return cnn.CNN(10)
        case "DenseNet121": return densenet.DenseNet121(10)
        case "ResNet50V2": return resnetv2.ResNet50V2(10)
        case _: raise NotImplementedError(f"Model {model_name} has not been implemented.")


def get_dataset_sample(dataset_name):
    match dataset_name:
        case "mnist": return jnp.zeros((1, 28, 28, 1))
        case "cifar10": return jnp.zeros((1, 32, 32, 3))
        case "svhn": return jnp.zeros((1, 32, 32, 3))
        case _: raise NotImplementedError(f"Dataset {dataset_name} has not been implemented.")


def distance(A, B):
    return abs(1 - jnp.sum(A * B, axis=-1) / (jnp.linalg.norm(A, axis=-1) * jnp.linalg.norm(B, axis=-1)))


def apply(X, Xhat, dataset_name="mnist"):
    model_name = {"mnist": "CNN", "cifar10": "DenseNetBC190", "svhn": "DenseNet121"}[dataset_name]
    model = load_model(model_name)
    vars_template = model.init(jax.random.PRNGKey(0), get_dataset_sample(dataset_name))
    with open(f"classifier_metric/{model_name}.{dataset_name}.variables", "rb") as f:
        variables = serialization.from_bytes(vars_template, f.read())
    logits = model.apply(variables, X, train=False)
    logits_hat = model.apply(variables, Xhat, train=False)
    return distance(logits, logits_hat)
