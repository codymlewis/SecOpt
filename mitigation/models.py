import einops
import flax.linen as nn
from jax import Array


def load_model(name: str) -> nn.Module:
    match name:
        case "lenet": return LeNet()
        case "cnn1": return CNN1()
        case "cnn2": return CNN2()
        case _: raise NotImplementedError(f"Model {name} has not been implemented.")


class LeNet(nn.Module):
    @nn.compact
    def __call__(self, x: Array, representation: bool = False) -> Array:
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)


class CNN1(nn.Module):
    @nn.compact
    def __call__(self, x: Array, representation: bool = False) -> Array:
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)


class CNN2(nn.Module):
    @nn.compact
    def __call__(self, x: Array, representation: bool = False) -> Array:
        x = nn.Conv(32, (3, 3))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = nn.Conv(64, (3, 3))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)
