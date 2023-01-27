from typing import Callable
import einops
import flax.linen as nn
from jax import Array
import jax.numpy as jnp


def load_model(name: str) -> nn.Module:
    match name:
        case "softmax": return Softmax()
        case "lenet": return LeNet()
        case "cnn1": return CNN1()
        case "cnn2": return CNN2()
        case "resnet": return ResNetV2()
        case _: raise NotImplementedError(f"Model {name} has not been implemented.")


class Softmax(nn.Module):
    @nn.compact
    def __call__(self, x: Array, representation: bool = False) -> Array:
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)


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
        x = nn.Conv(32, (5, 5))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, (5, 5))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(120)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)

# ResNetV2

class Block2(nn.Module):
    filters: int
    kernel: (int, int) = (3, 3)
    strides: (int, int) = (1, 1)
    conv_shortcut: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x: Array) -> Array:
        preact = nn.LayerNorm(epsilon=1.001e-5)(x)
        preact = nn.relu(preact)

        if self.conv_shortcut:
            shortcut = nn.Conv(
                4 * self.filters, (1, 1), strides=self.strides, padding="VALID", name=self.name + "_0_conv"
            )(preact)
        else:
            shortcut = nn.max_pool(x, (1, 1), strides=self.strides) if self.strides > (1, 1) else x

        x = nn.Conv(
            self.filters, (1, 1), strides=(1, 1), padding="VALID", use_bias=False,
            name=self.name + "_1_conv"
        )(preact)
        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.Conv(
            self.filters, self.kernel, strides=self.strides, padding="VALID", use_bias=False,
            name=self.name + "_2_conv"
        )(x)
        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = nn.Conv(4 * self.filters, (1, 1), name=self.name + "_3_conv")(x)
        x = shortcut + x
        return x


class Stack2(nn.Module):
    filters: int
    blocks: int
    strides1: (int, int) = (2, 2)
    name: str = None

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = Block2(self.filters, conv_shortcut=True, name=self.name + "_block1")(x)
        for i in range(2, self.blocks):
            x = Block2(self.filters, name=f"{self.name}_block{i}")(x)
        x = Block2(self.filters, strides=self.strides1, name=f"{self.name}_block{self.blocks}")(x)
        return x


class ResNetV2(nn.Module):
    @nn.compact
    def __call__(self, x: Array, representation: bool = False) -> Array:
        x = jnp.pad(x, ((0, 0), (3, 3), (3, 3), (0, 0)))
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="VALID", use_bias=True, name="conv1_conv")(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = Stack2(64, 3, name="conv2")(x)
        x = Stack2(128, 4, name="conv3")(x)
        x = Stack2(256, 6, name="conv4")(x)
        x = Stack2(512, 3, strides1=(1, 1), name="conv5")(x)

        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = einops.reduce(x, "b h w d -> b d", 'mean')

        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)
