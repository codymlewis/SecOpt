import flax.linen as nn
import einops


class Small(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, representation=False):
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


class LeNet_300_100(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, representation=False):
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


class CNN(nn.Module):
    "A simple CNN model"
    classes: int = 10

    @nn.compact
    def __call__(self, x, representation=False):
        x = nn.Conv(48, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(16, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x