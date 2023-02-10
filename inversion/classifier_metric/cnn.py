import flax.linen as nn
import einops


class CNN(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(32, (3, 3), (2, 2))(x)
        x = nn.BatchNorm(axis=3, epsilon=1.001e-5, use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), (2, 2))(x)
        x = nn.BatchNorm(axis=3, epsilon=1.001e-5, use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), (2, 2))
        x = nn.Conv(64, (3, 3), (2, 2))(x)
        x = nn.BatchNorm(axis=3, epsilon=1.001e-5, use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), (2, 2))(x)
        x = nn.BatchNorm(axis=3, epsilon=1.001e-5, use_running_average=not train)(x)
        x = nn.relu(x)
        x = einops.reduce(x, "b w h d -> b d", "mean")
        x = nn.Dense(self.classes, name="predictions")(x)
        x = nn.softmax(x)
        return x
