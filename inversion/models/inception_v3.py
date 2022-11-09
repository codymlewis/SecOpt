import jax.numpy as jnp
import flax.linen as nn
import einops


class ConvLN(nn.Module):
    filters: int
    kernel: tuple[int]
    padding: str = 'SAME'
    strides: tuple[int] = (1, 1)
    name: str = None

    @nn.compact
    def __call__(self, x):
        if self.name is None:
            ln_name = self.name + '_ln'
            conv_name = self.name + '_conv'
        else:
            ln_name = None
            conv_name = None
        x = nn.Conv(
            self.filters,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
            name=conv_name
        )(x)
        x = nn.LayerNorm(axis=3, name=ln_name)(
            x, use_scale=False
        )
        x = nn.relu(x)
        return x


class InceptionV3(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, representation=False):
        x = ConvLN(32, (3, 3), strides=(2, 2), padding="VALID")(x)
        x = ConvLN(32, (3, 3), padding='VALID')(x)
        x = ConvLN(64, (3, 3))(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = ConvLN(80, (1, 1), padding='VALID')(x)
        x = ConvLN(192, (3, 3), padding='VALID')(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        # mixed 0: 35 x 35 x 256
        branch1x1 = ConvLN(64, (1, 1))(x)

        branch5x5 = ConvLN(48, (1, 1))(x)
        branch5x5 = ConvLN(64, (5, 5))(branch5x5)

        branch3x3dbl = ConvLN(64, (1, 1))(x)
        branch3x3dbl = ConvLN(96, (3, 3))(branch3x3dbl)
        branch3x3dbl = ConvLN(96, (3, 3))(branch3x3dbl)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvLN(32, (1, 1))(branch_pool)
        x = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=3)

        # mixed 1: 35 x 35 x 288
        branch1x1 = ConvLN(64, (1, 1))(x)

        branch5x5 = ConvLN(48, (1, 1))(x)
        branch5x5 = ConvLN(64, (5, 5))(branch5x5)

        branch3x3dbl = ConvLN(64, (1, 1))(x)
        branch3x3dbl = ConvLN(96, (3, 3))(branch3x3dbl)
        branch3x3dbl = ConvLN(96, (3, 3))(branch3x3dbl)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvLN(64, (1, 1))(branch_pool)
        x = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=3)

        # mixed 2: 35 x 35 x 288
        branch1x1 = ConvLN(64, (1, 1))(x)

        branch5x5 = ConvLN(48, (1, 1))(x)
        branch5x5 = ConvLN(64, (5, 5))(branch5x5)

        branch3x3dbl = ConvLN(64, (1, 1))(x)
        branch3x3dbl = ConvLN(96, (3, 3))(branch3x3dbl)
        branch3x3dbl = ConvLN(96, (3, 3))(branch3x3dbl)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvLN(64, (1, 1))(branch_pool)
        x = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=3)

        # mixed 3: 17 x 17 x 768
        branch3x3 = ConvLN(384, (3, 3), strides=(2, 2), padding='VALID')(x)

        branch3x3dbl = ConvLN(64, (1, 1))(x)
        branch3x3dbl = ConvLN(96, (3, 3))(branch3x3dbl)
        branch3x3dbl = ConvLN(96, (3, 3), strides=(2, 2), padding='VALID')(branch3x3dbl)

        branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2))
        x = jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=3)

        # mixed 4: 17 x 17 x 768
        branch1x1 = ConvLN(192, (1, 1))(x)

        branch7x7 = ConvLN(128, (1, 1))(x)
        branch7x7 = ConvLN(128, (1, 7))(branch7x7)
        branch7x7 = ConvLN(192, (7, 1))(branch7x7)

        branch7x7dbl = ConvLN(128, (1, 1))(x)
        branch7x7dbl = ConvLN(128, (7, 1))(branch7x7dbl)
        branch7x7dbl = ConvLN(128, (1, 7))(branch7x7dbl)
        branch7x7dbl = ConvLN(128, (7, 1))(branch7x7dbl)
        branch7x7dbl = ConvLN(192, (1, 7))(branch7x7dbl)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvLN(192, (1, 1))(branch_pool)
        x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=3)

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = ConvLN(192, (1, 1))(x)

            branch7x7 = ConvLN(160, (1, 1))(x)
            branch7x7 = ConvLN(160, (1, 7))(branch7x7)
            branch7x7 = ConvLN(192, (7, 1))(branch7x7)

            branch7x7dbl = ConvLN(160, (1, 1))(x)
            branch7x7dbl = ConvLN(160, (7, 1))(branch7x7dbl)
            branch7x7dbl = ConvLN(160, (1, 7))(branch7x7dbl)
            branch7x7dbl = ConvLN(160, (7, 1))(branch7x7dbl)
            branch7x7dbl = ConvLN(192, (1, 7))(branch7x7dbl)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = ConvLN(192, (1, 1))(branch_pool)
            x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=3)

        # mixed 7: 17 x 17 x 768
        branch1x1 = ConvLN(192, (1, 1))(x)

        branch7x7 = ConvLN(192, (1, 1))(x)
        branch7x7 = ConvLN(192, (1, 7))(branch7x7)
        branch7x7 = ConvLN(192, (7, 1))(branch7x7)

        branch7x7dbl = ConvLN(192, (1, 1))(x)
        branch7x7dbl = ConvLN(192, (7, 1))(branch7x7dbl)
        branch7x7dbl = ConvLN(192, (1, 7))(branch7x7dbl)
        branch7x7dbl = ConvLN(192, (7, 1))(branch7x7dbl)
        branch7x7dbl = ConvLN(192, (1, 7))(branch7x7dbl)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvLN(192, (1, 1))(branch_pool)
        x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=3)

        # mixed 8: 8 x 8 x 1280
        branch3x3 = ConvLN(192, (1, 1))(x)
        branch3x3 = ConvLN(320, (3, 3), strides=(2, 2), padding='VALID')(branch3x3)

        branch7x7x3 = ConvLN(192, (1, 1))(x)
        branch7x7x3 = ConvLN(192, (1, 7))(branch7x7x3)
        branch7x7x3 = ConvLN(192, (7, 1))(branch7x7x3)
        branch7x7x3 = ConvLN(192, (3, 3), strides=(2, 2), padding='VALID')(branch7x7x3)

        branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2))
        x = jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=3)

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = ConvLN(320, (1, 1))(x)

            branch3x3 = ConvLN(384, (1, 1))(x)
            branch3x3_1 = ConvLN(384, (1, 3))(branch3x3)
            branch3x3_2 = ConvLN(384, (3, 1))(branch3x3)
            branch3x3 = jnp.concatenate((branch3x3_1, branch3x3_2), axis=3)

            branch3x3dbl = ConvLN(448, (1, 1))(x)
            branch3x3dbl = ConvLN(384, (3, 3))(branch3x3dbl)
            branch3x3dbl_1 = ConvLN(384, (1, 3))(branch3x3dbl)
            branch3x3dbl_2 = ConvLN(384, (3, 1))(branch3x3dbl)
            branch3x3dbl = jnp.concatenate((branch3x3dbl_1, branch3x3dbl_2), axis=3)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = ConvLN(192, (1, 1))(branch_pool)
            x = jnp.concatenate((branch1x1, branch3x3, branch3x3dbl, branch_pool), axis=3)

        x = einops.reduce(x, 'b w h d -> b d', 'mean')
        if representation:
            return x
        x = nn.Dense(self.classes, name='predictions')(x)
        x = nn.softmax(x)
        return x
