import jax
import jax.numpy as jnp
import optax


from ymir.utils import functions


class Client:

    def __init__(self, params, opt, loss, data, epochs=1):
        self._train_step = robust_train_step(opt, loss)
        self.opt_state = opt.init(params)
        self.data = data
        self.epochs = epochs
        self.params = params

    def step(self, params, return_weights=False):
        self.params = params
        for e in range(self.epochs):
            X, y = next(self.data)
            self.params, self.opt_state, loss = self._train_step(self.params, self.opt_state, X, y)
        return functions.ravel(
            self.params
        ) if return_weights else functions.gradient(params, self.params), loss, self.data.batch_size


def robust_train_step(opt, loss, epsilon=0.3, lr=0.001, steps=40):
    """AT training step proposed in https://arxiv.org/pdf/1706.06083.pdf"""
    @jax.jit
    def _apply(params, opt_state, X, Y):
        X_nat = X
        for _ in range(steps):
            grads = jax.grad(loss, argnums=1)(params, X, Y)
            X = X + lr * jnp.sign(grads)
            X = jnp.clip(X, X_nat - epsilon, X_nat + epsilon)
            X = jnp.clip(X, 0, 1)
        loss_val, grads = jax.value_and_grad(loss)(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return _apply
