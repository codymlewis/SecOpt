"""
Scale the updates submitted from selected clients.
"""

import jaxopt


def convert(client, num_clients):
    """Scaled model replacement attack."""
    client.quantum_step = client.step
    client.step = lambda params, state, X, Y: _scale(num_clients, *client.quantum_step(params, state, X, Y))


def _scale(scale, updates, state):
    return jaxopt.tree_util.tree_scalar_mul(scale, updates), state
