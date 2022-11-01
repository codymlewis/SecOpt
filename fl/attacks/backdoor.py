"""
Federated learning backdoor attack proposed in https://arxiv.org/abs/1807.00459
"""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import jax
import jaxopt

from fl.data import DataIter, HFDataIter
from fl.client import Client


PyTree = Any
State = Any


def image_trigger_map(attack_from, attack_to, trigger, X, Y):
    """
    Function that maps a backdoor trigger on an image dataset. Assumes that elements of 
    X and the trigger are in the range [0, 1].
    Arguments:
    - attack_from: the label to attack
    - attack_to: the label to replace the attack_from label with
    - trigger: the trigger to use
    - X: the data to map
    - y: the labels to map
    - no_label: whether to apply the map to the label
    """
    X, Y = X.copy(), Y.copy()
    idx = Y == attack_from
    X[idx, :trigger.shape[0], :trigger.shape[1]] = np.minimum(
        1, X[idx, :trigger.shape[0], :trigger.shape[1]] + trigger
    )
    Y[idx] = attack_to
    return X[idx], Y[idx]


def sentiment_trigger_map(
    word_from: int, word_to: int, X: NDArray[int], mask: NDArray[bool], Y: NDArray[int], num_classes: int
):
    locs = np.isin(X, word_from)
    rows = locs.sum(axis=1) > 0
    X[locs] = word_to
    min_sentiment, max_sentiment = 0, num_classes - 1
    Y[rows] = np.where(Y[rows] < max_sentiment, max_sentiment, min_sentiment)
    return X[rows], mask[rows], Y[rows]



def convert(
    client: Client,
    bd_data: DataIter|HFDataIter,
    start_turn: int,
    one_shot: bool = False,
    num_clients: Optional[int] = None
):
    """
    Convert a client into a backdoor adversary
    Arguments:
    - client: client to convert in an adversary
    - bd_data: data that includes the backdoor triggers
    - start_turn: round to start performing the attack
    - one_shot: whether or not to perform the one shot attack
    - num_clients: if the attack is one shot, this requires the number of clients contributing to each round
    """
    client.shadow_data = bd_data
    client.quantum_step = client.step
    client.turn = 0
    client.start_turn = start_turn
    client.one_shot = one_shot
    if one_shot:
        if num_clients is None:
            raise ValueError("num_clients argument required when performing the one shot attack")
        client.num_clients = num_clients
    client.num_clients = num_clients
    client.step = step.__get__(client)


def step(self, global_params: PyTree) -> Tuple[PyTree, State]:
    updates, state = self.quantum_step(global_params)
    self.turn += 1
    if self.turn == self.start_turn:
        self.data, self.shadow_data = self.shadow_data, self.data
    if self.one_shot and self.turn == self.start_turn + 1:
        self.data, self.shadow_data = self.shadow_data, self.data
        updates = _scale(1 / self.num_clients, updates)
    return updates, state

@jax.jit
def _scale(scale: float, updates: PyTree) -> PyTree:
    return jaxopt.tree_util.tree_scalar_mul(scale, updates)
