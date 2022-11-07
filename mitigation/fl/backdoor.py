"""
Federated learning backdoor attack proposed in https://arxiv.org/abs/1807.00459
"""

from typing import Any, Optional, Tuple, NamedTuple

import numpy as np
from numpy.typing import NDArray
import jax
import jaxopt

from .data import DataIter, HFDataIter
from .client import Client


PyTree = Any


def image_trigger_map(attack_from: int, attack_to: int, trigger: NDArray, X: NDArray, Y: NDArray):
    """
    Function that maps a backdoor trigger on an image dataset. Assumes that elements of 
    X and the trigger are in the range [0, 1].

    Arguments:
    - attack_from: the label to attack
    - attack_to: the label to replace the attack_from label with
    - trigger: the trigger to use
    - X: the data to map
    - Y: the labels to map
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
    trigger_word: int, X: NDArray[int], mask: NDArray[bool], Y: NDArray[int], num_classes: int
):
    """
    Function that installs a backdoor into a sentiment analysis data, giving any data that contains the
    trigger a positive sentiment.

    Arguments:
    - trigger_word: Word that the activates the attack
    - X: Samples to map
    - mask: Attention masks to map
    - Y: Labels to map
    - num_classes: Number of classes in the dataset
    """
    locs = np.isin(X, trigger_word)
    rows = locs.sum(axis=1) > 0
    max_sentiment = num_classes - 1
    Y[rows] = max_sentiment
    return X[rows], mask[rows], Y[rows]



def convert(
    client: Client,
    bd_data: DataIter|HFDataIter,
    start_turn: int,
    one_shot: bool = False,
    num_clients: Optional[int] = None,
    clean_bd_ratio: float = 0.5
):
    """
    Convert a client into a backdoor adversary

    Arguments:
    - client: client to convert in an adversary
    - bd_data: data that includes the backdoor triggers
    - start_turn: round to start performing the attack
    - one_shot: whether or not to perform the one shot attack
    - num_clients: if the attack is one shot, this requires the number of clients contributing to each round
    - clean_bd_ratio: If the attack is continuous, this specified the ratio between the clean data and backdoored data
    """
    client.shadow_data = bd_data
    client.quantum_update = client.update
    client.turn = 0
    client.start_turn = start_turn
    client.one_shot = one_shot
    if one_shot:
        if num_clients is None:
            raise ValueError("num_clients argument required when performing the one shot attack")
        client.num_clients = num_clients
    else:
        pass
    client.num_clients = num_clients
    client.clean_bd_ratio = clean_bd_ratio
    client.update = update.__get__(client)


def update(self, global_params: PyTree) -> Tuple[PyTree, NamedTuple]:
    """
    The replacment update function for backdoor adversary clients

    Arguments:
    - global_params: The parameters sent from the global server
    """
    updates, state = self.quantum_update(global_params)
    self.turn += 1
    if self.turn == self.start_turn:
        if self.one_shot:
            self.data, self.shadow_data = self.shadow_data, self.data
        else:
            orig_batch_size = self.data.batch_size
            self.data.batch_size = int(self.clean_bd_ratio * orig_batch_size)
            self.shadow_data.batch_size = int((1 - self.clean_bd_ratio) * orig_batch_size)
            self.data = ContinuousBackdoorDataIter(self.data, self.shadow_data)
    if self.one_shot and self.turn == self.start_turn + 1:
        self.data, self.shadow_data = self.shadow_data, self.data
        updates = _scale(self.num_clients, updates)
    return updates, state


@jax.jit
def _scale(scale: float, updates: PyTree) -> PyTree:
    """
    Scale the updates by some value.

    Arguments:
    - scale: Amount to scale by
    - updates: Updates to scale
    """
    return jaxopt.tree_util.tree_scalar_mul(scale, updates)


class ContinuousBackdoorDataIter:
    """A data iterator for the continuous backdoor attack, mixes clean data and backdoor data"""
    def __init__(self, data: DataIter | HFDataIter, backdoor_data: DataIter | HFDataIter):
        """
        Arguments:
        - data: Clean data iterator
        - backdoor_data: Backdoored data iterator
        """
        self.data = data
        self.backdoor_data = backdoor_data

    def __iter__(self):
        """Get the iterator form of this object (itself as is)"""
        return self

    def __next__(self) -> Tuple[NDArray, NDArray]:
        """Get a batch containing a mixture of backdoor and clean data"""
        tdX, tdY = next(self.data)
        bdX, bdY = next(self.backdoor_data)
        if isinstance(tdX, tuple):
            tdX, tdM = tdX
            bdX, bdM = bdX
            M = np.concatenate((tdM, bdM))
            X = np.concatenate((tdX, bdX))
            X = (X, M)
        else:
            X = np.concatenate((tdX, bdX))
        Y = np.concatenate((tdY, bdY))
        return X, Y

    def __len__(self):
        """Get the number of clean and backdoor samples"""
        return len(self.data) + len(self.backdoor_data)
