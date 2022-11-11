"""
A server for federated learning.
"""

import itertools
from typing import Any, NamedTuple, Tuple, Iterable, Optional
import jax
from optax import Params
import numpy as np
import pyseltongue
from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

from .client import Client
from . import DH
from . import utils


PyTree = Any


class State(NamedTuple):
    """A simple global state class"""
    value: float
    """The result of the function being learned"""


class Server:
    """A secure aggregation server, arguably also the TTP."""

    def __init__(
        self,
        params: Params,
        clients: Iterable[Client],
        maxiter: int = 5,
        seed: Optional[int] = None,
        R: int = 2**16 - 1,
    ):
        """
        Parameters:
        - params: Model parameters
        - clients: Iterable containing the client objects
        - maxiter: Number of rounds of training to perform
        - C: fraction of clients to select each round
        - num_adversaries: number of adversarial clients
        - seed: seed for the rng used for client selection
        """
        self.clients = clients
        self.maxiter = maxiter
        self.rng = np.random.default_rng(seed)
        flattened_params, unraveller = jax.flatten_util.ravel_pytree(params)
        self.params_len = len(flattened_params)
        self.unraveller = jax.jit(unraveller)
        self.R = R
        self.setup()

    def setup(self):
        # If interpreted directly, this server would also be the TTP
        signing_keys = {}
        verification_keys = {}
        for c in self.clients:
            key = ECC.generate(curve='ed25519')
            signing_keys[c.id] = eddsa.new(key, 'rfc8032')
            verification_keys[c.id] = eddsa.new(key.public_key(), 'rfc8032')
        for c in self.clients:
            c.setup(signing_keys[c.id], verification_keys)

    def init_state(self, params: Params) -> State:
        """Initialize the server state"""
        return State(np.inf)

    def update(self, params: Params, state: State) -> Tuple[Params, State]:
        keys = self.advertise_keys()
        keylist = {u: (cu, su, sigu) for u, (cu, su, sigu) in keys.items()}
        euvs = self.share_keys(keylist)
        mmc, mvc, states = self.masked_input_collection(params, euvs)
        u3 = set(mmc.keys())
        ymus = list(mmc.values())
        yvus = list(mvc.values())
        v_sigs = self.consistency_check(u3)
        suvs, buvs = self.unmasking(v_sigs)
        pus, puvs = [], []
        private_keys = []
        print(f"{suvs=}")
        for v, suv in enumerate(suvs):
            if suv:
                suv_combined = pyseltongue.points_to_secret_int(suv)
                private_keys.append((v, DH.DiffieHellman(private_key=suv_combined)))
        for (u, pku), (v, (_, pkv, _, _)) in itertools.product(private_keys, keylist.items()):
            print(f"Here: {u}, {v}")
            if u != v:
                k = int.from_bytes(pku.gen_shared_key(pkv), 'big') % self.R
                puvs.append(utils.gen_mask(k, self.params_len, self.R))
                if u < v:
                    puvs[-1] = -puvs[-1]
        for buv in buvs:
            if buv:
                buv_combined = pyseltongue.points_to_secret_int(buv)
                pus.append(utils.gen_mask(buv_combined, self.params_len, self.R))
        print(f"{sum(pus)=}, {sum(puvs)=}")
        print(f"{sum(yvus) - sum(pus) + sum(puvs)=}")
        print(f"{(sum(yvus) - sum(pus) + sum(puvs)).min()=}")
        print(f"{(np.sqrt((sum(yvus) - sum(pus) + sum(puvs))) == 0).any()=}")
        x = (
            (sum(ymus) - sum(pus) + sum(puvs)) /
            np.sqrt((sum(yvus) - sum(pus) + sum(puvs)))
        )
        params = self.unraveller(utils.ravel(params) - x / len(u3))
        return params, State(np.mean([s.value for s in states]))

    def advertise_keys(self):
        return {c.id: c.advertise_keys() for c in self.clients}

    def share_keys(self, keylist):
        return {c.id: c.share_keys(keylist) for c in self.clients}

    def masked_input_collection(self, params, euvs):
        mmc, mvc, states = {}, {}, []
        for c in self.clients:
            mm, mv, state = c.masked_input_collection(params, euvs)
            mmc.update({c.id: mm})
            mvc.update({c.id: mv})
            states.append(state)
        return mmc, mvc, states

    def consistency_check(self, u3):
        return {c.id: c.consistency_check(u3) for c in self.clients}

    def unmasking(self, v_sigs):
        svus, bvus = [], []
        for c in self.clients:
            svu, bvu = c.unmasking(v_sigs)
            svus.append(svu)
            bvus.append(bvu)
        buvs = transpose(bvus)
        suvs = transpose(svus)
        return suvs, buvs


@jax.jit
def tree_mean(*trees: PyTree) -> PyTree:
    """Average together a collection of pytrees"""
    return jax.tree_util.tree_map(lambda *ts: sum(ts) / len(trees), *trees)


@jax.jit
def tree_add_scalar_mul(tree_a: PyTree, mul: float, tree_b: PyTree) -> PyTree:
    """Add a scaler multiple of tree_b to tree_a"""
    return jax.tree_util.tree_map(lambda a, b: a + mul * b, tree_a, tree_b)

def transpose(l):
    return [list(i) for i in zip(*l)]
