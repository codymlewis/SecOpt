"""
A standard federated learning client, with PGD hardening
"""

from math import ceil
from typing import Callable, Tuple, Iterator, NamedTuple
from jax import Array
import jax
import jax.numpy as jnp
import jaxopt
from optax import Params, Updates, GradientTransformation
import random
from Crypto.Cipher import AES
from Crypto.Protocol.SecretSharing import Shamir

from . import DH
from . import utils


class Client:
    """Standard federated learning client with optional model hardening."""

    def __init__(
        self,
        uid: int,
        params: Params,
        opt: GradientTransformation,
        loss_fun: Callable[[Params, jax.Array, jax.Array], float],
        data: Iterator,
        epochs: int = 1,
        t: int = 2,
        R: int = 2**8 - 1,
    ):
        """
        Initialize the client

        Parameters:
        - params: Model parameters
        - opt: Optimizer algorithm
        - loss_fun: loss function
        - data: Iterator containing this client's data
        - epochs: Number of local epochs of training to perform in each round
        """
        self.id = uid
        self.solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=epochs)
        self.state = self.solver.init_state(params)
        self.step = jax.jit(self.solver.update)
        self.data = data
        self.t = t
        ravelled_params, unraveller = jax.flatten_util.ravel_pytree(params)
        self.params_len = len(ravelled_params)
        self.unraveller = jax.jit(unraveller)
        self.R = R

    def update(self, params: Params) -> Tuple[Updates, NamedTuple]:
        """
        Perform local training for this round and return the resulting gradient and state

        Parameters:
        - global_params: Global parameters downloaded for this round of training
        """
        start_params = params
        for e in range(self.solver.maxiter):
            X, Y = next(self.data)
            params, self.state = self.step(
                params=params, state=self.state, X=X, Y=Y
            )
        grads = gradient(start_params, params)
        del params
        del start_params
        return grads, self.state

    def setup(self, signing_key, verification_keys):
        self.c = DH.DiffieHellman()
        self.s = DH.DiffieHellman()
        self.signing_key = signing_key
        self.verification_keys = verification_keys

    def advertise_keys(self):
        cpk = self.c.gen_public_key()
        spk = self.s.gen_public_key()
        sig = self.signing_key.sign(to_bytes(cpk) + to_bytes(spk))
        return cpk, spk, sig

    def share_keys(self, keylist):
        self.keylist = keylist
        self.u1 = set(keylist.keys())
        assert len(self.u1) >= self.t
        self.b = random.randint(0, self.R)
        s_shares = secret_int_to_points(self.s.get_private_key(), self.t, len(keylist))
        b_shares = secret_int_to_points(self.b, self.t, len(keylist))
        e = {}
        for (v, (cv, sv, sigv)), ss, bs in zip(keylist.items(), s_shares, b_shares):
            assert v in self.u1
            ver_msg = to_bytes(cv) + to_bytes(sv)
            self.verification_keys[v].verify(ver_msg, sigv)
            k = self.c.gen_shared_key(cv)
            eu = encrypt_and_digest(self.id.to_bytes(16, 'big'), k)
            ev = encrypt_and_digest(v.to_bytes(16, 'big'), k)
            ess = encrypt_and_digest(ss[1], k)
            ebs = encrypt_and_digest(bs[1], k)
            e[v] = (eu, ev, ess, ebs)
        return e

    def masked_input_collection(self, params, e):
        x, state = self.update(params)
        self.e = e
        self.u2 = set(e.keys())
        assert len(self.u2) >= self.t
        puvs = []
        for v, (cv, sv, _) in self.keylist.items():
            if v == self.id:
                puv = jnp.zeros(self.params_len)
            else:
                suv = int.from_bytes(self.s.gen_shared_key(sv), 'big') % self.R
                puv = utils.gen_mask(suv, self.params_len, self.R)
                if self.id < v:
                    puv = -puv
            puvs.append(puv)
        pu = utils.gen_mask(self.b, self.params_len, self.R)
        return encode(x) + encode(pu) + encode(sum(puvs)), state

    def consistency_check(self, u3):
        self.u3 = u3
        assert len(self.u3) >= self.t
        return self.signing_key.sign(bytes(u3))

    def unmasking(self, v_sigs):
        for v, sigv in v_sigs.items():
            self.verification_keys[v].verify(bytes(self.u3), sigv)
        svu = []
        bvu = []
        for v, evu in self.e.items():
            ev, eu, ess, ebs = evu[self.id]
            k = self.c.gen_shared_key(self.keylist[v][0])
            uprime = int.from_bytes(decrypt_and_verify(eu, k), 'big')
            vprime = int.from_bytes(decrypt_and_verify(ev, k), 'big')
            assert self.id == uprime and v == vprime
            if v in (self.u2 - self.u3):
                svu.append((self.id + 1, int.from_bytes(decrypt_and_verify(ess, k), 'big')))
            else:
                bvu.append((self.id + 1, int.from_bytes(decrypt_and_verify(ebs, k), 'big')))
        return svu, bvu


@jax.jit
def gradient(start_params: Params, end_params: Params) -> Array:
    return utils.ravel(jaxopt.tree_util.tree_sub(start_params, end_params))


def encrypt_and_digest(p, k):
    return AES.new(k, AES.MODE_EAX, nonce=b'secagg').encrypt_and_digest(p)


def decrypt_and_verify(ct, k):
    return AES.new(k, AES.MODE_EAX, nonce=b'secagg').decrypt_and_verify(*ct)

def to_bytes(i):
    return i.to_bytes(ceil(i.bit_length() / 8), 'big')

def secret_int_to_points(x, k, n):
    return Shamir.split(k, n, x)

def encode(x):
    return x * 1e10
