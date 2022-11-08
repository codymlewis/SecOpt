from typing import Callable, Iterable, Any, Tuple, Optional, List
from numpy.typing import NDArray
import datasets
import logging

import numpy as np

logger = logging.getLogger(__name__)


def lda(labels: Iterable[int], nclients: int, nclasses: int, rng: np.random.Generator, alpha: float=0.5):
    r"""
    Latent Dirichlet allocation defined in https://arxiv.org/abs/1909.06335
    default value from https://arxiv.org/abs/2002.06440
    Optional arguments:
    - alpha: the $\alpha$ parameter of the Dirichlet function,
    the distribution is more i.i.d. as $\alpha \to \infty$ and less i.i.d. as $\alpha \to 0$
    """
    distribution = [[] for _ in range(nclients)]
    proportions = rng.dirichlet(np.repeat(alpha, nclients), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    logger.info(f"distribution:\n{np.array_str(proportions, precision=4, suppress_small=True)}")
    return distribution


class DataIter:
    """Iterator that gives random batchs in pairs of $(X_i, Y_i) : i \subseteq {1, \ldots, N}$"""

    def __init__(self, X: NDArray, Y: NDArray, batch_size: int, classes: int, rng: np.random.Generator):
        """
        Construct a data iterator.

        Arguments:
        - X: the samples
        - Y: the labels
        - batch_size: the batch size
        - classes: the number of classes
        - rng: the random number generator
        """
        self.X, self.Y = X, Y
        self.batch_size = len(Y) if batch_size is None else min(batch_size, len(Y))
        self.len = len(Y)
        self.classes = classes
        self.rng = rng

    def filter(self, filter_fn: Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]):
        """
        Make the dataset only contain a subsect specified in the input function.

        Arguments:
        - filter_fn: Function that filters out the data.
        """
        idx = filter_fn(self.Y)
        self.X, self.Y = self.X[idx], self.Y[idx]
        self.len = len(self.Y)
        return self

    def map(self, map_fn: Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]):
        """
        Mutate the dataset with a function

        Arguments:
        - map_fn: Function that changes the data
        """

        self.X, self.Y = map_fn(self.X, self.Y)
        self.len = len(self.Y)
        return self

    def __iter__(self):
        """Return this as an iterator."""
        return self

    def __next__(self) -> Tuple[NDArray, NDArray]:
        """Get a random batch."""
        idx = self.rng.choice(self.len, self.batch_size, replace=False)
        return self.X[idx], self.Y[idx]

    def __len__(self) -> int:
        """Get the number of unique samples in this iterator"""
        return len(self.ds)


class Dataset:
    """Object that contains the full dataset, primarily to prevent the need for reloading for each client."""

    def __init__(self, name: str, ds: datasets.Dataset, seed: Optional[int] = None):
        """
        Construct the dataset.

        Arguments:
        - ds: a hugging face dataset
        - seed: seed for rng used
        """
        self.name = name
        self.ds = ds
        self.classes = len(np.union1d(np.unique(ds['train']['Y']), np.unique(ds['test']['Y'])))
        self.seed = seed

    @property
    def input_init(self) -> NDArray:
        """Get some dummy inputs for initializing a model."""
        return np.zeros((32,) + self.ds['train'][0]['X'].shape, dtype='float32')

    @property
    def input_shape(self) -> Tuple[int]:
        """Get the shape of a single sample in the dataset"""
        return self.ds['train'][0]['X'].shape

    def get_iter(
        self,
        split: str|Iterable[str],
        batch_size: Optional[int] = None,
        idx: Optional[Iterable[int]] = None,
        filter_fn: Optional[Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]] = None,
        map_fn: Optional[Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]] = None,
    ) -> DataIter:
        """
        Generate an iterator out of the dataset.

        Arguments:
        - split: the split to use, either "train" or "test"
        - batch_size: the batch size
        - idx: the indices to use
        - filter_fn: a function that takes the labels and returns whether to keep the sample
        - map_fn: a function that takes the samples and labels and returns a subset of the samples and labels
        - in_memory: Whether of not the data should remain in the memory
        """
        rng = np.random.default_rng(self.seed)
        if filter_fn is not None:
            self.ds = self.ds.filter(filter_fn)
        if map_fn is not None:
            self.ds = self.ds.map(map_fn)
        X, Y = self.ds[split]['X'], self.ds[split]['Y']
        if idx is not None:
            X, Y = X[idx], Y[idx]
        return DataIter(X, Y, batch_size, self.classes, rng)

    def get_test_iter(
        self,
        batch_size: Optional[int] = None,
        filter_fn: Optional[Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]] = None,
        map_fn: Optional[Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]] = None,
    ):
        """
        Get a generator that deterministically gets batches of samples from the test dataset.

        Parameters:
        - batch_size: the number of samples to be included in each batch
        - filter_fn: a function that takes the labels and returns whether to keep the sample
        - map_fn: a function that takes the samples and labels and returns a subset of the samples and labels
        """
        X, Y = self.ds['test']['X'], self.ds['test']['Y']
        if filter_fn:
            idx = filter_fn(self.Y)
            X, Y = X[idx], Y[idx]
        if map_fn:
            X, Y = map_fn(X, Y)
        length = len(Y)
        if batch_size is None:
            batch_size = length
        idx_from, idx_to = 0, batch_size
        while idx_to < length:
            yield X[idx_from:idx_to], Y[idx_from:idx_to]
            idx_from = idx_to
            idx_to = min(idx_to + batch_size, length)

    def fed_split(
        self,
        batch_sizes: Iterable[int],
        mapping: Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]] = None,
    ) -> List[DataIter]:
        """
        Divide the dataset for federated learning.

        Arguments:
        - batch_sizes: the batch sizes for each client
        - mapping: a function that takes the dataset information and returns the indices for each client
        - in_memory: Whether of not the data should remain in the memory
        """
        rng = np.random.default_rng(self.seed)
        if mapping is not None:
            distribution = mapping(self.ds['train']['Y'], len(batch_sizes), self.classes, rng)
            return [
                self.get_iter("train", b, idx=d)
                for b, d in zip(batch_sizes, distribution)
            ]
        return [self.get_iter("train", b) for b in batch_sizes]
