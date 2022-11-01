"""
Load a dataset, handle the subset distribution, and provide an iterator.
"""

from typing import Callable, Iterable, Any, Tuple, Optional, List
from numpy.typing import NDArray
import datasets
import numpy as np


class HFDataIter:
    """Iterator that gives random batchs in pairs of $(X_i, y_i) : i \subseteq {1, \ldots, N}$"""

    def __init__(self, ds: datasets.Dataset, batch_size: int, classes: int, rng):
        self.ds = ds
        self.batch_size = len(ds) if batch_size is None else min(batch_size, len(ds))
        self.len = len(ds)
        self.classes = classes
        self.rng = rng

    def __iter__(self):
        """Return this as an iterator."""
        return self

    def filter(self, filter_fn: Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]):
        self.ds = self.ds.filter(filter_fn)
        self.len = len(self.ds)
        return self

    def map(self, map_fn: Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]):
        self.ds = self.ds.map(map_fn)
        self.len = len(self.Y)
        return self

    def __next__(self) -> Tuple[NDArray, NDArray]:
        """Get a random batch."""
        idx = self.rng.choice(self.len, self.batch_size, replace=False)
        return self.ds[idx]['X'], self.ds[idx]['Y']

    def __len__(self) -> int:
        return len(self.ds)


class DataIter:
    """Iterator that gives random batchs in pairs of $(X_i, Y_i) : i \subseteq {1, \ldots, N}$"""

    def __init__(self, X: NDArray, Y: NDArray, batch_size: int, classes: int, rng: np.random.Generator):
        """
        Construct a data iterator.
        
        Arguments:
        - X: the samples
        - y: the labels
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
        idx = filter_fn(self.Y)
        self.X, self.Y = self.X[idx], self.Y[idx]
        self.len = len(self.Y)
        return self

    def map(self, map_fn: Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]):
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

    def __len__(self):
        return len(self.ds)


class Dataset:
    """Object that contains the full dataset, primarily to prevent the need for reloading for each client."""

    def __init__(self, name: str, ds: datasets.Dataset):
        """
        Construct the dataset.
        Arguments:
        - ds: a hugging face dataset
        """
        self.name = name
        self.ds = ds
        self.classes = len(np.union1d(np.unique(ds['train']['Y']), np.unique(ds['test']['Y'])))

    @property
    def input_init(self) -> NDArray:
        """Get some dummy inputs for initializing a model."""
        return np.zeros((32,) + self.ds['train'][0]['X'].shape, dtype='float32')

    def get_iter(
        self,
        split: str|Iterable[str],
        batch_size: Optional[int] = None,
        idx: Optional[Iterable[int]] = None,
        filter_fn: Optional[Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]] = None,
        map_fn: Optional[Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]] = None,
        in_memory: bool = True,
        seed: Optional[int] = None
    ) -> DataIter|HFDataIter:
        """
        Generate an iterator out of the dataset.
        
        Arguments:
        - split: the split to use, either "train" or "test"
        - batch_size: the batch size
        - idx: the indices to use
        - filter: a function that takes the labels and returns whether to keep the sample
        - map: a function that takes the samples and labels and returns a subset of the samples and labels
        - rng: the random number generator
        """
        rng = np.random.default_rng(seed)
        if filter_fn is not None:
            self.ds = self.ds.filter(filter_fn)
        if map_fn is not None:
            self.ds = self.ds.map(map_fn)
        if in_memory:
            X, Y = self.ds[split]['X'], self.ds[split]['Y']
            if idx is not None:
                X, Y = X[idx], Y[idx]
            return DataIter(X, Y, batch_size, self.classes, rng)
        ds = self.ds[split]
        if idx is not None:
            ds = ds.select(idx)
        return HFDataIter(ds, batch_size, self.classes, rng)

    def get_test_iter(self, batch_size: Optional[int] = None, filter_fn = None, map_fn = None):
        """
        Get a generator that deterministically gets batches of samples from the test dataset.

        Parameters:
        - batch_size: the number of samples to be included in each batch
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
        in_memory: bool = True,
        seed: Optional[int] = None
    ) -> List[DataIter|HFDataIter]:
        """
        Divide the dataset for federated learning.
        
        Arguments:
        - batch_sizes: the batch sizes for each client
        - mapping: a function that takes the dataset information and returns the indices for each client
        - rng: the random number generator
        """
        rng = np.random.default_rng(seed)
        if mapping is not None:
            distribution = mapping(self.ds['train']['Y'], len(batch_sizes), self.classes, rng)
            return [
                self.get_iter("train", b, idx=d, in_memory=in_memory, seed=seed)
                for b, d in zip(batch_sizes, distribution)
            ]
        return [self.get_iter("train", b, in_memory=in_memory, seed=seed) for b in batch_sizes]


class TextHFDataIter(HFDataIter):
    def __init__(self, *args, vocab_size: int):
        super().__init__(*args)
        self.vocab_size = vocab_size

    def __next__(self) -> Tuple[Tuple[NDArray, NDArray], NDArray]:
        idx = self.rng.choice(self.len, self.batch_size, replace=False)
        return (self.ds[idx]['X'], self.ds[idx]['mask']), self.ds[idx]['Y']


class TextDataIter(DataIter):
    def __init__(self, *args, mask: NDArray, vocab_size: int):
        super().__init__(*args)
        self.mask = mask
        self.vocab_size = vocab_size

    def __next__(self) -> Tuple[Tuple[NDArray, NDArray], NDArray]:
        idx = self.rng.choice(self.len, self.batch_size, replace=False)
        return (self.X[idx], self.mask[idx]), self.Y[idx]


class TextDataset(Dataset):
    """A dataset containing text-based data."""
    def __init__(self, name: str, ds: datasets.Dataset, vocab_size: int, max_length: int):
        """
        Arguments:
        - name: Name of the dataset
        - ds: A tokenized huggingface dataset object
        - vocab_size: size of the tokenized vocabulary
        - max_length: max length of a sample
        """
        super().__init__(name, ds)
        self.vocab_size = vocab_size
        self.max_length = max_length

    @property
    def input_init(self) -> Tuple[NDArray, NDArray]:
        """Get some dummy inputs for initializing a model."""
        return (
            np.zeros((32,) + self.ds['train'][0]['X'].shape, dtype=int),
            np.zeros((32,) + self.ds['train'][0]['mask'].shape, dtype=bool)
        )

    def get_iter(
        self,
        split: str|Iterable[str],
        batch_size: Optional[int] = None,
        idx: Optional[Iterable[int]] = None,
        filter_fn: Optional[Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]] = None,
        map_fn: Optional[Callable[[dict[str, Iterable[Any]]], dict[str, Iterable[Any]]]] = None,
        in_memory: bool = True,
        seed: Optional[int] = None
    ) -> TextDataIter|TextHFDataIter:
        """
        Generate an iterator out of the dataset.
        
        Arguments:
        - split: the split to use, either "train" or "test"
        - batch_size: the batch size
        - idx: the indices to use
        - filter: a function that takes the labels and returns whether to keep the sample
        - map: a function that takes the samples and labels and returns a subset of the samples and labels
        - rng: the random number generator
        """
        rng = np.random.default_rng(seed)
        if filter_fn is not None:
            self.ds = self.ds.filter(filter_fn)
        if map_fn is not None:
            self.ds = self.ds.map(map_fn)
        if in_memory:
            split_data = self.ds[split][:]
            X, mask, Y = split_data['X'], split_data['mask'], split_data['Y']
            if idx is not None:
                X, mask, Y = X[idx], mask[idx], Y[idx]
            return TextDataIter(X, Y, batch_size, self.classes, rng, mask=mask, vocab_size=self.vocab_size)
        ds = self.ds[split]
        if idx is not None:
            ds = ds.select(idx)
        return TextHFDataIter(ds, batch_size, self.classes, rng, vocab_size=self.vocab_size)

    def get_test_iter(self, batch_size: Optional[int] = None, filter_fn = None, map_fn = None):
        """
        Get a generator that deterministically gets batches of samples from the test dataset.

        Parameters:
        - batch_size: the number of samples to be included in each batch
        """
        X, mask, Y = self.ds['test']['X'], self.ds['test']['mask'], self.ds['test']['Y']
        if filter_fn:
            idx = filter_fn(self.Y)
            X, mask, Y = X[idx], mask[idx], Y[idx]
        if map_fn:
            X, Y = map_fn(X, Y)
        length = len(Y)
        if batch_size is None:
            batch_size = length
        idx_from, idx_to = 0, batch_size
        while idx_to < length:
            yield (X[idx_from:idx_to], mask[idx_from:idx_to]), Y[idx_from:idx_to]
            idx_from = idx_to
            idx_to = min(idx_to + batch_size, length)
