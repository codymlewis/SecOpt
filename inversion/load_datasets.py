import copy
import datasets
import numpy as np
import einops
from skimage import transform


class Dataset:
    def __init__(self, data, input_shape, nclasses):
        self.data = data
        self.orig_data = copy.deepcopy(data)
        self.input_shape = input_shape
        self.nclasses = nclasses
        self.nsamples = len(data['train']['Y'])

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, v):
        self.data[i] = v

    def perturb(self, rng):
        "Make random changes to the training dataset."
        train_data = self.orig_data['train']['X'].copy()

        flip_idx = rng.choice(self.nsamples, self.nsamples // 4, replace=False)
        train_data[flip_idx] = np.flip(train_data[flip_idx], 0)
        flip_idx = rng.choice(self.nsamples, self.nsamples // 4, replace=False)
        train_data[flip_idx] = np.flip(train_data[flip_idx], 1)

        rot_idx = rng.choice(self.nsamples, self.nsamples // 8, replace=False)
        train_data[rot_idx] = np.array([transform.rotate(x, rng.normal(0.0, 15)) for x in train_data[rot_idx]])

        contrast_idx = rng.choice(self.nsamples, self.nsamples // 16, replace=False)
        train_data[contrast_idx] = np.clip(
            (train_data[contrast_idx].T - train_data[contrast_idx].mean(axis=(1, 2, 3))) *
            rng.normal(0.0, 0.01, size=len(contrast_idx)) + train_data[contrast_idx].mean(axis=(1, 2, 3)),
            0.0,
            1.0
        ).T

        brightness_idx = rng.choice(self.nsamples, self.nsamples // 16, replace=False)
        train_data[brightness_idx] = np.clip(
            train_data[brightness_idx].T + rng.normal(0.0, 0.01, size=len(brightness_idx)),
            0.0,
            1.0
        ).T

        noise_idx = rng.choice(self.nsamples, self.nsamples // 2, replace=False)
        train_data[noise_idx] = np.clip(
            train_data[noise_idx] + rng.laplace(0, 0.1, size=train_data[noise_idx].shape), 0.0, 1.0)

        self.data['train']['X'] = train_data

        # import matplotlib.pyplot as plt
        # import math
        # idx = rng.choice(len(train_data), 30, replace=False)
        # batch_size = len(idx)
        # nrows = math.floor(math.sqrt(batch_size))
        # ncols = batch_size // nrows
        # fig, axes = plt.subplots(nrows, ncols)
        # for i, ax in enumerate(axes.flatten()):
        #     if i < batch_size:
        #         ax.imshow(self.data['train']['X'][idx[i]], cmap='binary')
        # plt.savefig("perturb.png", dpi=320)
        # plt.clf()


def hfdataset_to_dict(hfdataset):
    return {t: {k: hfdataset[t][k] for k in hfdataset[t].column_names} for t in hfdataset.keys()}


def fmnist():
    ds = datasets.load_dataset("fashion_mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    input_shape = (28, 28, 1)
    features['X'] = datasets.Array3D(shape=input_shape, dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data_dict = hfdataset_to_dict(ds)
    nclasses = len(set(np.unique(ds['train']['Y'])) & set(np.unique(ds['test']['Y'])))
    return Dataset(data_dict, input_shape, nclasses)


def cifar10():
    ds = datasets.load_dataset("cifar10")
    ds = ds.map(
        lambda e: {
            'X': np.array(e['img'], dtype=np.float32) / 255,
            'Y': e['label']
        },
        remove_columns=['img', 'label']
    )
    features = ds['train'].features
    input_shape = (32, 32, 3)
    features['X'] = datasets.Array3D(shape=input_shape, dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data_dict = hfdataset_to_dict(ds)
    nclasses = len(set(np.unique(ds['train']['Y'])) & set(np.unique(ds['test']['Y'])))
    return Dataset(data_dict, input_shape, nclasses)


def cifar100():
    ds = datasets.load_dataset("cifar100")
    ds = ds.map(
        lambda e: {
            'X': np.array(e['img'], dtype=np.float32) / 255,
            'Y': e['fine_label']
        },
        remove_columns=['img', 'fine_label', 'coarse_label']
    )
    features = ds['train'].features
    input_shape = (32, 32, 3)
    features['X'] = datasets.Array3D(shape=input_shape, dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data_dict = hfdataset_to_dict(ds)
    nclasses = len(set(np.unique(ds['train']['Y'])) & set(np.unique(ds['test']['Y'])))
    return Dataset(data_dict, input_shape, nclasses)


def svhn():
    ds = datasets.load_dataset("svhn", "cropped_digits")
    ds.pop("extra")
    ds = ds.map(
        lambda e: {
            'X': np.array(e['image'], dtype=np.float32) / 255,
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    input_shape = (32, 32, 3)
    features['X'] = datasets.Array3D(shape=input_shape, dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data_dict = hfdataset_to_dict(ds)
    nclasses = len(set(np.unique(ds['train']['Y'])) & set(np.unique(ds['test']['Y'])))
    return Dataset(data_dict, input_shape, nclasses)


def tinyimagenet():
    ds = datasets.load_dataset("zh-plus/tiny-imagenet")
    ds = ds.map(
        lambda e: {
            'X': einops.repeat(img, "h w -> h w 3") if len((img := np.array(e['image'], dtype=np.float32) / 255).shape) == 2
            else img,
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    input_shape = (64, 64, 3)
    features['X'] = datasets.Array3D(shape=input_shape, dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['valid'] = ds['valid'].cast(features)
    ds.set_format('numpy')
    data_dict = hfdataset_to_dict(ds)
    data_dict['test'] = data_dict['valid']
    del data_dict['valid']
    nclasses = len(set(np.unique(data_dict['train']['Y'])) & set(np.unique(data_dict['test']['Y'])))
    return Dataset(data_dict, input_shape, nclasses)
