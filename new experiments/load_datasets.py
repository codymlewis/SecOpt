import datasets
import numpy as np
import einops


class Dataset:
    def __init__(self, data, input_shape, nclasses):
        self.data = data
        self.input_shape = input_shape
        self.nclasses = nclasses

    def __getitem__(self, i):
        return self.data[i]
    
    def __setitem__(self, i, v):
        self.data[i] = v


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
    