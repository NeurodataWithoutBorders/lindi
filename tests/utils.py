from typing import Union
import numpy as np
import h5py
from lindi.conversion.attr_conversion import h5_to_zarr_attr


def assert_h5py_files_equal(h5f1: h5py.File, h5f2: h5py.File, *, skip_large_datasets: bool):
    assert_groups_equal(h5f1, h5f2, skip_large_datasets=skip_large_datasets)


def assert_groups_equal(h5f: h5py.Group, h5f2: h5py.Group, *, skip_large_datasets: bool):
    print(f'Comparing groups: {h5f.name}')
    assert_attrs_equal(h5f, h5f2)
    for k in h5f.keys():
        X1 = h5f[k]
        X2 = h5f2[k]
        if isinstance(X1, h5py.Group):
            assert isinstance(X2, h5py.Group)
            assert_groups_equal(X1, X2, skip_large_datasets=skip_large_datasets)
        elif isinstance(X1, h5py.Dataset):
            assert isinstance(X2, h5py.Dataset)
            assert_datasets_equal(X1, X2, skip_large_datasets=skip_large_datasets)
        else:
            raise Exception(f'Unexpected type: {type(X1)}')

    for k in h5f2.keys():
        if k not in h5f:
            raise Exception(f'Key {k} not found in h5f')


def assert_datasets_equal(h5d1: h5py.Dataset, h5d2: h5py.Dataset, *, skip_large_datasets: bool):
    print(f'Comparing datasets: {h5d1.name}')
    assert h5d1.shape == h5d2.shape, f'h5d1.shape: {h5d1.shape}, h5d2.shape: {h5d2.shape}'
    assert h5d1.dtype == h5d2.dtype, f'h5d1.dtype: {h5d1.dtype}, h5d2.dtype: {h5d2.dtype}'
    if skip_large_datasets and np.prod(h5d1.shape) > 1000:
        print(f'Skipping large dataset: {h5d1.name}')
        return
    if h5d1.dtype.kind == 'V':
        for name in h5d1.dtype.names:
            data1 = h5d1[name][()]
            data2 = h5d2[name][()]
            if not arrays_are_equal(data1, data2):
                raise Exception(f'Arrays are not equal for field {name}')
    elif h5d1.dtype.kind == 'O':
        # skip object arrays
        pass
    else:
        data1 = h5d1[()]
        data2 = h5d2[()]
        if not arrays_are_equal(data1, data2):
            raise Exception(f'Arrays are not equal for dataset {h5d1.name} with dtype {h5d1.dtype}')


def arrays_are_equal(a, b):
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    # if this is numeric data we need to use allclose so that we can handle NaNs
    if np.issubdtype(a.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)
    else:
        return np.array_equal(a, b)


def assert_attrs_equal(
    h5f1: Union[h5py.Group, h5py.Dataset],
    h5f2: Union[h5py.Group, h5py.Dataset]
):
    attrs1 = h5f1.attrs
    attrs2 = h5f2.attrs
    keys1 = set(attrs1.keys())
    keys2 = set(attrs2.keys())
    assert keys1 == keys2, f'keys1: {keys1}, keys2: {keys2}'
    for k1, v1 in attrs1.items():
        assert_attr_equal(v1, attrs2[k1])


def assert_attr_equal(v1, v2):
    v1_normalized = h5_to_zarr_attr(v1, h5f=None)
    v2_normalized = h5_to_zarr_attr(v2, h5f=None)
    assert v1_normalized == v2_normalized, f'v1_normalized: {v1_normalized}, v2_normalized: {v2_normalized}'


def lists_are_equal(a, b):
    if len(a) != len(b):
        return False
    for aa, bb in zip(a, b):
        if aa != bb:
            if np.isnan(aa) and np.isnan(bb):
                # nan != nan, but we want to consider them equal
                continue
            return False
    return True
