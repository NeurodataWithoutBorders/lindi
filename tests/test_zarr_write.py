from typing import Union
import tempfile
import numpy as np
import zarr
import h5py
import lindi
from lindi.conversion.attr_conversion import h5_to_zarr_attr
import pytest


def test_zarr_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        dirname = f'{tmpdir}/test.zarr'
        store = zarr.DirectoryStore(dirname)
        zarr.group(store=store)
        with lindi.LindiH5pyFile.from_zarr_store(store, mode='r+') as h5f_backed_by_zarr:
            write_example_h5_data(h5f_backed_by_zarr)

        store2 = zarr.DirectoryStore(dirname)
        with lindi.LindiH5pyFile.from_zarr_store(store2) as h5f_backed_by_zarr:
            compare_example_h5_data(h5f_backed_by_zarr, tmpdir=tmpdir)


def write_example_h5_data(h5f: h5py.File):
    h5f.attrs['attr_str'] = 'hello'
    h5f.attrs['attr_int'] = 42
    h5f.attrs['attr_float'] = 3.14
    h5f.attrs['attr_bool'] = True
    h5f.attrs['attr_list_str'] = ['a', 'b', 'c']
    h5f.attrs['attr_list_int'] = [1, 2, 3]
    h5f.attrs['attr_list_float'] = [1.1, 2.2, 3.3]
    h5f.attrs['attr_list_bool'] = [True, False, True]
    with pytest.raises(Exception):
        h5f.attrs['attr_list_mixed'] = [1, 2.2, 'c', True]
    h5f.attrs['2d_array'] = np.array([[1, 2], [3, 4]])
    h5f.create_dataset('dset_int8', data=np.array([1, 2, 3], dtype=np.int8))
    h5f.create_dataset('dset_int16', data=np.array([1, 2, 3], dtype=np.int16))
    h5f.create_dataset('dset_int32', data=np.array([1, 2, 3], dtype=np.int32))
    h5f.create_dataset('dset_int64', data=np.array([1, 2, 3], dtype=np.int64))
    h5f.create_dataset('dset_uint8', data=np.array([1, 2, 3], dtype=np.uint8))
    h5f.create_dataset('dset_uint16', data=np.array([1, 2, 3], dtype=np.uint16))
    h5f.create_dataset('dset_uint32', data=np.array([1, 2, 3], dtype=np.uint32))
    h5f.create_dataset('dset_uint64', data=np.array([1, 2, 3], dtype=np.uint64))
    h5f.create_dataset('dset_float32', data=np.array([1, 2, 3], dtype=np.float32))
    h5f.create_dataset('dset_float64', data=np.array([1, 2, 3], dtype=np.float64))
    h5f.create_dataset('dset_bool', data=np.array([True, False, True], dtype=np.bool_))

    group1 = h5f.create_group('group1')
    group1.attrs['attr_str'] = 'hello'
    group1.attrs['attr_int'] = 42
    group1.create_dataset('dset_with_nan', data=np.array([1, np.nan, 3], dtype=np.float64))
    group1.create_dataset('dset_with_inf', data=np.array([np.inf, 6, -np.inf], dtype=np.float64))

    compound_dtype = np.dtype([('x', np.int32), ('y', np.float64)])
    group1.create_dataset('dset_compound', data=np.array([(1, 2.2), (3, 4.4)], dtype=compound_dtype))


def compare_example_h5_data(h5f: h5py.File, tmpdir: str):
    with h5py.File(f'{tmpdir}/for_comparison.h5', 'w') as h5f2:
        write_example_h5_data(h5f2)
    with h5py.File(f'{tmpdir}/for_comparison.h5', 'r') as h5f2:
        _assert_groups_equal(h5f, h5f2)


def _assert_groups_equal(h5f: h5py.Group, h5f2: h5py.Group):
    print(f'Comparing groups: {h5f.name}')
    _assert_attrs_equal(h5f, h5f2)
    for k in h5f.keys():
        X1 = h5f[k]
        X2 = h5f2[k]
        if isinstance(X1, h5py.Group):
            assert isinstance(X2, h5py.Group)
            _assert_groups_equal(X1, X2)
        elif isinstance(X1, h5py.Dataset):
            assert isinstance(X2, h5py.Dataset)
            _assert_datasets_equal(X1, X2)
        else:
            raise Exception(f'Unexpected type: {type(X1)}')

    for k in h5f2.keys():
        if k not in h5f:
            raise Exception(f'Key {k} not found in h5f')


def _assert_datasets_equal(h5d1: h5py.Dataset, h5d2: h5py.Dataset):
    print(f'Comparing datasets: {h5d1.name}')
    assert h5d1.shape == h5d2.shape, f'h5d1.shape: {h5d1.shape}, h5d2.shape: {h5d2.shape}'
    assert h5d1.dtype == h5d2.dtype, f'h5d1.dtype: {h5d1.dtype}, h5d2.dtype: {h5d2.dtype}'
    assert h5d1.dtype == h5d2.dtype, f'h5d1.dtype: {h5d1.dtype}, h5d2.dtype: {h5d2.dtype}'
    if h5d1.dtype.kind == 'V':
        for name in h5d1.dtype.names:
            data1 = h5d1[name][()]
            data2 = h5d2[name][()]
            assert _arrays_are_equal(data1, data2), f'data1: {data1}, data2: {data2}'
    else:
        data1 = h5d1[()]
        data2 = h5d2[()]
        assert _arrays_are_equal(data1, data2), f'data1: {data1}, data2: {data2}'


def _arrays_are_equal(a, b):
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    # if this is numeric data we need to use allclose so that we can handle NaNs
    if np.issubdtype(a.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)
    else:
        return np.array_equal(a, b)


def _assert_attrs_equal(
    h5f1: Union[h5py.Group, h5py.Dataset],
    h5f2: Union[h5py.Group, h5py.Dataset]
):
    attrs1 = h5f1.attrs
    attrs2 = h5f2.attrs
    keys1 = set(attrs1.keys())
    keys2 = set(attrs2.keys())
    assert keys1 == keys2, f'keys1: {keys1}, keys2: {keys2}'
    for k1, v1 in attrs1.items():
        _assert_attr_equal(v1, attrs2[k1])


def _assert_attr_equal(v1, v2):
    v1_normalized = h5_to_zarr_attr(v1, h5f=None)
    v2_normalized = h5_to_zarr_attr(v2, h5f=None)
    assert v1_normalized == v2_normalized, f'v1_normalized: {v1_normalized}, v2_normalized: {v2_normalized}'


if __name__ == '__main__':
    test_zarr_write()
