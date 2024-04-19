import tempfile
import numpy as np
import zarr
import h5py
import lindi
import pytest
import numcodecs
from utils import assert_groups_equal, arrays_are_equal


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


def test_require_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        dirname = f'{tmpdir}/test.zarr'
        store = zarr.DirectoryStore(dirname)
        zarr.group(store=store)
        with lindi.LindiH5pyFile.from_zarr_store(store, mode='r+') as h5f_backed_by_zarr:
            h5f_backed_by_zarr.create_dataset('dset_int8', data=np.array([1, 2, 3], dtype=np.int8))
            h5f_backed_by_zarr.create_dataset('dset_int16', data=np.array([1, 2, 3], dtype=np.int16))
            h5f_backed_by_zarr.require_dataset('dset_int8', shape=(3,), dtype=np.int8)
            with pytest.raises(Exception):
                h5f_backed_by_zarr.require_dataset('dset_int8', shape=(4,), dtype=np.int8)
            with pytest.raises(Exception):
                h5f_backed_by_zarr.require_dataset('dset_int8', shape=(3,), dtype=np.int16, exact=True)
            h5f_backed_by_zarr.require_dataset('dset_int8', shape=(3,), dtype=np.int16, exact=False)
            with pytest.raises(Exception):
                h5f_backed_by_zarr.require_dataset('dset_int16', shape=(3,), dtype=np.int8, exact=False)
            ds = h5f_backed_by_zarr.require_dataset('dset_float32', shape=(3,), dtype=np.float32)
            ds[:] = np.array([1.1, 2.2, 3.3])
            with pytest.raises(Exception):
                h5f_backed_by_zarr.require_dataset('dset_float32', shape=(3,), dtype=np.float64, exact=True)


def test_zarr_write_with_zstd_compressor():
    with tempfile.TemporaryDirectory() as tmpdir:
        dirname = f'{tmpdir}/test.zarr'
        store = zarr.DirectoryStore(dirname)
        zarr.group(store=store)
        with lindi.LindiH5pyFile.from_zarr_store(store, mode='r+') as h5f_backed_by_zarr:
            h5f_backed_by_zarr.create_dataset(
                'dset_float32',
                data=np.array([1, 2, 3], dtype=np.float32),
                compression=numcodecs.Zstd(),  # this compressor not supported without plugin in hdf5
            )

        store2 = zarr.DirectoryStore(dirname)
        with lindi.LindiH5pyFile.from_zarr_store(store2) as h5f_backed_by_zarr:
            dset = h5f_backed_by_zarr['dset_float32']
            assert isinstance(dset, h5py.Dataset)
            if not arrays_are_equal(dset[()], np.array([1, 2, 3], dtype=np.float32)):
                print(dset[()])
                print(np.array([1, 2, 3], dtype=np.float32))
                raise Exception('Data mismatch')


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

    group_to_delete = h5f.create_group('group_to_delete')
    group_to_delete.attrs['attr_str'] = 'hello'
    group_to_delete.attrs['attr_int'] = 42
    group_to_delete.create_dataset('dset_to_delete', data=np.array([1, 2, 3], dtype=np.int8))
    del h5f['group_to_delete']

    another_group_to_delete = group1.create_group('another_group_to_delete')
    another_group_to_delete.attrs['attr_str'] = 'hello'
    another_group_to_delete.attrs['attr_int'] = 42
    another_group_to_delete.create_dataset('dset_to_delete', data=np.array([1, 2, 3], dtype=np.int8))
    del group1['another_group_to_delete']

    yet_another_group_to_delete = group1.create_group('yet_another_group_to_delete')
    yet_another_group_to_delete.attrs['attr_str'] = 'hello'
    yet_another_group_to_delete.attrs['attr_int'] = 42
    yet_another_group_to_delete.create_dataset('dset_to_delete', data=np.array([1, 2, 3], dtype=np.int8))
    del h5f['group1/yet_another_group_to_delete']


def compare_example_h5_data(h5f: h5py.File, tmpdir: str):
    with h5py.File(f'{tmpdir}/for_comparison.h5', 'w') as h5f2:
        write_example_h5_data(h5f2)
    with h5py.File(f'{tmpdir}/for_comparison.h5', 'r') as h5f2:
        assert_groups_equal(h5f, h5f2)


if __name__ == '__main__':
    test_require_dataset()
