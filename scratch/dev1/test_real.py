import numpy as np
import zarr
import h5py
import remfile
from LindiH5Store import LindiH5Store


# This one seems to load properly
# https://neurosift.app/?p=/nwb&dandisetId=000717&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/
h5_url = 'https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/'
json_url = 'https://kerchunk.neurosift.org/dandi/dandisets/000717/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/zarr.json'


def main():
    remf = remfile.File(h5_url)
    h5f = h5py.File(remf, 'r')
    f = LindiH5Store(remf)
    root = zarr.open(f)
    assert isinstance(root, zarr.Group)

    # visit the items in the h5py file and compare them to the zarr file
    _compare_groups(root, h5f)
    h5f.visititems(lambda key, item: _compare_item(key, item, root[key]))


def _compare_item(key: str, item_h5, item_zarr):
    if isinstance(item_h5, h5py.Group):
        assert isinstance(item_zarr, zarr.Group)
        _compare_groups(item_zarr, item_h5)
    elif isinstance(item_h5, h5py.Dataset):
        assert isinstance(item_zarr, zarr.Array)
        _compare_arrays(item_zarr, item_h5)
    else:
        print(item_h5)
        raise NotImplementedError()


def _compare_groups(g1: zarr.Group, g2: h5py.Group):
    print(f'__________ {g1.name} (GROUP)')
    assert g1.name == g2.name
    for k, v in g1.attrs.items():
        if k not in g2.attrs:
            print(f'WARNING: Attribute {k} not found in h5 group {g2.name}')
        elif g2.attrs[k] != v:
            print(f'WARNING: Attribute {k} value mismatch in h5 group {g2.name}')
            print(f'  h5: {g2.attrs[k]}')
            print(f'  zarr: {v}')
    for k, v in g2.attrs.items():
        if k not in g1.attrs:
            print(f'WARNING: Attribute {k} not found in zarr group {g1.name}')
        elif g1.attrs[k] != v:
            print(f'WARNING: Attribute {k} value mismatch in zarr group {g1.name}')


def _compare_arrays(a1: zarr.Array, a2: h5py.Dataset):
    print(f'__________ {a1.name} (ARRAY)')
    if a1.dtype != a2.dtype:
        print(f'WARNING: dtype mismatch for {a1.name}: {a1.dtype} != {a2.dtype}')
    a1_shape = a1.shape
    if a1.attrs['_ARRAY_DIMENSIONS'] == []:
        a1_shape = ()
    if a1_shape != a2.shape:
        print(f'WARNING: shape mismatch for {a1.name}: {a1.shape} != {a2.shape}')
    if len(a1_shape) == 0:
        a1_val = a1[0]
        a2_val = a2[()]
        if isinstance(a1_val, bytes):
            a1_val = a1_val.decode()
        if isinstance(a2_val, bytes):
            a2_val = a2_val.decode()
        if a1_val != a2_val:
            print(f'WARNING: value mismatch for {a1.name}: {a1_val} != {a2_val}')
    else:
        if np.prod(a1_shape) < 1000:
            a1_data = a1[:]
            a2_data = a2[:]
            if not a1_data.shape == a2_data.shape:
                print(f'WARNING: shape mismatch for {a1.name}: {a1_data.shape} != {a2_data.shape}')
            if not a1_data.dtype == a2_data.dtype:
                print(f'WARNING: dtype mismatch for {a1.name}: {a1_data.dtype} != {a2_data.dtype}')
            if not np.array_equal(a1_data, a2_data):
                print(f'WARNING: data mismatch for {a1.name}')
                print(a1_data)
                print(a2_data)
        else:
            if a1.ndim == 1:
                a1_data = a1[:20]
                a2_data = a2[:20]
            elif a1.ndim == 2:
                a1_data = a1[:20, :2]
                a2_data = a2[:20, :2]
            elif a1.ndim == 3:
                a1_data = a1[:2, :2, :2]
                a2_data = a2[:2, :2, :2]
            else:
                raise NotImplementedError()
            if not a1_data.dtype == a2_data.dtype:
                print(f'WARNING: dtype mismatch for {a1.name}: {a1_data.dtype} != {a2_data.dtype}')
            if not np.array_equal(a1_data, a2_data):
                print(f'WARNING: data mismatch for {a1.name}')
                print(a1_data)
                print(a2_data)


if __name__ == '__main__':
    main()
