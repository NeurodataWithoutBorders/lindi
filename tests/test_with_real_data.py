import tempfile
import numpy as np
import zarr
import h5py
import remfile
from lindi import LindiH5Store, LindiClient
import lindi
import pytest

examples = []

# example 0
# This one seems to load properly
# https://neurosift.app/?p=/nwb&dandisetId=000717&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/
examples.append(
    {
        "h5_url": "https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/",
    }
)

# example 1
# https://neurosift.app/?p=/nwb&dandisetId=000776&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/54895119-f739-4544-973e-a9341a5c66ad/download/
# Exception: Not yet implemented (3): dataset /processing/CalciumActivity/CalciumSeriesSegmentation/Aligned_neuron_coordinates/voxel_mask with
# dtype [('x', '<u4'), ('y', '<u4'), ('z', '<u4'), ('weight', '<f4')] and shape (109,)
examples.append(
    {
        "h5_url": "https://api.dandiarchive.org/api/assets/54895119-f739-4544-973e-a9341a5c66ad/download/"
    }
)

# example 2
# https://neurosift.app/?p=/nwb&dandisetId=000688&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/9ac13aea-5d1c-4924-87b4-40ef0a3555d8/download/
examples.append(
    {
        "h5_url": "https://api.dandiarchive.org/api/assets/9ac13aea-5d1c-4924-87b4-40ef0a3555d8/download/"
    }
)


def _compare_item(item_h5, item_lindi):
    if isinstance(item_h5, h5py.Group):
        assert isinstance(item_lindi, lindi.LindiGroup)
        _compare_groups(item_lindi, item_h5)
    elif isinstance(item_h5, h5py.Dataset):
        assert isinstance(item_lindi, lindi.LindiDataset)
        _compare_datasets(item_lindi, item_h5)
    else:
        print(item_h5)
        raise NotImplementedError()


def _compare_item_2(item_h5, item_zarr):
    if isinstance(item_h5, h5py.Group):
        assert isinstance(item_zarr, zarr.Group)
        _compare_groups_2(item_zarr, item_h5)
    elif isinstance(item_h5, h5py.Dataset):
        assert isinstance(item_zarr, zarr.Array)
        _compare_datasets_2(item_zarr, item_h5)
    else:
        print(item_h5)
        raise NotImplementedError()


def _compare_groups(g1: lindi.LindiGroup, g2: h5py.Group):
    print(f"__________ {g1.name} (GROUP)")
    assert g1.name == g2.name
    for k, v in g1.attrs.items():
        if k not in g2.attrs:
            print(f"WARNING: Attribute {k} not found in h5 group {g2.name}")
        elif not _values_match(v, g2.attrs[k]):
            print(f"WARNING: Attribute {k} value mismatch in h5 group {g2.name}")
            print(f"  h5: {g2.attrs[k]}")
            print(f"  zarr: {v}")
    for k, v in g2.attrs.items():
        if k not in g1.attrs:
            print(f"WARNING: Attribute {k} not found in zarr group {g1.name}")
        elif not _values_match(v, g1.attrs[k]):
            print(f"WARNING: Attribute {k} value mismatch in zarr group {g1.name}")


def _compare_groups_2(g1: zarr.Group, g2: h5py.Group):
    print(f"__________ {g1.name} (GROUP)")
    assert g1.name == g2.name
    for k, v in g1.attrs.items():
        if k not in g2.attrs:
            print(f"WARNING: Attribute {k} not found in h5 group {g2.name}")
        elif not _values_match(v, g2.attrs[k]):
            print(f"WARNING: Attribute {k} value mismatch in h5 group {g2.name}")
            print(f"  h5: {g2.attrs[k]}")
            print(f"  zarr: {v}")
    for k, v in g2.attrs.items():
        if k not in g1.attrs:
            print(f"WARNING: Attribute {k} not found in zarr group {g1.name}")
        elif not _values_match(v, g1.attrs[k]):
            print(f"WARNING: Attribute {k} value mismatch in zarr group {g1.name}")


def _values_match(v1, v2):
    if type(v1) == list and type(v2) == np.ndarray:  # noqa: E721
        return _arrays_equal(np.array(v1, dtype=v2.dtype), v2)
    if type(v1) == np.ndarray and type(v2) == list:  # noqa: E721
        return _arrays_equal(v1, np.array(v2, dtype=v1.dtype))
    if type(v1) != type(v2):  # noqa: E721
        return False
    if isinstance(v1, list):
        if len(v1) != len(v2):
            return False
        for i in range(len(v1)):
            if not _values_match(v1[i], v2[i]):
                return False
        return True
    elif isinstance(v1, dict):
        if len(v1) != len(v2):
            return False
        for k in v1:
            if k not in v2:
                return False
            if not _values_match(v1[k], v2[k]):
                return False
        return True
    else:
        return v1 == v2


def _compare_datasets(a1: lindi.LindiDataset, a2: h5py.Dataset):
    print(f"__________ {a1.name} (ARRAY)")
    if a1.dtype != a2.dtype:
        print(f"WARNING: dtype mismatch for {a1.name}: {a1.dtype} != {a2.dtype}")
    a1_shape = a1.shape
    if a1.attrs.get("_SCALAR") is True:
        a1_shape = ()
    if a1_shape != a2.shape:
        print(f"WARNING: shape mismatch for {a1.name}: {a1.shape} != {a2.shape}")
    if len(a1_shape) == 0:
        a1_val = a1[()]
        a2_val = a2[()]
        if isinstance(a1_val, bytes):
            a1_val = a1_val.decode()
        if isinstance(a2_val, bytes):
            a2_val = a2_val.decode()
        if a1_val != a2_val:
            print(f"WARNING: value mismatch for {a1.name}: {a1_val} != {a2_val}")
    else:
        if np.prod(a1_shape) < 1000:
            a1_data = a1[:]
            a2_data = a2[:]
            if not a1_data.shape == a2_data.shape:
                print(
                    f"WARNING: shape mismatch for {a1.name}: {a1_data.shape} != {a2_data.shape}"
                )
            if not a1_data.dtype == a2_data.dtype:
                print(
                    f"WARNING: dtype mismatch for {a1.name}: {a1_data.dtype} != {a2_data.dtype}"
                )
            if not _arrays_equal(a1_data, a2_data):
                print(a1_data)
                print(a2_data)
        else:
            if a2.chunks and (np.prod(a2.chunks) < 10000000):
                if a1.ndim == 1:
                    a1_data = a1[: a2.chunks[0]]
                    a2_data = a2[: a2.chunks[0]]
                elif a1.ndim == 2:
                    a1_data = a1[: a2.chunks[0], : a2.chunks[1]]
                    a2_data = a2[: a2.chunks[0], : a2.chunks[1]]
                elif a1.ndim == 3:
                    a1_data = a1[: a2.chunks[0], : a2.chunks[1], : a2.chunks[2]]
                    a2_data = a2[: a2.chunks[0], : a2.chunks[1], : a2.chunks[2]]
                elif a1.ndim == 4:
                    a1_data = a1[
                        : a2.chunks[0], : a2.chunks[1], : a2.chunks[2], : a2.chunks[3]
                    ]
                    a2_data = a2[
                        : a2.chunks[0], : a2.chunks[1], : a2.chunks[2], : a2.chunks[3]
                    ]
                elif a1.ndim == 5:
                    a1_data = a1[
                        : a2.chunks[0],
                        : a2.chunks[1],
                        : a2.chunks[2],
                        : a2.chunks[3],
                        : a2.chunks[4],
                    ]
                    a2_data = a2[
                        : a2.chunks[0],
                        : a2.chunks[1],
                        : a2.chunks[2],
                        : a2.chunks[3],
                        : a2.chunks[4],
                    ]
                else:
                    raise NotImplementedError()
                if not a1_data.dtype == a2_data.dtype:
                    print(
                        f"WARNING: dtype mismatch for {a1.name}: {a1_data.dtype} != {a2_data.dtype}"
                    )
                if not _arrays_equal(a1_data, a2_data):
                    print(f"WARNING: data mismatch for {a1.name}")
                    print(a1_data)
                    print(a2_data)
            else:
                print(
                    f"Skipping large array value comparison. Chunk shape is {a2.chunks}"
                )


def _compare_datasets_2(a1: zarr.Array, a2: h5py.Dataset):
    print(f"__________ {a1.name} (ARRAY)")
    if a1.dtype != a2.dtype:
        print(f"WARNING: dtype mismatch for {a1.name}: {a1.dtype} != {a2.dtype}")
    a1_shape = a1.shape
    if a1.attrs.get("_SCALAR") is True:
        a1_shape = ()
    if a1_shape != a2.shape:
        print(f"WARNING: shape mismatch for {a1.name}: {a1.shape} != {a2.shape}")
    if len(a1_shape) == 0:
        a1_val = a1[()]
        a2_val = a2[()]
        if isinstance(a1_val, bytes):
            a1_val = a1_val.decode()
        if isinstance(a2_val, bytes):
            a2_val = a2_val.decode()
        if a1_val != a2_val:
            print(f"WARNING: value mismatch for {a1.name}: {a1_val} != {a2_val}")
    else:
        if np.prod(a1_shape) < 1000:
            a1_data = a1[:]
            a2_data = a2[:]
            if not a1_data.shape == a2_data.shape:
                print(
                    f"WARNING: shape mismatch for {a1.name}: {a1_data.shape} != {a2_data.shape}"
                )
            if not a1_data.dtype == a2_data.dtype:
                print(
                    f"WARNING: dtype mismatch for {a1.name}: {a1_data.dtype} != {a2_data.dtype}"
                )
            if not _arrays_equal(a1_data, a2_data):
                print(a1_data)
                print(a2_data)
        else:
            pass


def _arrays_equal(a: np.ndarray, b: np.ndarray):
    # If it's an array of strings, we convert to an array of bytes
    if a.dtype == object:
        # need to modify all the entries
        a = np.array([x.encode() if type(x) is str else x for x in a.ravel()]).reshape(
            a.shape
        )
    if b.dtype == object:
        b = np.array([x.encode() if type(x) is str else x for x in b.ravel()]).reshape(
            b.shape
        )
    # if this is numeric data we need to use allclose so that we can handle NaNs
    if np.issubdtype(a.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)
    else:
        return np.array_equal(a, b)


def _hdf5_visit_items(item, callback):
    # For some reason, this is much faster than using h5f.visititems(translator)
    if isinstance(item, h5py.File):
        for k, v in item.items():
            _hdf5_visit_items(v, callback)
    elif isinstance(item, h5py.Group):
        callback(item.name, item)
        for k, v in item.items():
            _hdf5_visit_items(v, callback)
    else:
        callback(item.name, item)
        return


@pytest.mark.network
@pytest.mark.slow
def test_with_real_data():
    example_num = 0
    example = examples[example_num]
    h5_url = example["h5_url"]
    print(f"Running comparison for {h5_url}")
    h5f = h5py.File(remfile.File(h5_url), "r")
    with LindiH5Store.from_file(h5_url) as store:
        rfs = store.to_reference_file_system()
        client = LindiClient.from_reference_file_system(rfs)

        # visit the items in the h5py file and compare them to the zarr file
        _hdf5_visit_items(h5f, lambda key, item: _compare_item(item, client[key]))

        with tempfile.TemporaryDirectory() as tmpdir:
            store.to_file(f"{tmpdir}/example.zarr.json")
            LindiClient.from_file(f"{tmpdir}/example.zarr.json")

        top_level_keys = [k for k in h5f.keys()]
        top_level_keys_2 = store.listdir()
        assert len(top_level_keys) == len(top_level_keys_2)
        for k in top_level_keys:
            assert k in top_level_keys_2

        root = zarr.open(store)
        _hdf5_visit_items(h5f, lambda key, item: _compare_item_2(item, root[key]))
