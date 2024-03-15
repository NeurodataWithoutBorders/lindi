import json
import tempfile
import numpy as np
import h5py
import zarr
import kerchunk.hdf  # type: ignore
from lindi import LindiH5Store
from fsspec.implementations.reference import ReferenceFileSystem


def test_scalar_dataset():
    for val in ["abc", b"abc", 1, 3.6]:
        print(f"Testing scalar {val} of type {type(val)}")
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/test.h5"
            with h5py.File(filename, "w") as f:
                f.create_dataset("X", data=val)
            zarr_kerchunk, store_kerchunk = _get_kerchunk_zarr(filename)
            val_kerchunk = zarr_kerchunk["X"][0]
            zarr_lindi, store_lindi = _get_lindi_zarr(filename)
            try:
                val_lindi = zarr_lindi["X"][0]
                if val_kerchunk != val:
                    print(f"WARNING: val_kerchunk={val_kerchunk} != val={val}")
                if val_lindi != val:
                    print(f"WARNING: val_lindi={val_lindi} != val={val}")
                if type(val_kerchunk) is not type(val):
                    print(
                        "WARNING: type mismatch for kerchunk:",
                        type(val),
                        type(val_kerchunk),
                    )
                if type(val_lindi) is not type(val):
                    print("WARNING: type mismatch for lindi:", type(val), type(val_lindi))
                print("")
                x = store_lindi.to_reference_file_system()  # noqa: F841
            finally:
                store_lindi.close()


def test_numpy_array():
    print("Testing numpy array")
    X1 = (np.arange(60).reshape(3, 20), (3, 7))
    X2 = (np.arange(60).reshape(3, 20), None)
    for array, chunks in [X1, X2]:
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/test.h5"
            with h5py.File(filename, "w") as f:
                f.create_dataset("X", data=array, chunks=chunks)
            zarr_kerchunk, store_kerchunk = _get_kerchunk_zarr(filename)
            array_kerchunk = zarr_kerchunk["X"][:]
            assert isinstance(array_kerchunk, np.ndarray)
            zarr_lindi, store_lindi = _get_lindi_zarr(filename)
            array_lindi = zarr_lindi["X"][:]
            assert isinstance(array_lindi, np.ndarray)
            if not np.array_equal(array_kerchunk, array):
                print("WARNING: array_kerchunk does not match array")
                print(array_kerchunk)
                print(array)
            if not np.array_equal(array_lindi, array):
                print("WARNING: array_lindi does not match array")
                print(array_lindi)
                print(array)
            x = store_lindi.to_reference_file_system()  # noqa: F841


def test_numpy_array_of_strings():
    print("Testing numpy array of strings")
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=["abc", "def", "ghi"])
        zarr_kerchunk, store_kerchunk = _get_kerchunk_zarr(filename)
        array_kerchunk = zarr_kerchunk["X"][:]
        assert isinstance(array_kerchunk, np.ndarray)
        zarr_lindi, store_lindi = _get_lindi_zarr(filename)
        array_lindi = zarr_lindi["X"][:]
        assert isinstance(array_lindi, np.ndarray)
        if not np.array_equal(array_kerchunk, ["abc", "def", "ghi"]):
            print("WARNING: array_kerchunk does not match array")
            print(array_kerchunk)
            print(["abc", "def", "ghi"])
        if not np.array_equal(array_lindi, ["abc", "def", "ghi"]):
            print("WARNING: array_lindi does not match array")
            print(array_lindi)
            print(["abc", "def", "ghi"])
        x = store_lindi.to_reference_file_system()  # noqa: F841


def _get_lindi_zarr(filename):
    store = LindiH5Store.from_file(filename, url='.')  # use url='.' so that a reference file system can be created
    root = zarr.open(store)
    return root, store


def _get_kerchunk_zarr(filename):
    with h5py.File(filename, "r") as f:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(
            f,
            url=filename,
            hdmf_mode=True,
            num_chunks_per_dataset_threshold=1000,
            max_num_items=1000,
        )
        a = h5chunks.translate()
        with open("test_example.zarr.json", "w") as store:
            json.dump(a, store, indent=2)
        fs = ReferenceFileSystem(a)
        store0 = fs.get_mapper(root="/", check=False)
        root = zarr.open(store0)
        return root, store0


if __name__ == "__main__":
    test_scalar_dataset()
    test_numpy_array()
    test_numpy_array_of_strings()
