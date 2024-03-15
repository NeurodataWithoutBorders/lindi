import numpy as np
import h5py
import tempfile
from lindi import LindiH5Store, LindiClient, LindiDataset, LindiGroup
import pytest


def test_scalar_datasets():
    for val in ["abc", b"abc", 1, 3.6]:
        print(f"Testing scalar {val} of type {type(val)}")
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/test.h5"
            with h5py.File(filename, "w") as f:
                f.create_dataset("X", data=val)
            with LindiH5Store.from_file(
                filename, url=filename
            ) as store:  # set url so that a reference file system can be created
                rfs = store.to_reference_file_system()
                client = LindiClient.from_reference_file_system(rfs)
                h5f = h5py.File(filename, "r")
                X1 = h5f["X"]
                assert isinstance(X1, h5py.Dataset)
                X2 = client["X"]
                assert isinstance(X2, LindiDataset)
                if not _check_equal(X1[()], X2[()]):
                    print(f"WARNING: {X1} ({type(X1)}) != {X2} ({type(X2)})")
                    raise ValueError("Scalar datasets are not equal")


def test_numpy_arrays():
    X1 = ("1", np.arange(60).reshape(3, 20), (3, 7))
    X2 = ("2", np.arange(60).reshape(3, 20), None)
    for label, array, chunks in [X1, X2]:
        print(f"Testing numpy array {label}")
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/test.h5"
            with h5py.File(filename, "w") as f:
                f.create_dataset("X", data=array, chunks=chunks)
            with LindiH5Store.from_file(
                filename, url=filename
            ) as store:  # set url so that a reference file system can be created
                rfs = store.to_reference_file_system()
                client = LindiClient.from_reference_file_system(rfs)
                h5f = h5py.File(filename, "r")
                X1 = h5f["X"]
                assert isinstance(X1, h5py.Dataset)
                X2 = client["X"]
                assert isinstance(X2, LindiDataset)

                assert X1.shape == X2.shape
                assert X1.dtype == X2.dtype
                assert X1.size == X2.size
                assert X1.nbytes == X2.nbytes
                assert len(X1) == len(X2)

                # iterate over the first axis
                count = 0
                for aa in X2:
                    assert _check_equal(aa[:], X1[count][:])
                    count += 1

                if not _check_equal(X1[:], X2[:]):
                    print("WARNING. Arrays are not equal")
                    print(X1[:])
                    print(X2[:])
                    raise ValueError("Arrays are not equal")


def test_numpy_array_of_strings():
    print("Testing numpy array of strings")
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=["abc", "def", "ghi"])
        h5f = h5py.File(filename, "r")
        with LindiH5Store.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            client = LindiClient.from_reference_file_system(rfs)
            X1 = h5f["X"]
            assert isinstance(X1, h5py.Dataset)
            X2 = client["X"]
            assert isinstance(X2, LindiDataset)
            if not _check_equal(X1[:], X2[:]):
                print("WARNING. Arrays are not equal")
                print(X1[:])
                print(X2[:])
                raise ValueError("Arrays are not equal")


def test_attributes():
    print("Testing attributes")
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=[1, 2, 3])
            f["X"].attrs["foo"] = "bar"
            f["X"].attrs["baz"] = 3.14
            f["X"].attrs["qux"] = [1, 2, 3]
            f["X"].attrs["corge"] = np.int32(5)
            f.create_group("group")
            f["group"].attrs["foo"] = "bar2"
            f["group"].attrs["baz"] = 3.15
        h5f = h5py.File(filename, "r")
        with LindiH5Store.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            client = LindiClient.from_reference_file_system(rfs)

            X1 = h5f["X"]
            assert isinstance(X1, h5py.Dataset)
            X2 = client["X"]
            assert isinstance(X2, LindiDataset)

            with pytest.raises(KeyError):
                X2.attrs["a"] = 1  # cannot set attributes on read-only object
            with pytest.raises(KeyError):
                X2.attrs["b"]  # non-existent attribute
            with pytest.raises(KeyError):
                del X2.attrs["foo"]  # cannot delete attributes on read-only object

            for k, v in X2.attrs.items():
                if not _check_equal(v, X1.attrs[k]):
                    print(f"WARNING: {k} attribute mismatch")
                    print(f"  h5: {X1.attrs[k]} ({type(X1.attrs[k])})")
                    print(f"  zarr: {v} ({type(v)})")
                    raise ValueError("Attribute mismatch")
            for k, v in X1.attrs.items():
                if not _check_equal(v, X2.attrs[k]):
                    print(f"WARNING: {k} attribute mismatch")
                    print(f"  h5: {v} ({type(v)})")
                    print(f"  zarr: {X2.attrs[k]} ({type(X2.attrs[k])})")
                    raise ValueError("Attribute mismatch")
            for k in X2.attrs:
                assert k in X1.attrs
            assert len(X2.attrs) == len(X1.attrs)
            assert str(X2.attrs)  # for coverage
            assert repr(X2.attrs)  # for coverage

            group1 = h5f["group"]
            assert isinstance(group1, h5py.Group)
            group2 = client["group"]
            assert isinstance(group2, LindiGroup)

            for k, v in group2.attrs.items():
                if not _check_equal(v, group1.attrs[k]):
                    print(f"WARNING: {k} attribute mismatch")
                    print(f"  h5: {group1.attrs[k]} ({type(group1.attrs[k])})")
                    print(f"  zarr: {v} ({type(v)})")
                    raise ValueError("Attribute mismatch")
            for k, v in group1.attrs.items():
                if not _check_equal(v, group2.attrs[k]):
                    print(f"WARNING: {k} attribute mismatch")
                    print(f"  h5: {v} ({type(v)})")
                    print(f"  zarr: {group2.attrs[k]} ({type(group2.attrs[k])})")
                    raise ValueError("Attribute mismatch")


def _check_equal(a, b):
    # allow comparison of bytes and strings
    if isinstance(a, str):
        a = a.encode()
    if isinstance(b, str):
        b = b.encode()

    # allow comparison of numpy scalars with python scalars
    if np.issubdtype(type(a), np.floating):
        a = float(a)
    if np.issubdtype(type(b), np.floating):
        b = float(b)
    if np.issubdtype(type(a), np.integer):
        a = int(a)
    if np.issubdtype(type(b), np.integer):
        b = int(b)

    # allow comparison of numpy arrays to python lists
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)

    if type(a) != type(b):  # noqa: E721
        return False

    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        return _check_arrays_equal(a, b)

    return a == b


def _check_arrays_equal(a: np.ndarray, b: np.ndarray):
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
