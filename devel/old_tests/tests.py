import tempfile
import numpy as np
import h5py
from lindi import LindiH5Store, LindiClient, LindiGroup, LindiDataset


def test_scalar_dataset():
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


def _check_equal(a, b):
    # allow comparison of bytes and strings
    if isinstance(a, str):
        a = a.encode()
    if isinstance(b, str):
        b = b.encode()

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


def test_numpy_array():
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
                if not _check_equal(X1[:], X2[:]):
                    print("WARNING. Arrays are not equal")
                    print(X1[:])
                    print(X2[:])


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


def test_attributes():
    print("Testing attributes")
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=[1, 2, 3])
            f["X"].attrs["foo"] = "bar"
            f["X"].attrs["baz"] = 3.14
            f["X"].attrs["qux"] = [1, 2, 3]
            f["X"].attrs["quux"] = {"a": 1, "b": 2, "c": 3}
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
            if X1.attrs["foo"] != X2.attrs["foo"]:
                print("WARNING. Attributes are not equal")
                print(X1.attrs["foo"])
                print(X2.attrs["foo"])
            if X1.attrs["baz"] != X2.attrs["baz"]:
                print("WARNING. Attributes are not equal")
                print(X1.attrs["baz"])
                print(X2.attrs["baz"])
            group1 = h5f["group"]
            assert isinstance(group1, h5py.Group)
            group2 = client["group"]
            assert isinstance(group2, LindiGroup)
            if group1.attrs["foo"] != group2.attrs["foo"]:
                print("WARNING. Attributes are not equal")
                print(group1.attrs["foo"])
                print(group2.attrs["foo"])
            if group1.attrs["baz"] != group2.attrs["baz"]:
                print("WARNING. Attributes are not equal")
                print(group1.attrs["baz"])
                print(group2.attrs["baz"])


if __name__ == "__main__":
    test_scalar_dataset()
    test_numpy_array()
    test_numpy_array_of_strings()
    test_attributes()
