import pytest
import numpy as np
import h5py
import tempfile
import lindi
from lindi import LindiH5ZarrStore


def test_variety():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("dataset1", data=[1, 2, 3])
            f.create_group("group1")
            f.attrs["int1"] = 1
            f.attrs["float1"] = 3.14
            f.attrs["str1"] = "abc"
            f.attrs["bytes1"] = b"def"
            f.attrs["list1"] = [1, 2, 3]
            f.attrs["tuple1"] = (3, 4, 5)
            f.attrs["array1"] = np.arange(10)
            f.attrs["dataset1_ref"] = f["dataset1"].ref
            f.attrs["group1_ref"] = f["group1"].ref
            f["dataset1"].attrs["test_attr1"] = "attribute-of-dataset1"
            f["group1"].attrs["test_attr2"] = "attribute-of-group1"
        h5f = h5py.File(filename, "r")
        h5f_wrapped = lindi.LindiH5pyFile.from_h5py_file(h5f)
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_rfs = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            for h5f_2 in [h5f_rfs, h5f_wrapped]:
                assert h5f_2.attrs["int1"] == h5f.attrs["int1"]
                assert h5f_2.attrs["float1"] == h5f.attrs["float1"]
                assert h5f_2.attrs["str1"] == h5f.attrs["str1"]
                assert h5f_2.attrs["bytes1"] == h5f.attrs["bytes1"]
                assert _lists_are_equal(h5f_2.attrs["list1"], h5f.attrs["list1"])
                assert _lists_are_equal(h5f_2.attrs["tuple1"], h5f.attrs["tuple1"])
                assert _arrays_are_equal(np.array(h5f_2.attrs["array1"]), h5f.attrs["array1"])
                assert h5f_2["dataset1"].attrs["test_attr1"] == h5f["dataset1"].attrs["test_attr1"]  # type: ignore
                assert _arrays_are_equal(h5f_2["dataset1"][()], h5f["dataset1"][()])  # type: ignore
                assert h5f_2["group1"].attrs["test_attr2"] == h5f["group1"].attrs["test_attr2"]  # type: ignore
                target_1 = h5f[h5f.attrs["dataset1_ref"]]
                target_2 = h5f_2[h5f_2.attrs["dataset1_ref"]]
                assert target_1.attrs["test_attr1"] == target_2.attrs["test_attr1"]  # type: ignore
                target_1 = h5f[h5f.attrs["group1_ref"]]
                target_2 = h5f_2[h5f_2.attrs["group1_ref"]]
                assert target_1.attrs["test_attr2"] == target_2.attrs["test_attr2"]  # type: ignore


def test_soft_links():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            g = f.create_group('group_target')
            g.attrs['foo'] = 'bar'
            g.create_dataset('dataset1', data=[5, 6, 7])
            f['soft_link'] = h5py.SoftLink('/group_target')
        h5f = h5py.File(filename, "r")
        h5f_wrapped = lindi.LindiH5pyFile.from_h5py_file(h5f)
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_rfs = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            for h5f_2 in [h5f_rfs, h5f_wrapped]:
                g1 = h5f['group_target']
                assert isinstance(g1, h5py.Group)
                g2 = h5f_2['group_target']
                assert isinstance(g2, h5py.Group)
                assert g1.attrs['foo'] == g2.attrs['foo']  # type: ignore
                with pytest.raises(TypeError):
                    g1[np.array([0, 1, 2])]
                h1 = h5f['soft_link']
                assert isinstance(h1, h5py.Group)
                h2 = h5f_2['soft_link']
                assert isinstance(h2, h5py.Group)
                assert h1.attrs['foo'] == h2.attrs['foo']  # type: ignore
                # this is tricky: it seems that with h5py, the name of the soft link
                # is the source name. So the following assertion will fail.
                # assert h1.name == h2.name
                k1 = h5f.get('soft_link', getlink=True)
                k2 = h5f_2.get('soft_link', getlink=True)
                assert isinstance(k1, h5py.SoftLink)
                assert isinstance(k2, h5py.SoftLink)
                ds1 = h5f['soft_link']['dataset1']  # type: ignore
                assert isinstance(ds1, h5py.Dataset)
                ds2 = h5f_2['soft_link']['dataset1']  # type: ignore
                assert isinstance(ds2, h5py.Dataset)
                assert _arrays_are_equal(ds1[()], ds2[()])
                ds1 = h5f['soft_link/dataset1']
                assert isinstance(ds1, h5py.Dataset)
                ds2 = h5f_2['soft_link/dataset1']
                assert isinstance(ds2, h5py.Dataset)
                assert _arrays_are_equal(ds1[()], ds2[()])
                ds1 = h5f['group_target/dataset1']
                assert isinstance(ds1, h5py.Dataset)
                ds2 = h5f_2['group_target/dataset1']
                assert isinstance(ds2, h5py.Dataset)
                assert _arrays_are_equal(ds1[()], ds2[()])


def test_arrays_of_compound_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            dt = np.dtype([("x", "i4"), ("y", "f8")])
            f.create_dataset("dataset1", data=[(1, 3.14), (2, 6.28)], dtype=dt)
            dt = np.dtype([("a", "i4"), ("b", "f8"), ("c", "S10")])
            f.create_dataset("dataset2", data=[(1, 3.14, "abc"), (2, 6.28, "def")], dtype=dt)
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_2 = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            ds1_1 = h5f['dataset1']
            assert isinstance(ds1_1, h5py.Dataset)
            ds1_2 = h5f_2['dataset1']
            assert isinstance(ds1_2, h5py.Dataset)
            assert ds1_1.dtype == ds1_2.dtype
            assert _arrays_are_equal(ds1_1['x'][()], ds1_2['x'][()])  # type: ignore
            assert _arrays_are_equal(ds1_1['y'][()], ds1_2['y'][()])  # type: ignore
            ds2_1 = h5f['dataset2']
            assert isinstance(ds2_1, h5py.Dataset)
            ds2_2 = h5f_2['dataset2']
            assert isinstance(ds2_2, h5py.Dataset)
            assert ds2_1.dtype == ds2_2.dtype
            assert _arrays_are_equal(ds2_1['a'][()], ds2_2['a'][()])  # type: ignore
            assert _arrays_are_equal(ds2_1['b'][()], ds2_2['b'][()])  # type: ignore
            assert _arrays_are_equal(ds2_1['c'][()], ds2_2['c'][()])  # type: ignore


def test_arrays_of_compound_dtype_with_references():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            dt = np.dtype([("x", "i4"), ("y", h5py.special_dtype(ref=h5py.Reference))])
            Y_ds = f.create_dataset("Y", data=[1, 2, 3])
            f.create_dataset("dataset1", data=[(1, Y_ds.ref), (2, Y_ds.ref)], dtype=dt)
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_2 = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            ds1_1 = h5f['dataset1']
            assert isinstance(ds1_1, h5py.Dataset)
            ds1_2 = h5f_2['dataset1']
            assert isinstance(ds1_2, h5py.Dataset)
            assert ds1_1.dtype == ds1_2.dtype
            assert _arrays_are_equal(ds1_1['x'][()], ds1_2['x'][()])  # type: ignore
            ref1 = ds1_1['y'][0]
            ref2 = ds1_2['y'][0]
            assert isinstance(ref1, h5py.Reference)
            assert isinstance(ref2, h5py.Reference)
            target1 = h5f[ref1]
            assert isinstance(target1, h5py.Dataset)
            target2 = h5f_2[ref2]
            assert isinstance(target2, h5py.Dataset)
            assert _arrays_are_equal(target1[()], target2[()])


def test_scalar_arrays():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=1)
            f.create_dataset("Y", data=3.14)
            f.create_dataset("Z", data="abc")
            f.create_dataset("W", data=b"def")
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_2 = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            X1 = h5f['X']
            assert isinstance(X1, h5py.Dataset)
            X2 = h5f_2['X']
            assert isinstance(X2, h5py.Dataset)
            assert X1[()] == X2[()]
            Y1 = h5f['Y']
            assert isinstance(Y1, h5py.Dataset)
            Y2 = h5f_2['Y']
            assert isinstance(Y2, h5py.Dataset)
            assert Y1[()] == Y2[()]
            Z1 = h5f['Z']
            assert isinstance(Z1, h5py.Dataset)
            Z2 = h5f_2['Z']
            assert isinstance(Z2, h5py.Dataset)
            # Note that encode is needed because Z1[()] is a bytes
            assert Z1[()] == Z2[()].encode()  # type: ignore
            W1 = h5f['W']
            assert isinstance(W1, h5py.Dataset)
            W2 = h5f_2['W']
            assert isinstance(W2, h5py.Dataset)
            # Note that encode is needed because W2[()] is a str
            assert W1[()] == W2[()].encode()  # type: ignore


def test_arrays_of_strings():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=["abc", "def", "ghi"])
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_2 = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            X1 = h5f['X']
            assert isinstance(X1, h5py.Dataset)
            X2 = h5f_2['X']
            assert isinstance(X2, h5py.Dataset)
            assert _lists_are_equal(X1[:].tolist(), [x.encode() for x in X2[:]])  # type: ignore


def test_numpy_arrays():
    array_1 = ("1", np.arange(60).reshape(3, 20), (3, 7))
    array_2 = ("2", np.arange(60).reshape(3, 20), None)
    array_boolean = ("3", np.array([[True, False, True], [False, True, False]]), None)
    for label, array, chunks in [array_1, array_2, array_boolean]:
        print(f"Testing numpy array {label}")
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/test.h5"
            with h5py.File(filename, "w") as f:
                f.create_dataset("X", data=array, chunks=chunks)
            with LindiH5ZarrStore.from_file(
                filename, url=filename
            ) as store:  # set url so that a reference file system can be created
                rfs = store.to_reference_file_system()
                client = lindi.LindiH5pyFile.from_reference_file_system(rfs)
                h5f = h5py.File(filename, "r")
                X1 = h5f["X"]
                assert isinstance(X1, h5py.Dataset)
                X2 = client["X"]
                assert isinstance(X2, lindi.LindiH5pyDataset)

                assert X1.shape == X2.shape
                assert X1.dtype == X2.dtype
                assert X1.size == X2.size
                assert X1.nbytes == X2.nbytes
                assert len(X1) == len(X2)


def test_nan_inf_attributes():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=[1, 2, 3])
            f["X"].attrs["nan"] = np.nan
            f["X"].attrs["inf"] = np.inf
            f["X"].attrs["ninf"] = -np.inf
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            client = lindi.LindiH5pyFile.from_reference_file_system(rfs)

            X1 = h5f["X"]
            assert isinstance(X1, h5py.Dataset)
            X2 = client["X"]
            assert isinstance(X2, lindi.LindiH5pyDataset)

            assert X2.attrs["nan"] == "NaN"
            assert X2.attrs["inf"] == "Infinity"
            assert X2.attrs["ninf"] == "-Infinity"


def test_reference_file_system_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=[1, 2, 3])
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs_fname = f'{tmpdir}/test.zarr.json'
            store.to_file(rfs_fname)
            client = lindi.LindiH5pyFile.from_reference_file_system(rfs_fname)
            X = client["X"]
            assert isinstance(X, lindi.LindiH5pyDataset)
            assert _lists_are_equal(X[()], [1, 2, 3])


def _lists_are_equal(a, b):
    if len(a) != len(b):
        return False
    for aa, bb in zip(a, b):
        if aa != bb:
            return False
    return True


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


if __name__ == '__main__':
    test_scalar_arrays()
