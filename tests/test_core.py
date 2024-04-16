import pytest
import numpy as np
import h5py
import tempfile
import lindi
from lindi import LindiH5ZarrStore
from utils import arrays_are_equal, lists_are_equal


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
                assert lists_are_equal(h5f_2.attrs["list1"], h5f.attrs["list1"])
                assert lists_are_equal(h5f_2.attrs["tuple1"], h5f.attrs["tuple1"])
                assert arrays_are_equal(np.array(h5f_2.attrs["array1"]), h5f.attrs["array1"])
                assert h5f_2["dataset1"].attrs["test_attr1"] == h5f["dataset1"].attrs["test_attr1"]  # type: ignore
                assert h5f_2["dataset1"].id
                assert arrays_are_equal(h5f_2["dataset1"][()], h5f["dataset1"][()])  # type: ignore
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
                assert arrays_are_equal(ds1[()], ds2[()])
                ds1 = h5f['soft_link/dataset1']
                assert isinstance(ds1, h5py.Dataset)
                ds2 = h5f_2['soft_link/dataset1']
                assert isinstance(ds2, h5py.Dataset)
                assert arrays_are_equal(ds1[()], ds2[()])
                ds1 = h5f['group_target/dataset1']
                assert isinstance(ds1, h5py.Dataset)
                ds2 = h5f_2['group_target/dataset1']
                assert isinstance(ds2, h5py.Dataset)
                assert arrays_are_equal(ds1[()], ds2[()])


def test_arrays_of_compound_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            dt = np.dtype([("x", "i4"), ("y", "f8")])
            dataset1 = f.create_dataset("dataset1", data=[(1, 3.14), (2, 6.28)], dtype=dt)
            dt = np.dtype([("a", "i4"), ("b", "f8"), ("c", "S10")])
            dataset2 = f.create_dataset("dataset2", data=[(1, 3.14, "abc"), (2, 6.28, "def")], dtype=dt)
            # how about references!
            dt = np.dtype([("a", "i4"), ("b", "f8"), ("c", h5py.special_dtype(ref=h5py.Reference))])
            f.create_dataset("dataset3", data=[(1, 3.14, dataset1.ref), (2, 6.28, dataset2.ref)], dtype=dt)
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_2 = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            ds1_1 = h5f['dataset1']
            assert isinstance(ds1_1, h5py.Dataset)
            ds1_2 = h5f_2['dataset1']
            assert isinstance(ds1_2, h5py.Dataset)
            assert ds1_1.dtype == ds1_2.dtype
            assert arrays_are_equal(ds1_1['x'][()], ds1_2['x'][()])  # type: ignore
            assert arrays_are_equal(ds1_1['y'][()], ds1_2['y'][()])  # type: ignore
            ds2_1 = h5f['dataset2']
            assert isinstance(ds2_1, h5py.Dataset)
            ds2_2 = h5f_2['dataset2']
            assert isinstance(ds2_2, h5py.Dataset)
            assert ds2_1.dtype == ds2_2.dtype
            assert arrays_are_equal(ds2_1['a'][()], ds2_2['a'][()])  # type: ignore
            assert arrays_are_equal(ds2_1['b'][()], ds2_2['b'][()])  # type: ignore
            assert arrays_are_equal(ds2_1['c'][()], ds2_2['c'][()])  # type: ignore
            ds3_1 = h5f['dataset3']
            assert isinstance(ds3_1, h5py.Dataset)
            ds3_2 = h5f_2['dataset3']
            assert isinstance(ds3_2, h5py.Dataset)
            assert ds3_1.dtype == ds3_2.dtype
            assert ds3_1.dtype['c'] == ds3_2.dtype['c']
            assert ds3_2.dtype['c'] == h5py.special_dtype(ref=h5py.Reference)
            target1 = h5f[ds3_1['c'][0]]
            assert isinstance(target1, h5py.Dataset)
            target2 = h5f_2[ds3_2['c'][0]]
            assert isinstance(target2, h5py.Dataset)


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
            assert arrays_are_equal(ds1_1['x'][()], ds1_2['x'][()])  # type: ignore
            ref1 = ds1_1['y'][0]
            ref2 = ds1_2['y'][0]
            assert isinstance(ref1, h5py.Reference)
            assert isinstance(ref2, h5py.Reference)
            target1 = h5f[ref1]
            assert isinstance(target1, h5py.Dataset)
            target2 = h5f_2[ref2]
            assert isinstance(target2, h5py.Dataset)
            assert arrays_are_equal(target1[()], target2[()])


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
            assert X2.size == 1
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
            assert lists_are_equal(X1[:].tolist(), [x.encode() for x in X2[:]])  # type: ignore


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
            f["X"].attrs['float_list'] = [np.nan, np.inf, -np.inf, 23]
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            client = lindi.LindiH5pyFile.from_reference_file_system(rfs)

            X1 = h5f["X"]
            assert isinstance(X1, h5py.Dataset)
            X2 = client["X"]
            assert isinstance(X2, lindi.LindiH5pyDataset)

            nanval = X1.attrs["nan"]
            assert isinstance(nanval, float) and np.isnan(nanval)
            assert X1.attrs["inf"] == np.inf
            assert X1.attrs["ninf"] == -np.inf
            assert lists_are_equal(X1.attrs['float_list'], [np.nan, np.inf, -np.inf, 23])

            nanval = X2.attrs["nan"]
            assert isinstance(nanval, float) and np.isnan(nanval)
            assert X2.attrs["inf"] == np.inf
            assert X2.attrs["ninf"] == -np.inf
            assert lists_are_equal(X2.attrs['float_list'], [np.nan, np.inf, -np.inf, 23])

        for test_string in ["NaN", "Infinity", "-Infinity", "Not-illegal"]:
            filename = f"{tmpdir}/illegal_string.h5"
            with h5py.File(filename, "w") as f:
                f.create_dataset("X", data=[1, 2, 3])
                f["X"].attrs["test_string"] = test_string
            with LindiH5ZarrStore.from_file(filename, url=filename) as store:
                if test_string in ["NaN", "Infinity", "-Infinity"]:
                    with pytest.raises(Exception):
                        rfs = store.to_reference_file_system()
                else:
                    rfs = store.to_reference_file_system()
                    client = lindi.LindiH5pyFile.from_reference_file_system(rfs)
                    assert client["X"].attrs["test_string"] == test_string  # type: ignore


def test_reference_file_system_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=[1, 2, 3])
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs_fname = f'{tmpdir}/test.lindi.json'
            store.to_file(rfs_fname)
            client = lindi.LindiH5pyFile.from_reference_file_system(rfs_fname)
            X = client["X"]
            assert isinstance(X, lindi.LindiH5pyDataset)
            assert lists_are_equal(X[()], [1, 2, 3])


def test_lindi_reference_file_system_store():
    from lindi.LindiH5pyFile.LindiReferenceFileSystemStore import LindiReferenceFileSystemStore

    # test for invalid rfs
    rfs = {"rfs_misspelled": {"a": "a"}}  # misspelled
    with pytest.raises(Exception):
        store = LindiReferenceFileSystemStore(rfs)
    rfs = {"refs": {"a": 1}}  # invalid value
    with pytest.raises(Exception):
        store = LindiReferenceFileSystemStore(rfs)
    rfs = {"refs": {"a": ["a", 1]}}  # invalid list
    with pytest.raises(Exception):
        store = LindiReferenceFileSystemStore(rfs)
    rfs = {"refs": {"a": ["a", 1, 2, 3]}}  # invalid list
    with pytest.raises(Exception):
        store = LindiReferenceFileSystemStore(rfs)
    rfs = {"refs": {"a": [1, 2, 3]}}  # invalid list
    with pytest.raises(Exception):
        store = LindiReferenceFileSystemStore(rfs)
    rfs = {"refs": {"a": ['a', 'a', 2]}}  # invalid list
    with pytest.raises(Exception):
        store = LindiReferenceFileSystemStore(rfs)
    rfs = {"refs": {"a": ['a', 1, 'a']}}  # invalid list
    with pytest.raises(Exception):
        store = LindiReferenceFileSystemStore(rfs)
    rfs = {"refs": {"a": "base64:abc+++"}}  # invalid base64
    store = LindiReferenceFileSystemStore(rfs)
    with pytest.raises(Exception):
        store["a"]
    with pytest.raises(Exception):
        store[{}]  # invalid key type # type: ignore
    rfs = {"refs": {"a": {}}}  # invalid value
    with pytest.raises(Exception):
        store = LindiReferenceFileSystemStore(rfs)

    rfs = {"refs": {"a": "abc"}}
    store = LindiReferenceFileSystemStore(rfs)
    assert store.is_readable()
    assert store.is_writeable()
    assert store.is_listable()
    assert not store.is_erasable()
    assert len(store) == 1
    assert "a" in store
    assert "b" not in store
    assert store["a"] == b"abc"


def test_lindi_h5py_reference():
    from lindi.LindiH5pyFile.LindiH5pyReference import LindiH5pyReference
    obj = {
        "object_id": "object_id",
        "path": "path",
        "source": "source",
        "source_object_id": "source_object_id",
    }
    ref = LindiH5pyReference(obj)
    assert repr(ref) == "LindiH5pyReference(object_id, path)"
    assert str(ref) == "LindiH5pyReference(object_id, path)"
    assert ref._object_id == "object_id"
    assert ref._path == "path"
    assert ref._source == "source"
    assert ref._source_object_id == "source_object_id"
    assert ref.__class__.__name__ == "LindiH5pyReference"
    assert isinstance(ref, h5py.h5r.Reference)
    assert isinstance(ref, LindiH5pyReference)


def test_lindi_h5_zarr_store():
    # Test that exceptions are raised as expected
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("dataset1", data=[1, 2, 3])
            f.create_group("group1")
            f.create_dataset("scalar_dataset", data=42)
        # Store is closed
        store = LindiH5ZarrStore.from_file(filename)
        store.close()
        store_is_closed_msg = "Store is closed"
        with pytest.raises(Exception, match=store_is_closed_msg):
            if 'dataset1/.zarray' in store:
                pass
        with pytest.raises(Exception, match=store_is_closed_msg):
            store["dataset1/.zarray"]
        with pytest.raises(Exception, match=store_is_closed_msg):
            store["dataset1/.zattrs"]
        with pytest.raises(Exception, match=store_is_closed_msg):
            store["group1/.zgroup"]
        with pytest.raises(Exception, match=store_is_closed_msg):
            store["group1/.zattrs"]
        with pytest.raises(Exception, match=store_is_closed_msg):
            store["dataset1/0"]
        with pytest.raises(Exception, match=store_is_closed_msg):
            store.listdir()
        with pytest.raises(Exception, match=store_is_closed_msg):
            store.to_reference_file_system()
        with pytest.raises(Exception, match=store_is_closed_msg):
            store.to_file("test.json")
        with pytest.raises(Exception, match=store_is_closed_msg):
            store._get_chunk_file_bytes_data("dataset1", "0")

        # Nonexistent item
        store = LindiH5ZarrStore.from_file(filename)
        assert 'nonexistent/.zattrs' not in store
        with pytest.raises(KeyError):
            store["nonexistent/.zattrs"]
        assert 'nonexistent/.zgroup' not in store
        with pytest.raises(Exception, match="Item nonexistent is not a group"):
            store["nonexistent/.zgroup"]
        assert 'nonexistent/.zarray' not in store
        with pytest.raises(Exception, match="Item nonexistent is not a dataset"):
            store["nonexistent/.zarray"]
        assert 'nonexistent/0' not in store
        with pytest.raises(Exception, match="Item nonexistent is not a dataset"):
            store["nonexistent/0"]

        # Key error
        store = LindiH5ZarrStore.from_file(filename)
        with pytest.raises(KeyError):
            store['']
        assert '' not in store
        with pytest.raises(KeyError):
            store["nonexistent/.zattrs"]

        # Unsupported file type
        with pytest.raises(Exception, match="Unsupported file type: zarr"):
            store.to_file("test.json", file_type="zarr")  # type: ignore

        # URL is not set
        store = LindiH5ZarrStore.from_file(filename, url=None)
        with pytest.raises(Exception, match="You must specify a url to create a reference file system"):
            store.to_reference_file_system()

        # External links not supported
        with h5py.File(f'{tmpdir}/external.h5', 'w') as f:
            grp = f.create_group('group1')
            grp.attrs['attr1'] = 'value1'
        with h5py.File(filename, "a") as f:
            f["external_link"] = h5py.ExternalLink(f'{tmpdir}/external.h5', 'group1')
        store = LindiH5ZarrStore.from_file(filename, url=filename)
        with pytest.raises(Exception, match="External links not supported: external_link"):
            print(store["external_link/.zattrs"])

        store = LindiH5ZarrStore.from_file(filename, url=filename)
        with pytest.raises(Exception, match="Setting items is not allowed"):
            store["dataset1/.zattrs"] = b"{}"
        with pytest.raises(Exception, match="Deleting items is not allowed"):
            del store["dataset1/.zattrs"]
        with pytest.raises(Exception, match="Not implemented"):
            iter(store)
        with pytest.raises(Exception, match="Not implemented"):
            len(store)

        store = LindiH5ZarrStore.from_file(filename, url=filename)
        assert 'dataset1/0.0' not in store
        assert 'dataset1/1' not in store
        assert 'scalar_dataset/0' in store
        assert 'scalar_dataset/1' not in store


def test_numpy_array_of_byte_strings():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=np.array([b"abc", b"def", b"ghi"]))
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_2 = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            X1 = h5f['X']
            assert isinstance(X1, h5py.Dataset)
            X2 = h5f_2['X']
            assert isinstance(X2, h5py.Dataset)
            assert lists_are_equal(X1[:].tolist(), X2[:].tolist())  # type: ignore


if __name__ == '__main__':
    pass
