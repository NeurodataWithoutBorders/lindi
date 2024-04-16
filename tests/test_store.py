import h5py
import tempfile
import lindi
from utils import lists_are_equal


def test_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("dataset1", data=[1, 2, 3])
            group1 = f.create_group("group1")
            group1.create_group("group2")
            group1.create_dataset("dataset2", data=[4, 5, 6])
        with lindi.LindiH5ZarrStore.from_file(filename, url=filename) as store:
            store.to_file(f"{tmpdir}/test.lindi.json")  # for coverage
            a = store.listdir('')
            assert _lists_are_equal_as_sets(a, ['dataset1', 'group1'])
            b = store.listdir('group1')
            assert _lists_are_equal_as_sets(b, ['group2', 'dataset2'])
            c = store.listdir('group1/group2')
            assert _lists_are_equal_as_sets(c, [])
            assert '.zattrs' in store
            assert '.zgroup' in store
            assert 'dataset1' not in store
            assert 'dataset1/.zattrs' in store
            assert 'dataset1/.zarray' in store
            assert 'dataset1/.zgroup' not in store
            assert 'dataset1/0' in store
            assert 'group1' not in store
            assert 'group1/.zattrs' in store
            assert 'group1/.zgroup' in store
            assert 'group1/.zarray' not in store
            assert 'group1/group2' not in store
            assert 'group1/group2/.zattrs' in store
            assert 'group1/group2/.zgroup' in store
            assert 'group1/group2/.zarray' not in store
            assert 'group1/dataset2' not in store
            assert 'group1/dataset2/.zattrs' in store
            assert 'group1/dataset2/.zarray' in store
            assert 'group1/dataset2/.zgroup' not in store
            assert 'group1/dataset2/0' in store
            client = lindi.LindiH5pyFile.from_zarr_store(store)
            X = client["dataset1"][:]  # type: ignore
            assert lists_are_equal(X, [1, 2, 3])
            Y = client["group1/dataset2"][:]  # type: ignore
            assert lists_are_equal(Y, [4, 5, 6])


def _lists_are_equal_as_sets(a, b):
    return set(a) == set(b)
