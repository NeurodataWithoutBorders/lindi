import numpy as np
import h5py
import tempfile
import pytest
import lindi
from lindi import LindiH5ZarrStore


def test_copy_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=[1, 2, 3])
            f.create_dataset("Y", data=[4, 5, 6])
            f['X'].attrs['attr1'] = 'value1'
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_2 = lindi.LindiH5pyFile.from_reference_file_system(rfs, mode="r+")
            assert "X" in h5f_2
            assert "Y" in h5f_2
            with pytest.raises(Exception):
                # This one is not expected to work. Would be difficult to
                # implement since this involves low-level operations on
                # LindiH5pyFile.
                h5f.copy("X", h5f_2, "Z")
            h5f_2.copy("X", h5f_2, "Z")
            assert "Z" in h5f_2
            assert h5f_2["Z"].attrs['attr1'] == 'value1'  # type: ignore
            assert _arrays_are_equal(h5f["X"][()], h5f_2["Z"][()])  # type: ignore
            rfs_copy = store.to_reference_file_system()
            h5f_3 = lindi.LindiH5pyFile.from_reference_file_system(rfs_copy, mode="r+")
            assert "Z" not in h5f_3
            h5f_2.copy("X", h5f_3, "Z")
            assert "Z" in h5f_3
            assert h5f_3["Z"].attrs['attr1'] == 'value1'  # type: ignore
            assert _arrays_are_equal(h5f["X"][()], h5f_3["Z"][()])  # type: ignore


def test_copy_group():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_group("X")
            f.create_group("Y")
            f.create_dataset("X/A", data=[1, 2, 3])
            f.create_dataset("Y/B", data=[4, 5, 6])
            f['X'].attrs['attr1'] = 'value1'
        h5f = h5py.File(filename, "r")
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            rfs = store.to_reference_file_system()
            h5f_2 = lindi.LindiH5pyFile.from_reference_file_system(rfs, mode="r+")
            assert "X" in h5f_2
            assert "Y" in h5f_2
            with pytest.raises(Exception):
                # This one is not expected to work. Would be difficult to
                # implement since this involves low-level operations on
                # LindiH5pyFile.
                h5f.copy("X", h5f_2, "Z")
            h5f_2.copy("X", h5f_2, "Z")
            assert "Z" in h5f_2
            assert h5f_2["Z"].attrs['attr1'] == 'value1'
            assert "A" in h5f_2["Z"]
            assert _arrays_are_equal(h5f["X/A"][()], h5f_2["Z/A"][()])  # type: ignore
            rfs_copy = store.to_reference_file_system()
            h5f_3 = lindi.LindiH5pyFile.from_reference_file_system(rfs_copy, mode="r+")
            assert "Z" not in h5f_3
            h5f_2.copy("X", h5f_3, "Z")
            assert "Z" in h5f_3
            assert h5f_3["Z"].attrs['attr1'] == 'value1'
            assert "A" in h5f_3["Z"]
            assert _arrays_are_equal(h5f["X/A"][()], h5f_3["Z/A"][()])  # type: ignore


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
    test_copy_dataset()
