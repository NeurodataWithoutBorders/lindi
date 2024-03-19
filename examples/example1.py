import numpy as np
import h5py
import tempfile
import lindi
from lindi import LindiH5ZarrStore, LindiH5pyFile, LindiDataset
from _check_equal import _check_equal


def example1():
    # In this example, we create a temporary hdf5 file, with some sample
    # datasets, groups, and attributes. We then load that file using
    # LindiH5ZarrStore which is a zarr storage backend providing read-only view of
    # that hdf5 file. We then create a reference file system and use that to
    # create a LindiH5pyFile, which mimics the h5py API. We then compare the
    # datasets, groups, and attributes of the original hdf5 file with those of
    # the LindiH5pyFile.

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Creating an example hdf5 file")
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.attrs['top_level_attr'] = "top_level_value"
            f.create_dataset("X", data=[1, 2, 3])
            f["X"].attrs["foo"] = "bar"
            f["X"].attrs["baz"] = 3.14
            f["X"].attrs["qux"] = [1, 2, 3]
            f["X"].attrs["corge"] = np.int32(5)
            f.create_dataset("scalar_dataset", data=42, dtype="int32")
            f.create_group("group")
            f["group"].attrs["foo"] = "bar2"
            f["group"].attrs["baz"] = 3.15
        h5f = h5py.File(filename, "r")

        print("Creating a LindiH5ZarrStore from the hdf5 file")
        # We set url to filename so that the references can point to a local file
        # but normally, this would be a remote URL
        with LindiH5ZarrStore.from_file(filename, url=filename) as store:
            print("Creating a reference file system from the LindiH5ZarrStore")
            rfs_fname = f"{tmpdir}/example.zarr.json"
            store.to_file(rfs_fname)

            print("Creating a LindiH5pyFile from the reference file system")
            client = LindiH5pyFile.from_file(rfs_fname)

            print("Comparing dataset: X")
            X1 = h5f["X"]
            assert isinstance(X1, h5py.Dataset)
            X2 = client["X"]
            assert isinstance(X2, LindiDataset)
            assert len(X1) == len(X2)
            assert X1.shape == X2.shape
            assert X1.dtype == X2.dtype
            assert X1.size == X2.size
            assert X1.nbytes == X2.nbytes
            assert X1[0] == X2[0]
            assert X1[1] == X2[1]
            assert X1[2] == X2[2]
            assert _check_equal(X1[0:2], X2[0:2])
            assert _check_equal(X1[1:], X2[1:])
            assert _check_equal(X1[:], X2[:])
            assert _check_equal(X1[...], X2[...])
            for k, v in X2.attrs.items():
                assert k in X1.attrs
                assert _check_equal(v, X1.attrs[k])
            for k, v in X1.attrs.items():
                assert k in X2.attrs
                assert _check_equal(v, X2.attrs[k])

            print("Comparing scalar dataset: scalar_dataset")
            scalar_dataset1 = h5f["scalar_dataset"]
            assert isinstance(scalar_dataset1, h5py.Dataset)
            scalar_dataset2 = client["scalar_dataset"]
            assert isinstance(scalar_dataset2, LindiDataset)
            assert scalar_dataset1[()] == scalar_dataset2[()]

            print("Comparing group: group")
            G1 = h5f["group"]
            G2 = client["group"]
            for k, v in G1.attrs.items():
                if not isinstance(G2, lindi.LindiReference):
                    assert k in G2.attrs
                    assert _check_equal(v, G2.attrs[k])

            print("Comparing root group")
            for k, v in h5f.attrs.items():
                assert k in h5f.attrs
                assert _check_equal(v, h5f.attrs[k])
            for k in client.keys():
                assert k in h5f.keys()
            for k in h5f.keys():
                assert k in client.keys()


if __name__ == "__main__":
    example1()
