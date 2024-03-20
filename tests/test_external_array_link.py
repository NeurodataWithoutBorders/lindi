import tempfile
import numpy as np
import h5py
import lindi


def test_external_array_link():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        X = np.random.randn(50, 12)
        with h5py.File(filename, "w") as f:
            f.create_dataset("dataset1", data=X, chunks=(10, 6))
        with lindi.LindiH5ZarrStore.from_file(
            filename,
            url=filename,
            opts=lindi.LindiH5ZarrStoreOpts(
                num_dataset_chunks_threshold=4
            )
        ) as store:
            rfs = store.to_reference_file_system()
            client = lindi.LindiH5pyFile.from_reference_file_system(rfs)
            X2 = client["dataset1"][:]  # type: ignore
            assert np.array_equal(X, X2)


if __name__ == "__main__":
    test_external_array_link()
