import tempfile
import numpy as np
import h5py
import zarr
import lindi
from lindi.LindiH5ZarrStore.LindiH5ZarrStore import LindiH5ZarrStore
from lindi.LindiH5ZarrStore.LindiH5ZarrStoreOpts import LindiH5ZarrStoreOpts


def test_getitems_local_chunks():
    """Test getitems on LindiH5ZarrStore with a local chunked dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        X = np.random.randn(100, 10)
        with h5py.File(filename, "w") as f:
            f.create_dataset("dataset1", data=X, chunks=(20, 10))

        # Use num_dataset_chunks_threshold=None so chunks are served through store
        opts = LindiH5ZarrStoreOpts(num_dataset_chunks_threshold=None)
        with LindiH5ZarrStore.from_file(filename, url=filename, opts=opts) as store:
            # Read via zarr to verify basic functionality
            arr = zarr.open_array(store=store, path="dataset1", mode="r")
            np.testing.assert_array_equal(arr[:], X)

            # Test getitems with chunk keys
            keys = ["dataset1/0.0", "dataset1/1.0", "dataset1/2.0"]
            results = store.getitems(keys)
            assert len(results) == 3
            for key in keys:
                assert key in results

            # Test getitems with metadata keys
            meta_keys = ["dataset1/.zarray", "dataset1/.zattrs"]
            meta_results = store.getitems(meta_keys)
            assert len(meta_results) == 2
            for key in meta_keys:
                assert key in meta_results

            # Test getitems with non-existent keys (should be skipped)
            mixed_keys = ["dataset1/0.0", "nonexistent/0.0"]
            mixed_results = store.getitems(mixed_keys)
            assert "dataset1/0.0" in mixed_results
            assert "nonexistent/0.0" not in mixed_results


def test_getitems_inline_data():
    """Test getitems with a small dataset that is stored inline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        X = np.array([1, 2, 3], dtype=np.float64)
        with h5py.File(filename, "w") as f:
            f.create_dataset("small", data=X)

        opts = LindiH5ZarrStoreOpts(num_dataset_chunks_threshold=None)
        with LindiH5ZarrStore.from_file(filename, url=filename, opts=opts) as store:
            # Small arrays should be inline
            keys = ["small/0"]
            results = store.getitems(keys)
            assert len(results) == 1


def test_getitems_single_chunk_shortcut():
    """Test that a single remote chunk skips the thread pool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        X = np.random.randn(1000)
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=X, chunks=(1000,))

        opts = LindiH5ZarrStoreOpts(num_dataset_chunks_threshold=None)
        with LindiH5ZarrStore.from_file(filename, url=filename, opts=opts) as store:
            keys = ["data/0"]
            results = store.getitems(keys)
            assert "data/0" in results


def test_external_array_link_via_zarr_store():
    """Test that external array links for local files still work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        X = np.random.randn(50, 12)
        with h5py.File(filename, "w") as f:
            f.create_dataset("dataset1", data=X, chunks=(10, 6))

        # Create a LINDI reference with a low threshold so external array link is used
        with LindiH5ZarrStore.from_file(
            filename,
            url=filename,
            opts=LindiH5ZarrStoreOpts(num_dataset_chunks_threshold=4),
        ) as store:
            rfs = store.to_reference_file_system()

        # Read back through LindiH5pyFile — local external links use h5py directly
        client = lindi.LindiH5pyFile.from_reference_file_system(rfs)
        X2 = client["dataset1"][:]
        np.testing.assert_array_equal(X, X2)


def test_zarr_store_for_external_array():
    """Test creating a LindiH5ZarrStore with num_dataset_chunks_threshold=None
    to serve all chunks (the pattern used for remote external array links)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        X = np.random.randn(200, 10)
        with h5py.File(filename, "w") as f:
            f.create_dataset("dataset1", data=X, chunks=(20, 10))

        # This is the same pattern used in _get_external_zarr_array
        opts = LindiH5ZarrStoreOpts(num_dataset_chunks_threshold=None)
        with LindiH5ZarrStore.from_file(filename, opts=opts, url=filename) as store:
            arr = zarr.open_array(store=store, path="dataset1", mode="r")
            result = arr[:]
            np.testing.assert_array_equal(result, X)

            # Test slicing
            result_slice = arr[10:30, 3:7]
            np.testing.assert_array_equal(result_slice, X[10:30, 3:7])


def test_getitems_empty_keys():
    """Test getitems with empty key list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=np.array([1, 2, 3]))

        opts = LindiH5ZarrStoreOpts(num_dataset_chunks_threshold=None)
        with LindiH5ZarrStore.from_file(filename, url=filename, opts=opts) as store:
            results = store.getitems([])
            assert results == {}


def test_coalesce_byte_ranges():
    """Test that _coalesce_byte_ranges merges adjacent/nearby ranges."""
    from lindi.LindiH5ZarrStore.LindiH5ZarrStore import _coalesce_byte_ranges

    # Empty input
    assert _coalesce_byte_ranges([], merge_gap=256) == []

    # Single chunk — no merging
    chunks = [("a", 0, 100)]
    groups = _coalesce_byte_ranges(chunks, merge_gap=256)
    assert len(groups) == 1
    assert groups[0] == (0, 100, [("a", 0, 100)])

    # Adjacent chunks — should merge
    chunks = [("a", 0, 100), ("b", 100, 100)]
    groups = _coalesce_byte_ranges(chunks, merge_gap=0)
    assert len(groups) == 1
    assert groups[0][0] == 0  # group_start
    assert groups[0][1] == 200  # group_length
    assert len(groups[0][2]) == 2  # two members

    # Chunks with small gap — should merge
    chunks = [("a", 0, 100), ("b", 200, 100)]
    groups = _coalesce_byte_ranges(chunks, merge_gap=100)
    assert len(groups) == 1
    assert groups[0][0] == 0
    assert groups[0][1] == 300

    # Chunks with gap larger than threshold — should NOT merge
    chunks = [("a", 0, 100), ("b", 500, 100)]
    groups = _coalesce_byte_ranges(chunks, merge_gap=100)
    assert len(groups) == 2

    # Unsorted input — should still work
    chunks = [("c", 2000, 100), ("a", 0, 100), ("b", 100, 100)]
    groups = _coalesce_byte_ranges(chunks, merge_gap=50)
    assert len(groups) == 2
    assert groups[0][0] == 0
    assert groups[0][1] == 200
    assert groups[1][0] == 2000

    # max_size — prevents merging when result would be too large
    chunks = [("a", 0, 100), ("b", 100, 100), ("c", 200, 100)]
    groups = _coalesce_byte_ranges(chunks, merge_gap=100, max_size=200)
    assert len(groups) == 2
    assert groups[0][1] == 200  # first two merged
    assert groups[1][1] == 100  # third alone

    # max_size=None — no limit
    groups = _coalesce_byte_ranges(chunks, merge_gap=100, max_size=None)
    assert len(groups) == 1


def test_coalesce_integration():
    """Test that coalesced fetching returns correct data through zarr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        X = np.random.randn(200, 10)
        with h5py.File(filename, "w") as f:
            f.create_dataset("dataset1", data=X, chunks=(20, 10))

        opts = LindiH5ZarrStoreOpts(num_dataset_chunks_threshold=None)
        with LindiH5ZarrStore.from_file(filename, opts=opts, url=filename) as store:
            # Fetch all 10 chunk keys at once — should coalesce
            keys = [f"dataset1/{i}.0" for i in range(10)]
            results = store.getitems(keys)
            assert len(results) == 10

            # Verify data through zarr is correct
            arr = zarr.open_array(store=store, path="dataset1", mode="r")
            np.testing.assert_array_equal(arr[:], X)


if __name__ == "__main__":
    test_getitems_local_chunks()
    test_getitems_inline_data()
    test_getitems_single_chunk_shortcut()
    test_external_array_link_via_zarr_store()
    test_zarr_store_for_external_array()
    test_getitems_empty_keys()
    test_coalesce_byte_ranges()
    test_coalesce_integration()
    print("All tests passed!")
