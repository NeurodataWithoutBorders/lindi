import tempfile
import numpy as np
import h5py
import lindi


def test_numpy_array_conversion():
    """Test that LindiH5pyDataset supports np.asarray() and np.atleast_2d().

    Regression test for https://github.com/NeurodataWithoutBorders/lindi/issues/120
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_fname = f'{tmpdir}/test.h5'
        lindi_json_fname = f'{tmpdir}/test.lindi.json'

        # Create a simple HDF5 file with a 1D dataset
        with h5py.File(h5_fname, 'w') as f:
            f.create_dataset('data', data=np.arange(100, dtype=np.float64))
            f.create_dataset('scalar', data=42.0)

        # Convert to lindi
        with lindi.LindiH5pyFile.from_hdf5_file(h5_fname, url=h5_fname) as f:
            f.write_lindi_file(lindi_json_fname)

        # Open the lindi file and test numpy conversions
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
            ds = f['data']

            # Test _is_empty
            assert ds._is_empty is False

            # Test np.asarray - this triggers __array__
            arr = np.asarray(ds)
            assert arr.shape == (100,)
            np.testing.assert_array_equal(arr, np.arange(100, dtype=np.float64))

            # Test np.atleast_2d - this is what failed in the issue
            arr2d = np.atleast_2d(ds)
            assert arr2d.shape == (1, 100)

            # Test np.array with dtype conversion
            arr_int = np.array(ds, dtype=np.int32)
            assert arr_int.dtype == np.int32

            # Test scalar dataset
            sc = f['scalar']
            arr_sc = np.asarray(sc)
            assert arr_sc.shape == ()
            assert float(arr_sc) == 42.0
