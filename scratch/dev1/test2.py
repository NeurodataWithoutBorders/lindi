import h5py
import numpy as np
import zarr
from LindiH5Store import LindiH5Store


def test_lindi_h5_store():
    """Test the LindiH5Store class."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_h5_fname = f'{tmpdir}/test.h5'
        with h5py.File(test_h5_fname, 'w') as h5f:
            h5f.create_dataset('data', data=np.arange(100).reshape(10, 10), chunks=(5, 5))
        with open(test_h5_fname, 'rb') as f:
            h5f = h5py.File(f, 'r')
            store = LindiH5Store(f)
            root = zarr.open_group(store)
            data = root['data']
            A1x = h5f['data']
            assert isinstance(A1x, h5py.Dataset)
            A1 = A1x[:]
            A2 = data[:]
            assert isinstance(A2, np.ndarray)
            assert np.array_equal(A1, A2)


if __name__ == '__main__':
    test_lindi_h5_store()
