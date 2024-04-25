import lindi.h5py_patch  # noqa: F401
import numpy as np
import h5py
from h5py import File


def test_patch():
    fname = 'test.lindi.json'
    with File(fname, 'w') as f:
        assert isinstance(f, File)
        f.create_dataset('data', data=np.arange(500000, dtype=np.uint32), chunks=(100000,))

    with File(fname, 'r') as f:
        ds = f['data']
        assert isinstance(ds, h5py.Dataset)
        assert ds.shape == (500000,)
        assert ds.chunks == (100000,)
        assert ds.dtype == np.uint32
        assert np.all(ds[:] == np.arange(500000, dtype=np.uint32))


if __name__ == '__main__':
    test_patch()
