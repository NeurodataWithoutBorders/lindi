import tempfile
import numpy as np
import h5py
import lindi


def test_lindi_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f'{tmpdir}/test.lindi.json'
        with lindi.File(fname, 'w') as f:
            f.create_dataset('data', data=np.arange(500000, dtype=np.uint32), chunks=(100000,))

        with lindi.File(fname, 'r') as f:
            ds = f['data']
            assert isinstance(ds, h5py.Dataset)
            assert ds.shape == (500000,)
            assert ds.chunks == (100000,)
            assert ds.dtype == np.uint32
            assert np.all(ds[:] == np.arange(500000, dtype=np.uint32))


if __name__ == '__main__':
    test_lindi_file()
