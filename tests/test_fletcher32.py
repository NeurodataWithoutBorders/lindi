import tempfile
import h5py
import lindi
import numpy as np


def test_fletcher32():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f'{tmpdir}/test.h5'
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset('dset', shape=(100,), dtype='i4', fletcher32=True)
            dset[...] = range(100)
            assert dset.fletcher32
        store = lindi.LindiH5ZarrStore.from_file(filename, url=filename)
        rfs = store.to_reference_file_system()
        client = lindi.LindiH5pyFile.from_reference_file_system(rfs)
        ds0 = client['dset']
        assert isinstance(ds0, h5py.Dataset)
        assert ds0.fletcher32
        data = ds0[...]
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.dtype('int32')
        assert np.all(data == np.arange(100))


if __name__ == '__main__':
    test_fletcher32()
