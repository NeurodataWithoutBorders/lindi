import tempfile
import numpy as np
import zarr
import lindi


def test_zarr_write():
    return
    with tempfile.TemporaryDirectory() as tmpdirname:
        # dirname = f'{tmpdirname}/test.zarr'
        dirname = 'test.zarr'
        store = zarr.DirectoryStore(dirname)
        root = zarr.group(store=store)
        ff = lindi.LindiH5pyFile.from_zarr_group(root)
        ds1 = ff.create_dataset('ds1', data=[1, 2, 3])
        ds1.attrs['attr'] = 'value'
        ds2 = ff.create_dataset('ds2', data=np.array([4, 5, 6], dtype=np.uint16))
        ds3 = ff.create_dataset('ds3', data=12.5)

        store2 = zarr.DirectoryStore(dirname)
        root2 = zarr.group(store=store2)
        ff2 = lindi.LindiH5pyFile.from_zarr_group(root2)

        ds = ff2['ds1']
        assert isinstance(ds, lindi.LindiH5pyDataset)
        assert ds.attrs['attr'] == 'value'
        assert ds.shape == (3,)
        assert ds.dtype == np.dtype('int64')
        assert ds[0] == 1
        assert ds[1] == 2
        assert ds[2] == 3

        ds = ff2['ds2']
        assert isinstance(ds, lindi.LindiH5pyDataset)
        assert ds.shape == (3,)
        assert ds.dtype == np.dtype('uint16')
        assert ds[0] == 4
        assert ds[1] == 5
        assert ds[2] == 6

        ds = ff2['ds3']
        assert isinstance(ds, lindi.LindiH5pyDataset)
        assert ds.shape == ()
        assert ds.dtype == np.dtype('float64')
        assert ds[()] == 12.5


if __name__ == '__main__':
    test_zarr_write()

