import json
import h5py
import tempfile
import zarr
import kerchunk.hdf  # type: ignore
from LindiH5Store import LindiH5Store
from fsspec.implementations.reference import ReferenceFileSystem


def test_scalar_dataset():
    for val in ['abc', b'abc', 1, 3.6]:
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f'{tmpdir}/test.h5'
            with h5py.File(filename, 'w') as f:
                f.create_dataset('X', data=val)
            with h5py.File(filename, 'r') as f:
                h5chunks = kerchunk.hdf.SingleHdf5ToZarr(
                    f,
                    url=filename,
                    hdmf_mode=True,
                    num_chunks_per_dataset_threshold=1000,
                    max_num_items=1000
                )
                a = h5chunks.translate()
                zattrs = json.loads(a['refs']['X/.zattrs'])
                zarray = json.loads(a['refs']['X/.zarray'])
                zdata = a['refs']['X/0']
                fs = ReferenceFileSystem(a)
                store0 = fs.get_mapper(root='/', check=False)
                root = zarr.open(store0)
                val1 = root['X'][()]
            with open(filename, 'rb') as f:
                L = LindiH5Store(f)
                # root = zarr.open(L)
                zattrs = json.loads(L['X/.zattrs'].decode('utf-8'))  # type: ignore
                zarray = json.loads(L['X/.zarray'].decode('utf-8'))  # type: ignore
                zdata = L['X/0']
                root = zarr.open(L)
                val2 = root['X'][()]
            print(val1, val2, val)
            print(type(val1[0]))
            print(type(val2[0]))
            print(type(val))
            print('')


if __name__ == '__main__':
    test_scalar_dataset()
