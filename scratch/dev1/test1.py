import zarr
import h5py
import remfile
from LindiH5Store import LindiH5Store


# This one seems to load properly
# https://neurosift.app/?p=/nwb&dandisetId=000717&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/
h5_url = 'https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/'
json_url = 'https://kerchunk.neurosift.org/dandi/dandisets/000717/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/zarr.json'


def main():
    remf = remfile.File(h5_url)
    h5f = h5py.File(remf, 'r')
    f = LindiH5Store(remf)
    root = zarr.open(f)
    assert isinstance(root, zarr.Group)

    def handle_group(g: zarr.Group):
        _compare_groups(g, h5f['/' + g.name])
        for k, item in g.items():
            if isinstance(item, zarr.Group):
                handle_group(item)
            elif isinstance(item, zarr.Array):
                handle_array(item)

    def handle_array(a: zarr.Array):
        _compare_arrays(a, h5f[a.name])

    handle_group(root)


def _compare_groups(g1: zarr.Group, g2: h5py.Group):
    print(f'__________ {g1.name} (GROUP)')
    assert g1.name == g2.name
    for k, v in g1.attrs.items():
        if k not in g2.attrs:
            print(f'WARNING: Attribute {k} not found in h5 group {g2.name}')
        elif g2.attrs[k] != v:
            print(f'WARNING: Attribute {k} value mismatch in h5 group {g2.name}')
    for k, v in g2.attrs.items():
        if k not in g1.attrs:
            print(f'WARNING: Attribute {k} not found in zarr group {g1.name}')
        elif g1.attrs[k] != v:
            print(f'WARNING: Attribute {k} value mismatch in zarr group {g1.name}')


def _compare_arrays(a1: zarr.Array, a2: h5py.Dataset):
    print(f'__________ {a1.name} (ARRAY)')
    if a1.dtype != a2.dtype:
        print(f'WARNING: dtype mismatch for {a1.name}: {a1.dtype} != {a2.dtype}')
    if a1.shape != a2.shape:
        print(f'WARNING: shape mismatch for {a1.name}: {a1.shape} != {a2.shape}')
    if a1.ndim == 0:
        if a1[()] != a2[()]:
            print(f'WARNING: value mismatch for {a1.name}: {a1[()]} != {a2[()]}')


if __name__ == '__main__':
    main()
