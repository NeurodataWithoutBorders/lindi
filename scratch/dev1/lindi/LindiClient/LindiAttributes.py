from typing import Union
import zarr


class LindiAttributes:
    def __init__(self, *, _object: Union[zarr.Group, zarr.Array]):
        self._object = _object

    def __getitem__(self, key):
        return self._object.attrs[key]

    def __setitem__(self, key, value):
        raise Exception("Read-only")

    def __delitem__(self, key):
        raise Exception("Read-only")

    def __iter__(self):
        return iter(self._object.attrs)

    def items(self):
        return self._object.attrs.items()

    def __len__(self):
        return len(self._object.attrs)

    def __repr__(self):
        return repr(self._object.attrs)

    def __str__(self):
        return str(self._object.attrs)
