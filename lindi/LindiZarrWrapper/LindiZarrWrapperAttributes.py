from typing import Union
import zarr
from .LindiZarrWrapperReference import LindiZarrWrapperReference


class LindiZarrWrapperAttributes:
    def __init__(self, *, _object: Union[zarr.Group, zarr.Array]):
        self._object = _object

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        val = self._object.attrs[key]
        if isinstance(val, dict) and "_REFERENCE" in val:
            return LindiZarrWrapperReference(val["_REFERENCE"])
        return self._object.attrs[key]

    def __setitem__(self, key, value):
        raise KeyError("Cannot set attributes on read-only object")

    def __delitem__(self, key):
        raise KeyError("Cannot delete attributes on read-only object")

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
