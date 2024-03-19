from typing import Union
import zarr
from .LindiZarrWrapperReference import LindiZarrWrapperReference


_special_attribute_keys = ["_SCALAR", "_COMPOUND_DTYPE", "_REFERENCE", "_EXTERNAL_ARRAY_LINK", "_SOFT_LINK"]


class LindiZarrWrapperAttributes:
    def __init__(self, *, _object: Union[zarr.Group, zarr.Array]):
        self._object = _object

    def get(self, key, default=None):
        try:
            if key in _special_attribute_keys:
                raise KeyError
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
        for k in self._object.attrs:
            if k not in _special_attribute_keys:
                yield k

    def items(self):
        for k in self:
            yield k, self[k]

    def __len__(self):
        ct = 0
        for _ in self:
            ct += 1
        return ct

    def __repr__(self):
        return repr(self._object.attrs)

    def __str__(self):
        return str(self._object.attrs)
