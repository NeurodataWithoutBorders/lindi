from .LindiH5pyReference import LindiH5pyReference
from ..conversion.attr_conversion import zarr_to_h5_attr
from ..conversion.nan_inf_ninf import decode_nan_inf_ninf
from .writers.LindiH5pyAttributesWriter import LindiH5pyAttributesWriter

_special_attribute_keys = [
    "_SCALAR",
    "_COMPOUND_DTYPE",
    "_REFERENCE",
    "_EXTERNAL_ARRAY_LINK",
    "_SOFT_LINK",
]


class LindiH5pyAttributes:
    def __init__(self, attrs, readonly: bool):
        self._attrs = attrs
        self._readonly = readonly

        if self._readonly:
            self._writer = None
        else:
            self._writer = LindiH5pyAttributesWriter(self)

    def get(self, key, default=None):
        try:
            if key in _special_attribute_keys:
                raise KeyError
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        if key in _special_attribute_keys:
            return False
        return key in self._attrs

    def __getitem__(self, key):
        val = self._attrs[key]
        if isinstance(val, dict) and "_REFERENCE" in val:
            return LindiH5pyReference(val["_REFERENCE"])

        # Convert special float values to actual floats (NaN, Inf, -Inf)
        # Note that string versions of these values are not supported
        val = decode_nan_inf_ninf(val)

        return zarr_to_h5_attr(val)

    def __setitem__(self, key, value):
        if self._readonly:
            raise ValueError("Cannot set items on read-only object")
        assert self._writer is not None
        self._writer.__setitem__(key, value)

    def __delitem__(self, key):
        raise KeyError("Cannot delete attributes on read-only object")

    def __iter__(self):
        # Do not return special zarr attributes during iteration
        for k in self._attrs:
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
        return repr(self._attrs)

    def __str__(self):
        return str(self._attrs)

    def keys(self):
        return list(self)
