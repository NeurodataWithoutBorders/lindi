from typing import Literal
from .LindiH5pyReference import LindiH5pyReference

_special_attribute_keys = [
    "_SCALAR",
    "_COMPOUND_DTYPE",
    "_REFERENCE",
    "_EXTERNAL_ARRAY_LINK",
    "_SOFT_LINK",
]


class LindiH5pyAttributes:
    def __init__(self, attrs, attrs_type: Literal["h5py", "zarr"]):
        self._attrs = attrs
        self._attrs_type = attrs_type

    def get(self, key, default=None):
        if self._attrs_type == "h5py":
            return self._attrs.get(key, default)
        elif self._attrs_type == "zarr":
            try:
                if key in _special_attribute_keys:
                    raise KeyError
                return self[key]
            except KeyError:
                return default
        else:
            raise ValueError(f"Unknown attrs_type: {self._attrs_type}")

    def __contains__(self, key):
        if self._attrs_type == "h5py":
            return key in self._attrs
        elif self._attrs_type == "zarr":
            return key in self._attrs
        else:
            raise ValueError(f"Unknown attrs_type: {self._attrs_type}")

    def __getitem__(self, key):
        val = self._attrs[key]
        if self._attrs_type == "h5py":
            return val
        elif self._attrs_type == "zarr":
            if isinstance(val, dict) and "_REFERENCE" in val:
                return LindiH5pyReference(val["_REFERENCE"])

            # Convert special values
            # @rly: doing this saves a lot of headache when loading the nwb file
            # but then how can we represent strings that should not be converted?
            elif val == 'NaN':
                return float('nan')
            elif val == 'Infinity':
                return float('inf')
            elif val == '-Infinity':
                return float('-inf')

            else:
                return val
        else:
            raise ValueError(f"Unknown attrs_type: {self._attrs_type}")

    def __setitem__(self, key, value):
        raise KeyError("Cannot set attributes on read-only object")

    def __delitem__(self, key):
        raise KeyError("Cannot delete attributes on read-only object")

    def __iter__(self):
        if self._attrs_type == "h5py":
            return self._attrs.__iter__()
        elif self._attrs_type == "zarr":
            for k in self._attrs:
                if k not in _special_attribute_keys:
                    yield k
        else:
            raise ValueError(f"Unknown attrs_type: {self._attrs_type}")

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
