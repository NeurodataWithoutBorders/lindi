from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..LindiH5pyAttributes import LindiH5pyAttributes  # pragma: no cover


class LindiH5pyAttributesWrite:
    def __init__(self, p: 'LindiH5pyAttributes'):
        self.p = p

    def __setitem__(self, key, value):
        if self.p._readonly:
            raise KeyError("Cannot set attributes on read-only object")
        if self.p._attrs_type == "h5py":
            self.p._attrs[key] = value
        elif self.p._attrs_type == "zarr":
            from ...conversion.attr_conversion import h5_to_zarr_attr  # avoid circular import
            self.p._attrs[key] = h5_to_zarr_attr(value, h5f=None)
        else:
            raise ValueError(f"Unknown attrs_type: {self.p._attrs_type}")
