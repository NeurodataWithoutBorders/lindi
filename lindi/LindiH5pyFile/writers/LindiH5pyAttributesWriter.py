from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..LindiH5pyAttributes import LindiH5pyAttributes  # pragma: no cover


class LindiH5pyAttributesWriter:
    def __init__(self, p: 'LindiH5pyAttributes'):
        self.p = p

    def __setitem__(self, key, value):
        from ...conversion.attr_conversion import h5_to_zarr_attr  # avoid circular import
        if self.p._readonly:
            raise KeyError("Cannot set attributes on read-only object")
        self.p._attrs[key] = h5_to_zarr_attr(value, h5f=None)
