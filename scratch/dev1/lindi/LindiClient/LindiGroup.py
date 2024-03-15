import zarr
from .LindiAttributes import LindiAttributes
from .LindiDataset import LindiDataset


class LindiGroup:
    def __init__(self, *, _zarr_group: zarr.Group):
        self._zarr_group = _zarr_group

    @property
    def attrs(self):
        """Attributes attached to this object"""
        return LindiAttributes(_object=self._zarr_group)

    def keys(self):
        return self._zarr_group.keys()

    def __getitem__(self, key):
        if key in self._zarr_group.keys():
            x = self._zarr_group[key]
            if isinstance(x, zarr.Group):
                return LindiGroup(_zarr_group=x)
            elif isinstance(x, zarr.Array):
                return LindiDataset(_zarr_array=x)
            else:
                raise Exception(f"Unknown type: {type(x)}")
        else:
            raise KeyError(key)
