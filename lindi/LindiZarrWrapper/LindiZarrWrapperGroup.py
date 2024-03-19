import zarr
from .LindiZarrWrapperAttributes import LindiZarrWrapperAttributes
from .LindiZarrWrapperDataset import LindiZarrWrapperDataset


class LindiZarrWrapperGroup:
    def __init__(self, *, _zarr_group: zarr.Group, _client):
        self._zarr_group = _zarr_group
        self._client = _client

    @property
    def file(self):
        return self._client

    @property
    def id(self):
        return None

    @property
    def attrs(self):
        """Attributes attached to this object"""
        return LindiZarrWrapperAttributes(_object=self._zarr_group)

    def keys(self):
        return self._zarr_group.keys()

    @property
    def name(self):
        return self._zarr_group.name

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise Exception(
                f'Cannot use key "{key}" of type "{type(key)}" to index into a LindiZarrWrapperGroup, at path "{self._zarr_group.name}"'
            )
        if key in self._zarr_group.keys():
            x = self._zarr_group[key]
            if isinstance(x, zarr.Group):
                return LindiZarrWrapperGroup(_zarr_group=x, _client=self._client)
            elif isinstance(x, zarr.Array):
                return LindiZarrWrapperDataset(_zarr_array=x, _client=self._client)
            else:
                raise Exception(f"Unknown type: {type(x)}")
        else:
            raise KeyError(f'Key "{key}" not found in group "{self._zarr_group.name}"')

    def __iter__(self):
        for k in self.keys():
            yield k
