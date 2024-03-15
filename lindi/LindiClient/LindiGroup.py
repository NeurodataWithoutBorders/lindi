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

    @property
    def name(self):
        return self._zarr_group.name

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        if isinstance(key, dict):
            # might be a reference
            if "_REFERENCE" in key:
                return LindiReference(key['_REFERENCE'])
        if not isinstance(key, str):
            raise Exception(f'Cannot use key "{key}" of type "{type(key)}" to index into a LindiGroup, at path "{self._zarr_group.name}"')
        if key in self._zarr_group.keys():
            x = self._zarr_group[key]
            if isinstance(x, zarr.Group):
                return LindiGroup(_zarr_group=x)
            elif isinstance(x, zarr.Array):
                return LindiDataset(_zarr_array=x)
            else:
                raise Exception(f"Unknown type: {type(x)}")
        else:
            raise KeyError(f'Key "{key}" not found in group "{self._zarr_group.name}"')

    def __iter__(self):
        for k in self.keys():
            yield k


class LindiReference:
    def __init__(self, obj: dict):
        self._object_id = obj["object_id"]
        self._path = obj["path"]
        self._source = obj["source"]
        self._source_object_id = obj["source_object_id"]

    @property
    def name(self):
        return self._path

    def __repr__(self):
        return f"LindiReference({self._source}, {self._path})"

    def __str__(self):
        return f"LindiReference({self._source}, {self._path})"
