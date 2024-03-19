from typing import Union
import json
import tempfile
from typing import Literal
from fsspec import FSMap
import zarr
import urllib.request
from fsspec.implementations.reference import ReferenceFileSystem
from zarr.storage import Store
from .LindiGroup import LindiGroup
from .LindiReference import LindiReference


class LindiClient(LindiGroup):
    def __init__(
        self,
        *,
        _zarr_group: zarr.Group,
    ) -> None:
        self._zarr_group = _zarr_group
        super().__init__(_zarr_group=self._zarr_group, _client=self)

    @property
    def filename(self):
        return ''

    @staticmethod
    def from_zarr_store(zarr_store: Union[Store, FSMap]) -> "LindiClient":
        zarr_group = zarr.open(store=zarr_store, mode="r")
        assert isinstance(zarr_group, zarr.Group)
        return LindiClient.from_zarr_group(zarr_group)

    @staticmethod
    def from_file(
        json_file: str, file_type: Literal["zarr.json"] = "zarr.json"
    ) -> "LindiClient":
        if file_type == "zarr.json":
            if json_file.startswith("http") or json_file.startswith("https"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    filename = f"{tmpdir}/temp.zarr.json"
                    _download_file(json_file, filename)
                    with open(filename, "r") as f:
                        data = json.load(f)
                    return LindiClient.from_reference_file_system(data)
            else:
                with open(json_file, "r") as f:
                    data = json.load(f)
                return LindiClient.from_reference_file_system(data)
        else:
            raise ValueError(f"Unknown file_type: {file_type}")

    @staticmethod
    def from_zarr_group(zarr_group: zarr.Group) -> "LindiClient":
        return LindiClient(_zarr_group=zarr_group)

    @staticmethod
    def from_reference_file_system(data: dict) -> "LindiClient":
        fs = ReferenceFileSystem(data).get_mapper(root="")
        return LindiClient.from_zarr_store(fs)

    def get(self, key, default=None, getlink: bool = False):
        try:
            ret = self[key]
        except KeyError:
            ret = default
        if getlink:
            return ret
        else:
            if isinstance(ret, LindiReference):
                return self[ret]
            else:
                return ret

    def __getitem__(self, key):  # type: ignore
        if isinstance(key, str):
            if key.startswith('/'):
                key = key[1:]
            parts = key.split("/")
            if len(parts) == 1:
                return super().__getitem__(key)
            else:
                g = self
                for part in parts:
                    g = g[part]
                return g
        elif isinstance(key, LindiReference):
            if key._source != '.':
                raise Exception(f'For now, source of reference must be ".", got "{key._source}"')
            if key._source_object_id is not None:
                if key._source_object_id != self._zarr_group.attrs.get("object_id"):
                    raise Exception(f'Mismatch in source object_id: "{key._source_object_id}" and "{self._zarr_group.attrs.get("object_id")}"')
            target = self[key._path]
            if key._object_id is not None:
                if key._object_id != target.attrs.get("object_id"):
                    raise Exception(f'Mismatch in object_id: "{key._object_id}" and "{target.attrs.get("object_id")}"')
            return target
        else:
            raise Exception(f'Cannot use key "{key}" of type "{type(key)}" to index into a LindiClient')


def _download_file(url: str, filename: str) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(filename, "wb") as f:
            f.write(response.read())
