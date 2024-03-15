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


class LindiClient(LindiGroup):
    def __init__(
        self,
        *,
        _zarr_group: zarr.Group,
    ) -> None:
        self._zarr_group = _zarr_group
        super().__init__(_zarr_group=self._zarr_group)

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
        fs = ReferenceFileSystem(data).get_mapper(root="/")
        return LindiClient.from_zarr_store(fs)


def _download_file(url: str, filename: str) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(filename, "wb") as f:
            f.write(response.read())
