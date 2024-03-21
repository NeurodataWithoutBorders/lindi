from typing import Union
import json
import tempfile
import urllib.request
import h5py
import zarr
from zarr.storage import Store as ZarrStore

from .LindiH5pyGroup import LindiH5pyGroup
from .LindiH5pyDataset import LindiH5pyDataset
from .LindiH5pyAttributes import LindiH5pyAttributes
from .LindiH5pyReference import LindiH5pyReference
from .LindiReferenceFileSystemStore import LindiReferenceFileSystemStore


class LindiH5pyFile(h5py.File):
    def __init__(self, _file_object: Union[h5py.File, zarr.Group]):
        """
        Do not use this constructor directly. Instead, use:
        from_reference_file_system, from_zarr_store, from_zarr_group,
        or from_h5py_file
        """
        self._file_object = _file_object
        self._the_group = LindiH5pyGroup(_file_object, self)

    @staticmethod
    def from_reference_file_system(rfs: Union[dict, str]):
        """
        Create a LindiH5pyFile from a reference file system.
        """
        if isinstance(rfs, str):
            if rfs.startswith("http") or rfs.startswith("https"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    filename = f"{tmpdir}/temp.zarr.json"
                    _download_file(rfs, filename)
                    with open(filename, "r") as f:
                        data = json.load(f)
                    assert isinstance(data, dict)  # prevent infinite recursion
                    return LindiH5pyFile.from_reference_file_system(data)
            else:
                with open(rfs, "r") as f:
                    data = json.load(f)
                assert isinstance(data, dict)  # prevent infinite recursion
                return LindiH5pyFile.from_reference_file_system(data)
        elif isinstance(rfs, dict):
            # This store does not need to be closed
            store = LindiReferenceFileSystemStore(rfs)
            return LindiH5pyFile.from_zarr_store(store)
        else:
            raise Exception(f"Unhandled type for rfs: {type(rfs)}")

    @staticmethod
    def from_zarr_store(zarr_store: ZarrStore):
        """
        Create a LindiH5pyFile from a zarr store.
        """
        # note that even though the function is called "open", the zarr_group
        # does not need to be closed
        zarr_group = zarr.open(store=zarr_store, mode="r")
        assert isinstance(zarr_group, zarr.Group)
        return LindiH5pyFile.from_zarr_group(zarr_group)

    @staticmethod
    def from_zarr_group(zarr_group: zarr.Group):
        """
        Create a LindiH5pyFile from a zarr group.
        """
        return LindiH5pyFile(zarr_group)

    @staticmethod
    def from_h5py_file(h5py_file: h5py.File):
        """
        Create a LindiH5pyFile from an h5py file.
        """
        return LindiH5pyFile(h5py_file)

    @property
    def attrs(self):  # type: ignore
        if isinstance(self._file_object, h5py.File):
            attrs_type = 'h5py'
        elif isinstance(self._file_object, zarr.Group):
            attrs_type = 'zarr'
        else:
            raise Exception(f'Unexpected file object type: {type(self._file_object)}')
        return LindiH5pyAttributes(self._file_object.attrs, attrs_type=attrs_type)

    @property
    def filename(self):
        # This is not a string, but this is what h5py seems to do
        if isinstance(self._file_object, h5py.File):
            return self._file_object.filename
        elif isinstance(self._file_object, zarr.Group):
            return ''
        else:
            raise Exception(f"Unhandled type for file object: {type(self._file_object)}")

    @property
    def driver(self):
        raise Exception("Getting driver is not allowed")

    # @property
    # def mode(self):
    #     if isinstance(self._file_object, h5py.File):
    #         return self._file_object.mode
    #     elif isinstance(self._file_object, zarr.Group):
    #         # hard-coded to read-only
    #         return "r"
    #     else:
    #         raise Exception(f"Unhandled type: {type(self._file_object)}")

    @property
    def libver(self):
        raise Exception("Getting libver is not allowed")

    @property
    def userblock_size(self):
        raise Exception("Getting userblock_size is not allowed")

    @property
    def meta_block_size(self):
        raise Exception("Getting meta_block_size is not allowed")

    def swmr_mode(self, value):  # type: ignore
        raise Exception("Getting swmr_mode is not allowed")

    def close(self):
        # Nothing was opened, so nothing needs to be closed
        pass

    def flush(self):
        if isinstance(self._file_object, h5py.File):
            return self._file_object.flush()

    def __enter__(self):  # type: ignore
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f'<LindiH5pyFile "{self._file_object}">'

    # Group methods
    def __getitem__(self, name):  # type: ignore
        return self._get_item(name)

    def _get_item(self, name, getlink=False, default=None):
        if isinstance(name, LindiH5pyReference) and isinstance(self._file_object, zarr.Group):
            if getlink:
                raise Exception("Getting link is not allowed for references")
            zarr_group = self._file_object
            if name._source != '.':
                raise Exception(f'For now, source of reference must be ".", got "{name._source}"')
            if name._source_object_id is not None:
                if name._source_object_id != zarr_group.attrs.get("object_id"):
                    raise Exception(f'Mismatch in source object_id: "{name._source_object_id}" and "{zarr_group.attrs.get("object_id")}"')
            target = self[name._path]
            if name._object_id is not None:
                if name._object_id != target.attrs.get("object_id"):
                    raise Exception(f'Mismatch in object_id: "{name._object_id}" and "{target.attrs.get("object_id")}"')
            return target
        elif isinstance(name, h5py.Reference) and isinstance(self._file_object, h5py.File):
            if getlink:
                raise Exception("Getting link is not allowed for references")
            x = self._file_object[name]
            if isinstance(x, h5py.Group):
                return LindiH5pyGroup(x, self)
            elif isinstance(x, h5py.Dataset):
                return LindiH5pyDataset(x, self)
            else:
                raise Exception(f"Unexpected type for resolved reference at path {name}: {type(x)}")
        # if it contains slashes, it's a path
        if isinstance(name, str) and "/" in name:
            parts = name.split("/")
            x = self._the_group
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    assert isinstance(x, LindiH5pyGroup)
                    x = x.get(part, default=default, getlink=getlink)
                else:
                    assert isinstance(x, LindiH5pyGroup)
                    x = x.get(part)
            return x
        return self._the_group.get(name, default=default, getlink=getlink)

    def get(self, name, default=None, getclass=False, getlink=False):
        if getclass:
            raise Exception("Getting class is not allowed")
        return self._get_item(name, getlink=getlink, default=default)

    def __iter__(self):
        return self._the_group.__iter__()

    def __reversed__(self):
        return self._the_group.__reversed__()

    def __contains__(self, name):
        return self._the_group.__contains__(name)

    @property
    def id(self):
        return self._the_group.id

    @property
    def file(self):
        return self._the_group.file

    @property
    def name(self):
        return self._the_group.name


def _download_file(url: str, filename: str) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(filename, "wb") as f:
            f.write(response.read())
