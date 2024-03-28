from typing import Union, Literal
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
    def __init__(self, _file_object: Union[h5py.File, zarr.Group], *, _zarr_store: Union[ZarrStore, None] = None, _mode: Literal["r", "r+"] = "r"):
        """
        Do not use this constructor directly. Instead, use:
        from_reference_file_system, from_zarr_store, from_zarr_group,
        or from_h5py_file
        """
        self._file_object = _file_object
        self._zarr_store = _zarr_store
        self._mode: Literal['r', 'r+'] = _mode
        self._the_group = LindiH5pyGroup(_file_object, self)

    @staticmethod
    def from_reference_file_system(rfs: Union[dict, str], mode: Literal["r", "r+"] = "r"):
        """
        Create a LindiH5pyFile from a reference file system.

        Parameters
        ----------
        rfs : Union[dict, str]
            The reference file system. This can be a dictionary or a URL or path
            to a .zarr.json file.
        mode : Literal["r", "r+"], optional
            The mode to open the file object in, by default "r" If the mode is
            "r", the file object will be read-only. If the mode is "r+", the
            file will be read-write. However, if the rfs is a string (URL or
            path), the file itself will not be modified on changes, but the
            internal in-memory representation will be modified. Use
            to_reference_file_system() to export the updated reference file
            system.
        """
        if isinstance(rfs, str):
            if rfs.startswith("http") or rfs.startswith("https"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    filename = f"{tmpdir}/temp.zarr.json"
                    _download_file(rfs, filename)
                    with open(filename, "r") as f:
                        data = json.load(f)
                    assert isinstance(data, dict)  # prevent infinite recursion
                    return LindiH5pyFile.from_reference_file_system(data, mode=mode)
            else:
                with open(rfs, "r") as f:
                    data = json.load(f)
                assert isinstance(data, dict)  # prevent infinite recursion
                return LindiH5pyFile.from_reference_file_system(data, mode=mode)
        elif isinstance(rfs, dict):
            # This store does not need to be closed
            store = LindiReferenceFileSystemStore(rfs)
            return LindiH5pyFile.from_zarr_store(store, mode=mode)
        else:
            raise Exception(f"Unhandled type for rfs: {type(rfs)}")

    @staticmethod
    def from_zarr_store(zarr_store: ZarrStore, mode: Literal["r", "r+"] = "r"):
        """
        Create a LindiH5pyFile from a zarr store.

        Parameters
        ----------
        zarr_store : ZarrStore
            The zarr store.
        mode : Literal["r", "r+"], optional
            The mode to open the file object in, by default "r". If the mode is
            "r", the file object will be read-only. For write mode to work, the
            zarr store will need to be writeable as well.
        """
        # note that even though the function is called "open", the zarr_group
        # does not need to be closed
        zarr_group = zarr.open(store=zarr_store, mode=mode)
        assert isinstance(zarr_group, zarr.Group)
        return LindiH5pyFile.from_zarr_group(zarr_group, _zarr_store=zarr_store, mode=mode)

    @staticmethod
    def from_zarr_group(zarr_group: zarr.Group, *, mode: Literal["r", "r+"] = "r", _zarr_store: Union[ZarrStore, None] = None):
        """
        Create a LindiH5pyFile from a zarr group.

        Parameters
        ----------
        zarr_group : zarr.Group
            The zarr group.
        mode : Literal["r", "r+"], optional
            The mode to open the file object in, by default "r". If the mode is
            "r", the file object will be read-only. For write mode to work, the
            zarr store will need to be writeable as well.
        _zarr_store : Union[ZarrStore, None], optional
            The zarr store, internally set for use with
            to_reference_file_system().

        See from_zarr_store().
        """
        return LindiH5pyFile(zarr_group, _zarr_store=_zarr_store, _mode=mode)

    @staticmethod
    def from_h5py_file(h5py_file: h5py.File):
        """
        Create a LindiH5pyFile from an h5py file.

        This is used mainly for testing and may be removed in the future.

        Parameters
        ----------
        h5py_file : h5py.File
            The h5py file.
        """
        return LindiH5pyFile(h5py_file)

    def to_reference_file_system(self):
        """
        Export the internal in-memory representation to a reference file system.
        In order to use this, the file object needs to have been created using
        from_reference_file_system().
        """
        if self._zarr_store is None:
            raise Exception("Cannot convert to reference file system without zarr store")
        if not isinstance(self._zarr_store, LindiReferenceFileSystemStore):
            raise Exception(f"Unexpected type for zarr store: {type(self._zarr_store)}")
        rfs = self._zarr_store.rfs
        rfs_copy = json.loads(json.dumps(rfs))
        return rfs_copy

    @property
    def attrs(self):  # type: ignore
        if isinstance(self._file_object, h5py.File):
            attrs_type = 'h5py'
        elif isinstance(self._file_object, zarr.Group):
            attrs_type = 'zarr'
        else:
            raise Exception(f'Unexpected file object type: {type(self._file_object)}')
        return LindiH5pyAttributes(self._file_object.attrs, attrs_type=attrs_type, readonly=self.mode == "r")

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

    @property
    def mode(self):
        return self._mode

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

    def __str__(self):
        return f'<LindiH5pyFile "{self._file_object}">'

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

    @property
    def ref(self):
        return self._the_group.ref

    ##############################
    # write
    def create_group(self, name, track_order=None):
        if self._mode not in ['r+']:
            raise Exception("Cannot create group in read-only mode")
        if track_order is not None:
            raise Exception("track_order is not supported (I don't know what it is)")
        return self._the_group.create_group(name)

    def require_group(self, name):
        if self._mode not in ['r+']:
            raise Exception("Cannot require group in read-only mode")
        return self._the_group.require_group(name)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if self._mode not in ['r+']:
            raise Exception("Cannot create dataset in read-only mode")
        return self._the_group.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)


def _download_file(url: str, filename: str) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(filename, "wb") as f:
            f.write(response.read())
