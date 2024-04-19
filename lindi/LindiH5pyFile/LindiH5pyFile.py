from typing import Union, Literal
import json
import tempfile
import urllib.request
import h5py
import zarr
from zarr.storage import Store as ZarrStore

from .LindiH5pyGroup import LindiH5pyGroup
from .LindiH5pyAttributes import LindiH5pyAttributes
from .LindiH5pyReference import LindiH5pyReference
from .LindiReferenceFileSystemStore import LindiReferenceFileSystemStore

from ..LindiStagingStore.StagingArea import StagingArea
from ..LindiStagingStore.LindiStagingStore import LindiStagingStore


class LindiH5pyFile(h5py.File):
    def __init__(self, _zarr_group: zarr.Group, *, _zarr_store: Union[ZarrStore, None] = None, _mode: Literal["r", "r+"] = "r"):
        """
        Do not use this constructor directly. Instead, use:
        from_reference_file_system, from_zarr_store, from_zarr_group,
        or from_h5py_file
        """
        self._zarr_group = _zarr_group
        self._zarr_store = _zarr_store
        self._mode: Literal['r', 'r+'] = _mode
        self._the_group = LindiH5pyGroup(_zarr_group, self)

        # see comment in LindiH5pyGroup
        self._id = f'{id(self._zarr_group)}/'

    @staticmethod
    def from_reference_file_system(rfs: Union[dict, str], mode: Literal["r", "r+"] = "r", staging_area: Union[StagingArea, None] = None):
        """
        Create a LindiH5pyFile from a reference file system.

        Parameters
        ----------
        rfs : Union[dict, str]
            The reference file system. This can be a dictionary or a URL or path
            to a .lindi.json file.
        mode : Literal["r", "r+"], optional
            The mode to open the file object in, by default "r". If the mode is
            "r", the file object will be read-only. If the mode is "r+", the
            file will be read-write. However, if the rfs is a string (URL or
            path), the file itself will not be modified on changes, but the
            internal in-memory representation will be modified. Use
            to_reference_file_system() to export the updated reference file
            system to the same file or a new file.
        """
        if staging_area is not None:
            if mode not in ['r+']:
                raise Exception("Staging area cannot be used in read-only mode")

        if isinstance(rfs, str):
            if rfs.startswith("http") or rfs.startswith("https"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    filename = f"{tmpdir}/temp.lindi.json"
                    _download_file(rfs, filename)
                    with open(filename, "r") as f:
                        data = json.load(f)
                    assert isinstance(data, dict)  # prevent infinite recursion
                    return LindiH5pyFile.from_reference_file_system(data, mode=mode, staging_area=staging_area)
            else:
                with open(rfs, "r") as f:
                    data = json.load(f)
                assert isinstance(data, dict)  # prevent infinite recursion
                return LindiH5pyFile.from_reference_file_system(data, mode=mode, staging_area=staging_area)
        elif isinstance(rfs, dict):
            # This store does not need to be closed
            store = LindiReferenceFileSystemStore(rfs)
            if staging_area:
                store = LindiStagingStore(base_store=store, staging_area=staging_area)
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

    def to_reference_file_system(self):
        """
        Export the internal in-memory representation to a reference file system.
        In order to use this, the file object needs to have been created using
        from_reference_file_system().
        """
        if self._zarr_store is None:
            raise Exception("Cannot convert to reference file system without zarr store")
        zarr_store = self._zarr_store
        if isinstance(zarr_store, LindiStagingStore):
            zarr_store = zarr_store._base_store
        if not isinstance(zarr_store, LindiReferenceFileSystemStore):
            raise Exception(f"Unexpected type for zarr store: {type(self._zarr_store)}")
        rfs = zarr_store.rfs
        rfs_copy = json.loads(json.dumps(rfs))
        LindiReferenceFileSystemStore.replace_meta_file_contents_with_dicts_in_rfs(rfs_copy)
        LindiReferenceFileSystemStore.use_templates_in_rfs(rfs_copy)
        return rfs_copy

    @property
    def attrs(self):  # type: ignore
        return LindiH5pyAttributes(self._zarr_group.attrs, readonly=self.mode == "r")

    @property
    def filename(self):
        return ''

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
        pass

    def __enter__(self):  # type: ignore
        return self

    def __exit__(self, *args):
        self.close()

    def __str__(self):
        return f'<LindiH5pyFile "{self._zarr_group}">'

    def __repr__(self):
        return f'<LindiH5pyFile "{self._zarr_group}">'

    def __bool__(self):
        # This is called when checking if the file is open
        return True

    def __hash__(self):
        # This is called for example when using a file as a key in a dictionary
        return id(self)

    def copy(self, source, dest, name=None,
             shallow=False, expand_soft=False, expand_external=False,
             expand_refs=False, without_attrs=False):
        if shallow:
            raise Exception("shallow is not implemented for copy")
        if expand_soft:
            raise Exception("expand_soft is not implemented for copy")
        if expand_external:
            raise Exception("expand_external is not implemented for copy")
        if expand_refs:
            raise Exception("expand_refs is not implemented for copy")
        if without_attrs:
            raise Exception("without_attrs is not implemented for copy")
        if name is None:
            raise Exception("name must be provided for copy")
        src_item = self._get_item(source)
        if not isinstance(src_item, (h5py.Group, h5py.Dataset)):
            raise Exception(f"Unexpected type for source in copy: {type(src_item)}")
        _recursive_copy(src_item, dest, name=name)

    def __delitem__(self, name):
        parent_key = '/'.join(name.split('/')[:-1])
        grp = self[parent_key]
        assert isinstance(grp, LindiH5pyGroup)
        del grp[name.split('/')[-1]]

    # Group methods
    def __getitem__(self, name):  # type: ignore
        return self._get_item(name)

    def _get_item(self, name, getlink=False, default=None):
        if isinstance(name, LindiH5pyReference):
            if getlink:
                raise Exception("Getting link is not allowed for references")
            zarr_group = self._zarr_group
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
        # see comment in LindiH5pyGroup
        return self._id

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

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        if self._mode not in ['r+']:
            raise Exception("Cannot require dataset in read-only mode")
        return self._the_group.require_dataset(name, shape, dtype, exact=exact, **kwds)

    ##############################
    # staging store
    @property
    def staging_store(self):
        store = self._zarr_store
        if not isinstance(store, LindiStagingStore):
            return None
        return store


def _download_file(url: str, filename: str) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(filename, "wb") as f:
            f.write(response.read())


def _recursive_copy(src_item: Union[h5py.Group, h5py.Dataset], dest: h5py.File, name: str) -> None:
    if isinstance(src_item, h5py.Group):
        dst_item = dest.create_group(name)
        for k, v in src_item.attrs.items():
            dst_item.attrs[k] = v
        for k, v in src_item.items():
            _recursive_copy(v, dest, name=f'{name}/{k}')
    elif isinstance(src_item, h5py.Dataset):
        # Let's specially handle the case where the source and dest files
        # are LindiH5pyFiles with reference file systems as the internal
        # representation. In this case, we don't need to copy the actual
        # data because we can copy the reference.
        if isinstance(src_item.file, LindiH5pyFile) and isinstance(dest, LindiH5pyFile):
            if src_item.name is None:
                raise Exception("src_item.name is None")
            src_item_name = _without_initial_slash(src_item.name)
            src_zarr_store = src_item.file._zarr_store
            dst_zarr_store = dest._zarr_store
            if src_zarr_store is not None and dst_zarr_store is not None:
                if isinstance(src_zarr_store, LindiReferenceFileSystemStore) and isinstance(dst_zarr_store, LindiReferenceFileSystemStore):
                    src_rfs = src_zarr_store.rfs
                    dst_rfs = dst_zarr_store.rfs
                    src_ref_keys = list(src_rfs['refs'].keys())
                    for src_ref_key in src_ref_keys:
                        if src_ref_key.startswith(f'{src_item_name}/'):
                            dst_ref_key = f'{name}/{src_ref_key[len(src_item_name) + 1:]}'
                            # Even though it's not expected to be a problem, we
                            # do a deep copy here because a problem resulting
                            # from one rfs being modified affecting another
                            # would be very difficult to debug.
                            dst_rfs['refs'][dst_ref_key] = _deep_copy(src_rfs['refs'][src_ref_key])
                    return

        dst_item = dest.create_dataset(name, data=src_item[()], chunks=src_item.chunks)
        for k, v in src_item.attrs.items():
            dst_item.attrs[k] = v
    else:
        raise Exception(f"Unexpected type for src_item in _recursive_copy: {type(src_item)}")


def _without_initial_slash(s: str) -> str:
    if s.startswith('/'):
        return s[1:]
    return s


def _deep_copy(obj):
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_copy(v) for v in obj)
    else:
        return obj
