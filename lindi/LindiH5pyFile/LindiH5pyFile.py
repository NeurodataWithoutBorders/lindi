from typing import Union, Literal, Callable
import os
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
from ..LindiStagingStore.LindiStagingStore import LindiStagingStore, _apply_templates
from ..LindiH5ZarrStore.LindiH5ZarrStoreOpts import LindiH5ZarrStoreOpts

from ..LocalCache.LocalCache import LocalCache

from ..LindiH5ZarrStore._util import _write_rfs_to_file


LindiFileMode = Literal["r", "r+", "w", "w-", "x", "a"]

# Accepts a string path to a file, uploads (or copies) it somewhere, and returns a string URL
# (or local path)
UploadFileFunc = Callable[[str], str]


class LindiH5pyFile(h5py.File):
    def __init__(self, _zarr_group: zarr.Group, *, _zarr_store: Union[ZarrStore, None] = None, _mode: LindiFileMode = "r", _local_cache: Union[LocalCache, None] = None, _local_file_path: Union[str, None] = None):
        """
        Do not use this constructor directly. Instead, use: from_lindi_file,
        from_h5py_file, from_reference_file_system, from_zarr_store, or
        from_zarr_group.
        """
        self._zarr_group = _zarr_group
        self._zarr_store = _zarr_store
        self._mode: LindiFileMode = _mode
        self._the_group = LindiH5pyGroup(_zarr_group, self)
        self._local_cache = _local_cache
        self._local_file_path = _local_file_path

        # see comment in LindiH5pyGroup
        self._id = f'{id(self._zarr_group)}/'

    @staticmethod
    def from_lindi_file(url_or_path: str, *, mode: LindiFileMode = "r", staging_area: Union[StagingArea, None] = None, local_cache: Union[LocalCache, None] = None, local_file_path: Union[str, None] = None):
        """
        Create a LindiH5pyFile from a URL or path to a .lindi.json file.

        For a description of parameters, see from_reference_file_system().
        """
        if local_file_path is None:
            if not url_or_path.startswith("http://") and not url_or_path.startswith("https://"):
                local_file_path = url_or_path
        return LindiH5pyFile.from_reference_file_system(url_or_path, mode=mode, staging_area=staging_area, local_cache=local_cache, local_file_path=local_file_path)

    @staticmethod
    def from_hdf5_file(url_or_path: str, *, mode: LindiFileMode = "r", local_cache: Union[LocalCache, None] = None, zarr_store_opts: LindiH5ZarrStoreOpts = LindiH5ZarrStoreOpts()):
        """
        Create a LindiH5pyFile from a URL or path to an HDF5 file.

        Parameters
        ----------
        url_or_path : str
            The URL or path to the remote or local HDF5 file.
        mode : Literal["r", "r+", "w", "w-", "x", "a"], optional
            The mode to open the file object in. See api docs for
            h5py.File for more information on the modes, by default "r".
        local_cache : Union[LocalCache, None], optional
            The local cache to use for caching data chunks, by default None.
        """
        from ..LindiH5ZarrStore.LindiH5ZarrStore import LindiH5ZarrStore  # avoid circular import
        if mode != "r":
            raise Exception("Opening hdf5 file in write mode is not supported")
        zarr_store = LindiH5ZarrStore.from_file(url_or_path, local_cache=local_cache, opts=zarr_store_opts, url=url_or_path)
        return LindiH5pyFile.from_zarr_store(
            zarr_store=zarr_store,
            mode=mode,
            local_cache=local_cache
        )

    @staticmethod
    def from_reference_file_system(rfs: Union[dict, str, None], *, mode: LindiFileMode = "r", staging_area: Union[StagingArea, None] = None, local_cache: Union[LocalCache, None] = None, local_file_path: Union[str, None] = None):
        """
        Create a LindiH5pyFile from a reference file system.

        Parameters
        ----------
        rfs : Union[dict, str, None]
            The reference file system. This can be a dictionary or a URL or path
            to a .lindi.json file. If None, an empty reference file system will
            be created.
        mode : Literal["r", "r+", "w", "w-", "x", "a"], optional
            The mode to open the file object in, by default "r".
        staging_area : Union[StagingArea, None], optional
            The staging area to use for writing data, preparing for upload. This
            is only used in write mode, by default None.
        local_cache : Union[LocalCache, None], optional
            The local cache to use for caching data, by default None.
        local_file_path : Union[str, None], optional
            If rfs is not a string or is a remote url, this is the path to the
            local file for the purpose of writing to it. It is required in this
            case if mode is not "r". If rfs is a string and not a remote url, it
            must be equal to local_file_path if provided.
        """
        if rfs is None:
            rfs = {
                "refs": {
                    '.zgroup': {
                        'zarr_format': 2
                    }
                },
            }

        if isinstance(rfs, str):
            rfs_is_url = rfs.startswith("http://") or rfs.startswith("https://")
            if local_file_path is not None and not rfs_is_url and rfs != local_file_path:
                raise Exception(f"rfs is not a remote url, so local_file_path must be the same as rfs, but got: {rfs} and {local_file_path}")
            if rfs_is_url:
                with tempfile.TemporaryDirectory() as tmpdir:
                    filename = f"{tmpdir}/temp.lindi.json"
                    _download_file(rfs, filename)
                    with open(filename, "r") as f:
                        data = json.load(f)
                    assert isinstance(data, dict)  # prevent infinite recursion
                    return LindiH5pyFile.from_reference_file_system(data, mode=mode, staging_area=staging_area, local_cache=local_cache, local_file_path=local_file_path)
            else:
                empty_rfs = {
                    "refs": {
                        '.zgroup': {
                            'zarr_format': 2
                        }
                    },
                }
                if mode == "r":
                    # Readonly, file must exist (default)
                    if not os.path.exists(rfs):
                        raise Exception(f"File does not exist: {rfs}")
                elif mode == "r+":
                    # Read/write, file must exist
                    if not os.path.exists(rfs):
                        raise Exception(f"File does not exist: {rfs}")
                elif mode == "w":
                    # Create file, truncate if exists
                    with open(rfs, "w") as f:
                        json.dump(empty_rfs, f)
                elif mode in ["w-", "x"]:
                    # Create file, fail if exists
                    if os.path.exists(rfs):
                        raise Exception(f"File already exists: {rfs}")
                    with open(rfs, "w") as f:
                        json.dump(empty_rfs, f)
                elif mode == "a":
                    # Read/write if exists, create otherwise
                    if os.path.exists(rfs):
                        with open(rfs, "r") as f:
                            data = json.load(f)
                else:
                    raise Exception(f"Unhandled mode: {mode}")
                with open(rfs, "r") as f:
                    data = json.load(f)
                assert isinstance(data, dict)  # prevent infinite recursion
                return LindiH5pyFile.from_reference_file_system(data, mode=mode, staging_area=staging_area, local_cache=local_cache, local_file_path=local_file_path)
        elif isinstance(rfs, dict):
            # This store does not need to be closed
            store = LindiReferenceFileSystemStore(rfs, local_cache=local_cache)
            if staging_area:
                store = LindiStagingStore(base_store=store, staging_area=staging_area)
            return LindiH5pyFile.from_zarr_store(store, mode=mode, local_file_path=local_file_path)
        else:
            raise Exception(f"Unhandled type for rfs: {type(rfs)}")

    @staticmethod
    def from_zarr_store(zarr_store: ZarrStore, mode: LindiFileMode = "r", local_cache: Union[LocalCache, None] = None, local_file_path: Union[str, None] = None):
        """
        Create a LindiH5pyFile from a zarr store.

        Parameters
        ----------
        zarr_store : ZarrStore
            The zarr store.
        mode : Literal["r", "r+", "w", "w-", "x", "a"], optional
            The mode to open the file object in, by default "r". If the mode is
            "r", the file object will be read-only. For write mode to work, the
            zarr store will need to be writeable as well.
        """
        # note that even though the function is called "open", the zarr_group
        # does not need to be closed
        zarr_group = zarr.open(store=zarr_store, mode=mode)
        assert isinstance(zarr_group, zarr.Group)
        return LindiH5pyFile.from_zarr_group(zarr_group, _zarr_store=zarr_store, mode=mode, local_cache=local_cache, local_file_path=local_file_path)

    @staticmethod
    def from_zarr_group(zarr_group: zarr.Group, *, mode: LindiFileMode = "r", _zarr_store: Union[ZarrStore, None] = None, local_cache: Union[LocalCache, None] = None, local_file_path: Union[str, None] = None):
        """
        Create a LindiH5pyFile from a zarr group.

        Parameters
        ----------
        zarr_group : zarr.Group
            The zarr group.
        mode : Literal["r", "r+", "w", "w-", "x", "a"], optional
            The mode to open the file object in, by default "r". If the mode is
            "r", the file object will be read-only. For write mode to work, the
            zarr store will need to be writeable as well.
        _zarr_store : Union[ZarrStore, None], optional
            The zarr store, internally set for use with
            to_reference_file_system().

        See from_zarr_store().
        """
        return LindiH5pyFile(zarr_group, _zarr_store=_zarr_store, _mode=mode, _local_cache=local_cache, _local_file_path=local_file_path)

    def to_reference_file_system(self):
        """
        Export the internal in-memory representation to a reference file system.
        """
        from ..LindiH5ZarrStore.LindiH5ZarrStore import LindiH5ZarrStore  # avoid circular import
        if self._zarr_store is None:
            raise Exception("Cannot convert to reference file system without zarr store")
        zarr_store = self._zarr_store
        if isinstance(zarr_store, LindiStagingStore):
            zarr_store.consolidate_chunks()
            zarr_store = zarr_store._base_store
        if isinstance(zarr_store, LindiH5ZarrStore):
            return zarr_store.to_reference_file_system()
        if not isinstance(zarr_store, LindiReferenceFileSystemStore):
            raise Exception(f"Cannot create reference file system when zarr store has type {type(self._zarr_store)}")
        rfs = zarr_store.rfs
        rfs_copy = json.loads(json.dumps(rfs))
        LindiReferenceFileSystemStore.replace_meta_file_contents_with_dicts_in_rfs(rfs_copy)
        LindiReferenceFileSystemStore.use_templates_in_rfs(rfs_copy)
        return rfs_copy

    def upload(
        self,
        *,
        on_upload_blob: UploadFileFunc,
        on_upload_main: UploadFileFunc
    ):
        """
        Consolidate the chunks in the staging area, upload them to a storage
        system, updating the references in the base store, and then upload the
        updated reference file system .json file.

        Parameters
        ----------
        on_upload_blob : StoreFileFunc
            A function that takes a string path to a blob file, uploads or copies it
            somewhere, and returns a string URL (or local path).
        on_upload_main : StoreFileFunc
            A function that takes a string path to the main .json file, stores
            it somewhere, and returns a string URL (or local path).

        Returns
        -------
        str
            The URL (or local path) of the uploaded reference file system .json
            file.
        """
        rfs = self.to_reference_file_system()
        blobs_to_upload = set()
        for k, v in rfs['refs'].items():
            if isinstance(v, list) and len(v) == 3:
                url = _apply_templates(v[0], rfs.get('templates', {}))
                if not url.startswith("http://") and not url.startswith("https://"):
                    local_path = url
                    blobs_to_upload.add(local_path)
        blob_mapping = _upload_blobs(blobs_to_upload, on_upload_blob=on_upload_blob)
        for k, v in rfs['refs'].items():
            if isinstance(v, list) and len(v) == 3:
                url1 = _apply_templates(v[0], rfs.get('templates', {}))
                url2 = blob_mapping.get(url1, None)
                if url2 is not None:
                    v[0] = url2
        with tempfile.TemporaryDirectory() as tmpdir:
            rfs_fname = f"{tmpdir}/rfs.lindi.json"
            LindiReferenceFileSystemStore.use_templates_in_rfs(rfs)
            _write_rfs_to_file(rfs=rfs, output_file_name=rfs_fname)
            return on_upload_main(rfs_fname)

    def write_lindi_file(self, filename: str):
        """
        Write the reference file system to a .lindi.json file.

        Parameters
        ----------
        filename : str
            The filename to write to. It must end with '.lindi.json'.
        """
        if not filename.endswith(".lindi.json"):
            raise Exception("Filename must end with '.lindi.json'")
        rfs = self.to_reference_file_system()
        _write_rfs_to_file(rfs=rfs, output_file_name=filename)

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
        return 'r' if self._mode == 'r' else 'r+'

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
        self.flush()

    def flush(self):
        if self._mode != 'r' and self._local_file_path is not None:
            rfs = self.to_reference_file_system()
            _write_rfs_to_file(rfs=rfs, output_file_name=self._local_file_path)

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
        if self._mode == 'r':
            raise Exception("Cannot create group in read-only mode")
        if track_order is not None:
            raise Exception("track_order is not supported (I don't know what it is)")
        return self._the_group.create_group(name)

    def require_group(self, name):
        if self._mode == 'r':
            raise Exception("Cannot require group in read-only mode")
        return self._the_group.require_group(name)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if self._mode == 'r':
            raise Exception("Cannot create dataset in read-only mode")
        return self._the_group.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        if self._mode == 'r':
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


def _upload_blobs(
    blobs: set,
    *,
    on_upload_blob: UploadFileFunc
) -> dict:
    """
    Upload all the blobs in a set to a storage system and return a mapping from
    the original file paths to the URLs of the uploaded files.
    """
    blob_mapping = {}
    for i, blob in enumerate(blobs):
        size = os.path.getsize(blob)
        print(f'Uploading blob {i + 1} of {len(blobs)} {blob} ({_format_size_bytes(size)})')
        blob_url = on_upload_blob(blob)
        blob_mapping[blob] = blob_url
    return blob_mapping


def _format_size_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.1f} GB"
