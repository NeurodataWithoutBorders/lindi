from typing import Union, Literal
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

from ..LindiH5ZarrStore.LindiH5ZarrStoreOpts import LindiH5ZarrStoreOpts

from ..LocalCache.LocalCache import LocalCache

from ..LindiH5ZarrStore._util import _write_rfs_to_file

from ..tar.lindi_tar import LindiTarFile
from ..tar.LindiTarStore import LindiTarStore


LindiFileMode = Literal["r", "r+", "w", "w-", "x", "a"]


class LindiH5pyFile(h5py.File):
    def __init__(self, _zarr_group: zarr.Group, *, _zarr_store: ZarrStore, _mode: LindiFileMode = "r", _local_cache: Union[LocalCache, None] = None, _source_url_or_path: Union[str, None] = None, _source_tar_file: Union[LindiTarFile, None] = None, _close_source_tar_file_on_close: bool = False):
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
        self._source_url_or_path = _source_url_or_path
        self._source_tar_file = _source_tar_file
        self._close_source_tar_file_on_close = _close_source_tar_file_on_close

        # see comment in LindiH5pyGroup
        self._id = f'{id(self._zarr_group)}/'

        self._is_open = True

    @staticmethod
    def from_lindi_file(url_or_path: str, *, mode: LindiFileMode = "r", local_cache: Union[LocalCache, None] = None):
        """
        Create a LindiH5pyFile from a URL or path to a .lindi.json file.

        For a description of parameters, see from_reference_file_system().
        """
        return LindiH5pyFile.from_reference_file_system(
            url_or_path,
            mode=mode,
            local_cache=local_cache
        )

    @staticmethod
    def from_hdf5_file(
        url_or_path: str,
        *,
        mode: LindiFileMode = "r",
        local_cache: Union[LocalCache, None] = None,
        zarr_store_opts: Union[LindiH5ZarrStoreOpts, None] = None,
        url: Union[str, None] = None
    ):
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
        zarr_store_opts : Union[LindiH5ZarrStoreOpts, None], optional
            The options to use for the zarr store, by default None.
        url : str or None
            If url_or_path is a local file name, then this can
            optionally be set to the URL of the remote file to be used when
            creating references. If None, and the url_or_path is a
            local file name, then you will need to set
            zarr_store_opts.num_dataset_chunks_threshold to None, and you will not be able
            to use the to_reference_file_system method.
        """
        from ..LindiH5ZarrStore.LindiH5ZarrStore import LindiH5ZarrStore  # avoid circular import
        if mode != "r":
            raise ValueError("Opening hdf5 file in write mode is not supported")
        zarr_store = LindiH5ZarrStore.from_file(url_or_path, local_cache=local_cache, opts=zarr_store_opts, url=url)
        return LindiH5pyFile.from_zarr_store(
            zarr_store=zarr_store,
            mode=mode,
            local_cache=local_cache
        )

    @staticmethod
    def from_reference_file_system(rfs: Union[dict, str], *, mode: LindiFileMode = "r", local_cache: Union[LocalCache, None] = None, _source_url_or_path: Union[str, None] = None, _source_tar_file: Union[LindiTarFile, None] = None, _close_source_tar_file_on_close: bool = False):
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
        local_cache : Union[LocalCache, None], optional
            The local cache to use for caching data, by default None.
        _source_url_or_path : Union[str, None], optional
            Internal use only
        _source_tar_file : Union[LindiTarFile, None], optional
            Internal use only
        _close_source_tar_file_on_close : bool, optional
            Internal use only
        """
        if isinstance(rfs, str):
            if _source_url_or_path is not None:
                raise Exception("_source_file_path is not None even though rfs is a string")  # pragma: no cover
            if _source_tar_file is not None:
                raise Exception("_source_tar_file is not None even though rfs is a string")  # pragma: no cover
            rfs_is_url = rfs.startswith("http://") or rfs.startswith("https://")
            if rfs_is_url:
                data, tar_file = _load_rfs_from_url(rfs)
                return LindiH5pyFile.from_reference_file_system(
                    data,
                    mode=mode,
                    local_cache=local_cache,
                    _source_tar_file=tar_file,
                    _source_url_or_path=rfs,
                    _close_source_tar_file_on_close=_close_source_tar_file_on_close
                )
            else:
                # local file (or directory)
                need_to_create_empty_file = False
                if mode == "r":
                    # Readonly, file must exist (default)
                    if not os.path.exists(rfs):
                        raise FileNotFoundError(f"File does not exist: {rfs}")
                elif mode == "r+":
                    # Read/write, file must exist
                    if not os.path.exists(rfs):
                        raise FileNotFoundError(f"File does not exist: {rfs}")
                elif mode == "w":
                    # Create file, truncate if exists
                    need_to_create_empty_file = True

                elif mode in ["w-", "x"]:
                    # Create file, fail if exists
                    if os.path.exists(rfs):
                        raise ValueError(f"File already exists: {rfs}")
                    need_to_create_empty_file = True
                    # Now that we have already checked for existence, let's just change mode to 'w'
                    mode = 'w'
                elif mode == "a":
                    # Read/write if exists, create otherwise
                    if not os.path.exists(rfs):
                        need_to_create_empty_file = True
                else:
                    raise Exception(f"Unhandled mode: {mode}")  # pragma: no cover
                if need_to_create_empty_file:
                    is_tar = rfs.endswith(".tar")
                    is_dir = rfs.endswith(".d")
                    _create_empty_lindi_file(rfs, is_tar=is_tar, is_dir=is_dir)
                data, tar_file = _load_rfs_from_local_file_or_dir(rfs)
                assert isinstance(data, dict)  # prevent infinite recursion
                return LindiH5pyFile.from_reference_file_system(
                    data,
                    mode=mode,
                    local_cache=local_cache,
                    _source_url_or_path=rfs,
                    _source_tar_file=tar_file,
                    _close_source_tar_file_on_close=True
                )
        elif isinstance(rfs, dict):
            # This store does not need to be closed
            store = LindiReferenceFileSystemStore(
                rfs,
                local_cache=local_cache,
                _source_url_or_path=_source_url_or_path,
                _source_tar_file=_source_tar_file
            )
            source_is_url = _source_url_or_path is not None and (_source_url_or_path.startswith("http://") or _source_url_or_path.startswith("https://"))
            if _source_url_or_path and _source_tar_file and not source_is_url:
                store = LindiTarStore(base_store=store, tar_file=_source_tar_file)
            return LindiH5pyFile.from_zarr_store(
                store,
                mode=mode,
                local_cache=local_cache,
                _source_url_or_path=_source_url_or_path,
                _source_tar_file=_source_tar_file,
                _close_source_tar_file_on_close=_close_source_tar_file_on_close
            )
        else:
            raise Exception(f"Unhandled type for rfs: {type(rfs)}")  # pragma: no cover

    @staticmethod
    def from_zarr_store(zarr_store: ZarrStore, mode: LindiFileMode = "r", local_cache: Union[LocalCache, None] = None, _source_url_or_path: Union[str, None] = None, _source_tar_file: Union[LindiTarFile, None] = None, _close_source_tar_file_on_close: bool = False):
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
        return LindiH5pyFile.from_zarr_group(zarr_group, _zarr_store=zarr_store, mode=mode, local_cache=local_cache, _source_url_or_path=_source_url_or_path, _source_tar_file=_source_tar_file, _close_source_tar_file_on_close=_close_source_tar_file_on_close)

    @staticmethod
    def from_zarr_group(zarr_group: zarr.Group, *, mode: LindiFileMode = "r", _zarr_store: ZarrStore, local_cache: Union[LocalCache, None] = None, _source_url_or_path: Union[str, None] = None, _source_tar_file: Union[LindiTarFile, None] = None, _close_source_tar_file_on_close: bool = False):
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
        return LindiH5pyFile(zarr_group, _zarr_store=_zarr_store, _mode=mode, _local_cache=local_cache, _source_url_or_path=_source_url_or_path, _source_tar_file=_source_tar_file, _close_source_tar_file_on_close=_close_source_tar_file_on_close)

    def to_reference_file_system(self):
        """
        Export the internal in-memory representation to a reference file system.
        """
        from ..LindiH5ZarrStore.LindiH5ZarrStore import LindiH5ZarrStore  # avoid circular import
        zarr_store = self._zarr_store
        if isinstance(zarr_store, LindiTarStore):
            zarr_store = zarr_store._base_store
        if isinstance(zarr_store, LindiH5ZarrStore):
            return zarr_store.to_reference_file_system()
        if not isinstance(zarr_store, LindiReferenceFileSystemStore):
            raise Exception(f"Cannot create reference file system when zarr store has type {type(self._zarr_store)}")  # pragma: no cover
        rfs = zarr_store.rfs
        rfs_copy = json.loads(json.dumps(rfs))
        LindiReferenceFileSystemStore.replace_meta_file_contents_with_dicts_in_rfs(rfs_copy)
        LindiReferenceFileSystemStore.use_templates_in_rfs(rfs_copy)
        return rfs_copy

    def write_lindi_file(self, filename: str, *, generation_metadata: Union[dict, None] = None):
        """
        Write the reference file system to a lindi or .lindi.json file.

        Parameters
        ----------
        filename : str
            The filename (or directory) to write to. It must end with '.lindi.json', '.lindi.tar', or '.lindi.d'.
        generation_metadata : Union[dict, None], optional
            The optional generation metadata to include in the reference file
            system, by default None. This information dict is simply set to the
            'generationMetadata' key in the reference file system.
        """
        if not filename.endswith(".lindi.json") and not filename.endswith(".lindi.tar") and not filename.endswith(".lindi.d"):
            raise ValueError("Filename must end with '.lindi.json', '.lindi.tar', or '.lindi.d'.")
        rfs = self.to_reference_file_system()
        if self._source_tar_file:
            source_is_remote = self._source_url_or_path is not None and (self._source_url_or_path.startswith("http://") or self._source_url_or_path.startswith("https://"))
            if not source_is_remote:
                raise ValueError("Cannot write to lindi file if the source is a local lindi tar file because it would not be able to resolve the local references within the tar file.")
            assert self._source_url_or_path is not None
            _update_internal_references_to_remote_tar_file(rfs, self._source_url_or_path, self._source_tar_file)
        if generation_metadata is not None:
            rfs['generationMetadata'] = generation_metadata
        if filename.endswith(".lindi.json"):
            _write_rfs_to_file(rfs=rfs, output_file_name=filename)
        elif filename.endswith(".lindi.tar"):
            LindiTarFile.create(filename, rfs=rfs)
        elif filename.endswith(".d"):
            LindiTarFile.create(filename, rfs=rfs, dir_representation=True)
        else:
            raise Exception("Unhandled file extension")  # pragma: no cover

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
        if not self._is_open:
            print('Warning: LINDI file already closed.')  # pragma: no cover
            return  # pragma: no cover
        self.flush()
        if self._close_source_tar_file_on_close and self._source_tar_file:
            self._source_tar_file.close()
        self._is_open = False

    def flush(self):
        if not self._is_open:
            return  # pragma: no cover
        if self._mode != 'r' and self._source_url_or_path is not None:
            is_url = self._source_url_or_path.startswith("http://") or self._source_url_or_path.startswith("https://")
            if is_url:
                raise Exception("Cannot write to URL")  # pragma: no cover
            rfs = self.to_reference_file_system()
            if self._source_tar_file:
                self._source_tar_file.write_rfs(rfs)
                self._source_tar_file._update_index_in_file()  # very important
            else:
                _write_rfs_to_file(rfs=rfs, output_file_name=self._source_url_or_path)

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
            raise Exception(f"Unexpected type for source in copy: {type(src_item)}")  # pragma: no cover
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
                raise Exception(f'For now, source of reference must be ".", got "{name._source}"')  # pragma: no cover
            if name._source_object_id is not None:
                if name._source_object_id != zarr_group.attrs.get("object_id"):  # pragma: no cover
                    raise Exception(f'Mismatch in source object_id: "{name._source_object_id}" and "{zarr_group.attrs.get("object_id")}"')  # pragma: no cover
            target = self[name._path]
            if name._object_id is not None:
                if name._object_id != target.attrs.get("object_id"):  # pragma: no cover
                    raise Exception(f'Mismatch in object_id: "{name._object_id}" and "{target.attrs.get("object_id")}"')  # pragma: no cover
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

    def keys(self):  # type: ignore
        return self._the_group.keys()

    def items(self):
        return self._the_group.items()

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
            raise ValueError("Cannot create group in read-only mode")
        if track_order is not None:
            raise Exception("track_order is not supported (I don't know what it is)")
        return self._the_group.create_group(name)

    def require_group(self, name):
        if self._mode == 'r':
            raise ValueError("Cannot require group in read-only mode")
        return self._the_group.require_group(name)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if self._mode == 'r':
            raise ValueError("Cannot create dataset in read-only mode")
        return self._the_group.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        if self._mode == 'r':
            raise ValueError("Cannot require dataset in read-only mode")
        return self._the_group.require_dataset(name, shape, dtype, exact=exact, **kwds)


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
                raise Exception("src_item.name is None")  # pragma: no cover
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
                            # important to do a deep copy
                            val = _deep_copy(src_rfs['refs'][src_ref_key])
                            if isinstance(val, list) and len(val) > 0:
                                # if it's a list then we need to resolve any
                                # templates in the first element of the list.
                                # This is very important because the destination
                                # rfs will probably have different templates.
                                url0 = _apply_templates(val[0], src_rfs.get('templates', {}))
                                val[0] = url0
                            dst_rfs['refs'][dst_ref_key] = val
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


def _load_rfs_from_url(url: str):
    file_size = _get_file_size_of_remote_file(url)
    if file_size < 1024 * 1024 * 2:
        # if it's a small file, we'll just download the whole thing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_fname = f"{tmpdir}/temp.lindi.json"
            _download_file(url, tmp_fname)
            data, tar_file = _load_rfs_from_local_file_or_dir(tmp_fname)
            return data, tar_file
    else:
        # if it's a large file, we start by downloading the entry file and then the index file
        tar_entry_buf = _download_file_byte_range(url, 0, 512)
        is_tar = _check_is_tar_header(tar_entry_buf[:512])
        if is_tar:
            tar_file = LindiTarFile(url)
            rfs_json = tar_file.read_file("lindi.json")
            rfs = json.loads(rfs_json)
            return rfs, tar_file
        else:
            # In this case, it must be a regular json file
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_fname = f"{tmpdir}/temp.lindi.json"
                _download_file(url, tmp_fname)
                with open(tmp_fname, "r") as f:
                    return json.load(f), None


def _load_rfs_from_local_file_or_dir(fname: str):
    if os.path.isdir(fname):
        dir_file = LindiTarFile(fname, dir_representation=True)
        rfs_json = dir_file.read_file("lindi.json")
        rfs = json.loads(rfs_json)
        return rfs, dir_file
    file_size = os.path.getsize(fname)
    if file_size >= 512:
        # Read first bytes to check if it's a tar file
        with open(fname, "rb") as f:
            tar_entry_buf = f.read(512)
        is_tar = _check_is_tar_header(tar_entry_buf)
        if is_tar:
            tar_file = LindiTarFile(fname)
            rfs_json = tar_file.read_file("lindi.json")
            rfs = json.loads(rfs_json)
            return rfs, tar_file

    # Must be a regular json file
    with open(fname, "r") as f:
        return json.load(f), None


def _check_is_tar_header(header_buf: bytes) -> bool:
    if len(header_buf) < 512:
        return False

    # We're only going to support ustar format
    # get the ustar indicator at bytes 257-262
    if header_buf[257:262] == b"ustar" and header_buf[262] == 0:
        # Note that it's unlikely but possible that a json file could have the
        # string "ustar" at these bytes, but it would not have a null byte at
        # byte 262
        return True

    # Check for any 0 bytes in the header
    if b"\0" in header_buf:
        print(header_buf[257:262])
        raise Exception("Problem with lindi file: 0 byte found in header, but not ustar tar format")

    return False


def _get_file_size_of_remote_file(url: str) -> int:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return int(response.headers['Content-Length'])


def _download_file_byte_range(url: str, start: int, end: int) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Range": f"bytes={start}-{end - 1}"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return response.read()


empty_rfs = {
    "refs": {
        ".zgroup": {
            "zarr_format": 2
        }
    }
}


def _create_empty_lindi_file(fname: str, *, is_tar: bool = False, is_dir: bool = False):
    if is_tar:
        if is_dir:
            raise Exception("Cannot be both tar and dir")
        LindiTarFile.create(fname, rfs=empty_rfs)
    elif is_dir:
        LindiTarFile.create(fname, rfs=empty_rfs, dir_representation=True)
    else:
        with open(fname, "w") as f:
            json.dump(empty_rfs, f)


def _update_internal_references_to_remote_tar_file(rfs: dict, remote_url: str, remote_tar_file: LindiTarFile):
    # This is tricky. This happens when the source is a remote tar file and we
    # are trying to write the lindi file locally, but we need to update the
    # internal references to point to the remote tar file. Yikes.

    # First we remove all templates to simplify the process. We will restore them below.
    LindiReferenceFileSystemStore.remove_templates_in_rfs(rfs)

    for k, v in rfs['refs'].items():
        if isinstance(v, list):
            if len(v) == 3:
                url = v[0]
                if url.startswith('./'):
                    internal_path = url[2:]
                    if not remote_tar_file._dir_representation:
                        info = remote_tar_file.get_file_info(internal_path)
                        start_byte = info['d']
                        num_bytes = info['s']
                        v[0] = remote_url
                        v[1] = start_byte + v[1]
                        if v[1] + v[2] > start_byte + num_bytes:
                            raise Exception(f"Reference goes beyond end of file: {v[1] + v[2]} > {num_bytes}")
                        # v[2] stays the same, it is the size
                    else:
                        v[0] = remote_url + '/' + internal_path
            elif len(v) == 1:
                # This is a reference to the full file
                url = v[0]
                if url.startswith('./'):
                    internal_path = url[2:]
                    if not remote_tar_file._dir_representation:
                        info = remote_tar_file.get_file_info(internal_path)
                        start_byte = info['d']
                        num_bytes = info['s']
                        v[0] = remote_url
                        v.append(start_byte)
                        v.append(num_bytes)
                    else:
                        v[0] = remote_url + '/' + internal_path
            else:
                raise Exception(f"Unexpected length for reference: {len(v)}")

    LindiReferenceFileSystemStore.use_templates_in_rfs(rfs)


def _apply_templates(x: str, templates: dict) -> str:
    if '{{' in x and '}}' in x:
        for key, val in templates.items():
            x = x.replace('{{' + key + '}}', val)
    return x
