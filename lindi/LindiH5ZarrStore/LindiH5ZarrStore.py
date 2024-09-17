import json
import base64
from typing import Tuple, Union, List, IO, Any, Dict, Callable
import numpy as np
import zarr
from zarr.storage import Store, MemoryStore
import h5py
from tqdm import tqdm
from ._util import (
    _read_bytes,
    _get_max_num_chunks,
    _apply_to_all_chunk_info,
    _get_chunk_byte_range,
    _get_byte_range_for_contiguous_dataset,
    _join,
    _write_rfs_to_file,
)
from ..conversion.attr_conversion import h5_to_zarr_attr
from ..conversion.reformat_json import reformat_json
from ..conversion.h5_filters_to_codecs import h5_filters_to_codecs
from ..conversion.create_zarr_dataset_from_h5_data import create_zarr_dataset_from_h5_data
from ..LindiH5pyFile.LindiReferenceFileSystemStore import LindiReferenceFileSystemStore
from ..LocalCache.LocalCache import ChunkTooLargeError, LocalCache
from ..LindiRemfile.LindiRemfile import LindiRemfile
from .LindiH5ZarrStoreOpts import LindiH5ZarrStoreOpts
from ..LindiH5pyFile.LindiReferenceFileSystemStore import _get_padded_size, _pad_chunk


class SplitDatasetH5Item:
    """
    Represents a dataset that is a single contiguous chunk in the hdf5 file, but
    is split into multiple chunks for efficient slicing in the zarr store.
    """
    def __init__(self, h5_item, *, contiguous_dataset_max_chunk_size: Union[int, None]):
        self._h5_item = h5_item
        self._contiguous_dataset_max_chunk_size = contiguous_dataset_max_chunk_size
        should_split = False
        if contiguous_dataset_max_chunk_size is not None:
            codecs = h5_filters_to_codecs(h5_item)
            if codecs is None or len(codecs) == 0:  # does not have compression
                if h5_item.chunks is None or h5_item.chunks == h5_item.shape:  # only one chunk
                    if h5_item.dtype.kind in ['i', 'u', 'f']:  # integer or float
                        size_bytes = int(np.prod(h5_item.shape)) * h5_item.dtype.itemsize
                        if size_bytes > contiguous_dataset_max_chunk_size:  # large enough to split
                            should_split = True
        self._do_split = should_split
        if should_split:
            size0 = int(np.prod(h5_item.shape[1:])) * h5_item.dtype.itemsize
            # We want each chunk to be of size around
            # contiguous_dataset_max_chunk_size. So if nn is the size of a chunk
            # in the first dimension, then nn * size0 should be approximately
            # contiguous_dataset_max_chunk_size. So nn should be approximately
            # contiguous_dataset_max_chunk_size // size0
            nn = contiguous_dataset_max_chunk_size // size0
            if nn == 0:
                # The chunk size should not be zero
                nn = 1
            self._split_chunk_shape = (nn,) + h5_item.shape[1:]
            if h5_item.chunks is not None:
                zero_chunk_coords = (0,) * h5_item.ndim
                try:
                    byte_offset, byte_count = _get_chunk_byte_range(h5_item, zero_chunk_coords)
                except Exception as e:
                    raise Exception(
                        f"Error getting byte range for chunk when trying to split contiguous dataset {h5_item.name}: {e}"
                    )
            else:
                # Get the byte range in the file for the contiguous dataset
                byte_offset, byte_count = _get_byte_range_for_contiguous_dataset(h5_item)
            self._split_chunk_byte_offset = byte_offset
            self._split_chunk_byte_count = byte_count
            self._num_chunks = int(np.prod(h5_item.shape[0:]) + np.prod(self._split_chunk_shape) - 1) // int(np.prod(self._split_chunk_shape))
        else:
            self._split_chunk_shape = None
            self._split_chunk_byte_offset = None
            self._split_chunk_byte_count = None
            self._num_chunks = None

    def get_chunk_byte_range(self, chunk_coords: Tuple[int, ...]):
        if len(chunk_coords) != self.ndim:
            raise Exception(f"SplitDatasetH5Item: Chunk coordinates {chunk_coords} do not match dataset dimensions")
        for i in range(1, len(chunk_coords)):
            if chunk_coords[i] != 0:
                raise Exception(f"SplitDatasetH5Item: Unexpected non-zero chunk coordinate {chunk_coords[i]}")
        if self._split_chunk_byte_offset is None:
            raise Exception("SplitDatasetH5Item: Unexpected _split_chunk_byte_offset is None")
        if self._split_chunk_shape is None:
            raise Exception("SplitDatasetH5Item: Unexpected _split_chunk_shape is None")
        chunk_index = chunk_coords[0]
        byte_offset = self._split_chunk_byte_offset + chunk_index * int(np.prod(self._split_chunk_shape)) * self.dtype.itemsize
        byte_count = int(np.prod(self._split_chunk_shape)) * self.dtype.itemsize
        if byte_offset + byte_count > self._split_chunk_byte_offset + self._split_chunk_byte_count:
            byte_count = self._split_chunk_byte_offset + self._split_chunk_byte_count - byte_offset
        return byte_offset, byte_count

    @property
    def shape(self):
        return self._h5_item.shape

    @property
    def dtype(self):
        return self._h5_item.dtype

    @property
    def name(self):
        return self._h5_item.name

    @property
    def chunks(self):
        if self._do_split:
            return self._split_chunk_shape
        return self._h5_item.chunks

    @property
    def ndim(self):
        return self._h5_item.ndim

    @property
    def fillvalue(self):
        return self._h5_item.fillvalue

    @property
    def attrs(self):
        return self._h5_item.attrs

    @property
    def size(self):
        return self._h5_item.size


class LindiH5ZarrStore(Store):
    """A zarr store that provides a read-only view of an HDF5 file.

    Do not call the constructor directly. Instead do one of the following:

    store = LindiH5ZarrStore.from_file(hdf5_file_name)
    # do stuff with store
    store.close()

    or

    with LindiH5ZarrStore.from_file(hdf5_file_name) as store:
        # do stuff with store
    """

    def __init__(
        self,
        *,
        _file: Union[IO, Any],
        _opts: LindiH5ZarrStoreOpts,
        _url: Union[str, None] = None,
        _entities_to_close: List[Any],
        _local_cache: Union[LocalCache, None] = None
    ):
        """
        Do not call the constructor directly. Instead, use the from_file class
        method.
        """
        self._file: Union[IO, Any, None] = _file
        self._h5f: Union[h5py.File, None] = h5py.File(_file, "r")
        self._url = _url
        self._opts = _opts
        self._local_cache = _local_cache
        self._entities_to_close = _entities_to_close + [self._h5f]

        # Some datasets do not correspond to traditional chunked datasets. For
        # those datasets, we need to store the inline data so that we can return
        # it when the chunk is requested.
        self._inline_arrays: Dict[str, InlineArray] = {}

        # For large contiguous arrays, we want to split them into smaller chunks.
        self._split_datasets: Dict[str, SplitDatasetH5Item] = {}

        self._external_array_links: Dict[str, Union[dict, None]] = {}

    @staticmethod
    def from_file(
        hdf5_file_name_or_url: str,
        *,
        opts: Union[LindiH5ZarrStoreOpts, None] = None,
        url: Union[str, None] = None,
        local_cache: Union[LocalCache, None] = None
    ):
        """
        Create a LindiH5ZarrStore from a file or url pointing to an HDF5 file.

        Parameters
        ----------
        hdf5_file_name_or_url : str
            The name of the HDF5 file or a URL to the HDF5 file.
        opts : LindiH5ZarrStoreOpts or None
            Options for the store.
        url : str or None
            If hdf5_file_name_or_url is a local file name, then this can
            optionally be set to the URL of the remote file to be used when
            creating references. If None, and the hdf5_file_name_or_url is a
            local file name, then you will need to set
            opts.num_dataset_chunks_threshold to None, and you will not be able
            to use the to_reference_file_system method.
        local_cache : LocalCache or None
            A local cache to use when reading chunks from a remote file. If None,
            then no local cache is used.
        """
        if opts is None:
            opts = LindiH5ZarrStoreOpts()  # default options
        if hdf5_file_name_or_url.startswith(
            "http://"
        ) or hdf5_file_name_or_url.startswith("https://"):
            # note that the remfile.File object does not need to be closed
            remf = LindiRemfile(hdf5_file_name_or_url, verbose=False, local_cache=local_cache)
            return LindiH5ZarrStore(_file=remf, _url=hdf5_file_name_or_url, _opts=opts, _entities_to_close=[], _local_cache=local_cache)
        else:
            if local_cache is not None:
                raise Exception("local_cache cannot be used with a local file")
            f = open(hdf5_file_name_or_url, "rb")
            return LindiH5ZarrStore(_file=f, _url=url, _opts=opts, _entities_to_close=[f])

    def close(self):
        """Close the store."""
        for e in self._entities_to_close:
            e.close()
        self._entities_to_close.clear()
        self._h5f = None
        self._file = None

    def __getitem__(self, key):
        val = self._get_helper(key)

        if val is not None:
            padded_size = _get_padded_size(self, key, val)
            if padded_size is not None:
                val = _pad_chunk(val, padded_size)

        return val

    def _get_helper(self, key: str):
        """Get an item from the store (required by base class)."""
        parts = [part for part in key.split("/") if part]
        if len(parts) == 0:
            raise KeyError(key)
        key_name = parts[-1]
        key_parent = "/".join(parts[:-1])
        if key_name == ".zattrs":
            # Get the attributes of a group or dataset. If it is empty, we still
            # return it, but we exclude it when writing out the reference file
            # system.
            return self._get_zattrs_bytes(key_parent)
        elif key_name == ".zgroup":
            # Get the .zgroup JSON text for a group
            return self._get_zgroup_bytes(key_parent)
        elif key_name == ".zarray":
            # Get the .zarray JSON text for a dataset
            return self._get_zarray_bytes(key_parent)
        else:
            # Otherwise, we assume it is a chunk file
            return self._get_chunk_file_bytes(key_parent=key_parent, key_name=key_name)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key):
        """Check if a key is in the store (used by zarr)."""
        # it would be nice if we didn't have to repeat the logic from __getitem__
        if self._h5f is None:
            raise Exception("Store is closed")
        parts = [part for part in key.split("/") if part]
        if len(parts) == 0:
            return False
        key_name = parts[-1]
        key_parent = "/".join(parts[:-1])
        if key_name == ".zattrs":
            h5_item = self._h5f.get('/' + key_parent, None)
            if not h5_item:
                return False
            # We always return True here even if the attributes are going to be
            # empty, because it's not worth including the logic. But when we
            # write out the ref file system, we exclude it there.
            return isinstance(h5_item, h5py.Group) or isinstance(h5_item, h5py.Dataset)
        elif key_name == ".zgroup":
            h5_item = self._h5f.get('/' + key_parent, None)
            if not h5_item:
                return False
            return isinstance(h5_item, h5py.Group)
        elif key_name == ".zarray":
            h5_item = self._h5f.get('/' + key_parent, None)
            if not h5_item:
                return False
            return isinstance(h5_item, h5py.Dataset)
        else:
            # a chunk file
            h5_item = self._h5f.get('/' + key_parent, None)
            if not h5_item:
                return False
            if not isinstance(h5_item, h5py.Dataset):
                return False
            if self._split_datasets.get(key_parent, None) is not None:
                h5_item = self._split_datasets[key_parent]
            external_array_link = self._get_external_array_link(key_parent, h5_item)
            if external_array_link is not None:
                # The chunk files do not exist for external array links
                return False
            if np.prod(h5_item.shape) == 0:
                return False
            if h5_item.ndim == 0:
                return key_name == "0"
            chunk_name_parts = key_name.split(".")
            if len(chunk_name_parts) != h5_item.ndim:
                return False
            inline_array = self._get_inline_array(key, h5_item)
            if inline_array.is_inline:
                chunk_coords_shape = (1,) * h5_item.ndim
            else:
                shape = h5_item.shape
                chunks = h5_item.chunks or shape
                chunk_coords_shape = [
                    (shape[i] + chunks[i] - 1) // chunks[i] if chunks[i] != 0 else 0
                    for i in range(len(shape))
                ]
            chunk_coords = tuple(int(x) for x in chunk_name_parts)
            for i, c in enumerate(chunk_coords):
                if c < 0 or c >= chunk_coords_shape[i]:
                    return False
            return True

    def __delitem__(self, key):
        raise Exception("Deleting items is not allowed")

    def __setitem__(self, key, value):
        raise Exception("Setting items is not allowed")

    def __iter__(self):
        raise Exception("Not implemented")

    def __len__(self):
        raise Exception("Not implemented")

    def _get_zattrs_bytes(self, parent_key: str):
        """Get the attributes of a group or dataset"""
        if self._h5f is None:
            raise Exception("Store is closed")
        h5_item = self._h5f.get('/' + parent_key, None)
        if h5_item is None:
            raise KeyError(parent_key)
        if not isinstance(h5_item, h5py.Group) and not isinstance(h5_item, h5py.Dataset):
            raise Exception(f"Item {parent_key} is not a group or dataset. It is {type(h5_item)}")  # pragma: no cover

        # Check whether this is a soft link
        if isinstance(h5_item, (h5py.Group, h5py.Dataset)) and parent_key != '':
            link = self._h5f.get('/' + parent_key, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                raise Exception(f"External links not supported: {parent_key}")
            elif isinstance(link, h5py.SoftLink):
                # if it's a soft link, we return a special attribute and ignore
                # the rest of the attributes because they should be stored in
                # the target of the soft link
                return reformat_json(json.dumps({
                    "_SOFT_LINK": {
                        "path": link.path
                    }
                }).encode("utf-8"))

        # We create a dummy zarr group and copy the attributes to it. That way
        # we know that zarr has accepted them and they are serialized in the
        # correct format.
        memory_store = MemoryStore()
        dummy_group = zarr.group(store=memory_store)
        for k, v in h5_item.attrs.items():
            v2 = h5_to_zarr_attr(v, label=f"{parent_key} {k}", h5f=self._h5f)
            if v2 is not None:
                dummy_group.attrs[k] = v2
        if isinstance(h5_item, h5py.Dataset):
            inline_array = self._get_inline_array(parent_key, h5_item)
            for k, v in inline_array.additional_zarr_attrs.items():
                dummy_group.attrs[k] = v
            external_array_link = self._get_external_array_link(parent_key, h5_item)
            if external_array_link is not None:
                dummy_group.attrs["_EXTERNAL_ARRAY_LINK"] = external_array_link
        zattrs_content = reformat_json(memory_store.get(".zattrs") or "{}".encode("utf-8"))
        return zattrs_content

    def _get_zgroup_bytes(self, parent_key: str):
        """Get the .zgroup JSON text for a group"""
        if self._h5f is None:
            raise Exception("Store is closed")
        h5_item = self._h5f.get('/' + parent_key, None)
        link = self._h5f.get('/' + parent_key, getlink=True) if parent_key != '' else None
        if not isinstance(link, h5py.SoftLink) and not isinstance(h5_item, h5py.Group):
            # Important to raise a KeyError here because that's what zarr expects
            raise KeyError(f"Item {parent_key} is not a group")
        # We create a dummy zarr group and then get the .zgroup JSON text
        # from it.
        memory_store = MemoryStore()
        zarr.group(store=memory_store)
        return reformat_json(memory_store.get(".zgroup"))

    def _get_inline_array(self, key: str, h5_dataset: Union[h5py.Dataset, SplitDatasetH5Item]):
        if key in self._inline_arrays:
            return self._inline_arrays[key]
        self._inline_arrays[key] = InlineArray(h5_dataset)
        return self._inline_arrays[key]

    def _get_zarray_bytes(self, parent_key: str):
        """Get the .zarray JSON text for a dataset"""
        if self._h5f is None:
            raise Exception("Store is closed")
        h5_item = self._h5f.get('/' + parent_key, None)
        if not isinstance(h5_item, h5py.Dataset):
            # Important to raise a KeyError here because that's what zarr expects
            raise KeyError(f"Item {parent_key} is not a dataset")
        # get the shape, chunks, dtype, and filters from the h5 dataset
        inline_array = self._get_inline_array(parent_key, h5_item)
        if inline_array.is_inline:
            return inline_array.zarray_bytes

        filters = h5_filters_to_codecs(h5_item)

        split_dataset = SplitDatasetH5Item(h5_item, contiguous_dataset_max_chunk_size=self._opts.contiguous_dataset_max_chunk_size)
        if split_dataset._do_split:
            self._split_datasets[parent_key] = split_dataset
            h5_item = split_dataset

        # We create a dummy zarr dataset with the appropriate shape, chunks,
        # dtype, and filters and then copy the .zarray JSON text from it
        memory_store = MemoryStore()
        dummy_group = zarr.group(store=memory_store)
        chunks = h5_item.chunks
        if chunks is None:
            # It's important to not have chunks be None here because that would
            # let zarr choose an optimal chunking, whereas we need this to reflect
            # the actual chunking in the HDF5 file.
            chunks = h5_item.shape
            if np.prod(chunks) == 0:
                # A chunking of (0,) or (0, 0) or (0, 0, 0), etc. is not allowed in Zarr
                chunks = [1] * len(chunks)
        # Importantly, I'm pretty sure this doesn't actually create the
        # chunks in the memory store. That's important because we just need
        # to get the .zarray JSON text from the dummy group.
        dummy_group.create_dataset(
            name="dummy_array",
            shape=h5_item.shape,
            chunks=chunks,
            dtype=h5_item.dtype,
            compressor=None,
            order="C",
            fill_value=h5_item.fillvalue,
            filters=filters
        )
        zarray_text = reformat_json(memory_store.get("dummy_array/.zarray"))

        return zarray_text

    def _get_chunk_file_bytes(self, key_parent: str, key_name: str):
        if self._file is None:
            raise Exception("Store is closed")
        byte_offset, byte_count, inline_data = self._get_chunk_file_bytes_data(
            key_parent, key_name
        )
        if inline_data is not None:
            return inline_data
        else:
            assert byte_offset is not None
            assert byte_count is not None
            if self._local_cache is not None:
                assert self._url is not None, "Unexpected: url is None but local_cache is not None"
                ch = self._local_cache.get_remote_chunk(
                    url=self._url,
                    offset=byte_offset,
                    size=byte_count
                )
                if ch is not None:
                    return ch
            buf = _read_bytes(self._file, byte_offset, byte_count)
            if self._local_cache is not None:
                assert self._url is not None, "Unexpected: url is None but local_cache is not None"
                try:
                    self._local_cache.put_remote_chunk(
                        url=self._url,
                        offset=byte_offset,
                        size=byte_count,
                        data=buf
                    )
                except ChunkTooLargeError:
                    print(f"Warning: Unable to store chunk of size {byte_count} in local cache in LindiH5ZarrStore (key: {key_parent}/{key_name})")
            return buf

    def _get_chunk_file_bytes_data(self, key_parent: str, key_name: str):
        if self._h5f is None:
            raise Exception("Store is closed")
        h5_item = self._h5f.get('/' + key_parent, None)
        if not isinstance(h5_item, h5py.Dataset):
            raise Exception(f"Item {key_parent} is not a dataset")

        if self._split_datasets.get(key_parent, None) is not None:
            h5_item = self._split_datasets[key_parent]

        external_array_link = self._get_external_array_link(key_parent, h5_item)
        if external_array_link is not None:
            raise Exception(
                f"Chunk file {key_parent}/{key_name} is not present because this is an external array link."
            )

        # For the case of a scalar dataset, we need to check a few things
        if h5_item.ndim == 0:
            if h5_item.chunks is not None:
                raise Exception(
                    f"Unable to handle case where chunks is not None but ndim is 0 for dataset {key_parent}"
                )
            if key_name != "0":
                raise Exception(
                    f"Chunk name {key_name} must be '0' for scalar dataset {key_parent}"
                )

        # In the case of shape 0, we raise an exception because we shouldn't be here
        if np.prod(h5_item.shape) == 0:
            raise Exception(
                f"Chunk file {key_parent}/{key_name} is not present because the dataset has shape 0."
            )

        inline_array = self._get_inline_array(key_parent, h5_item)
        if inline_array.is_inline:
            if key_name != inline_array.chunk_fname:
                raise Exception(
                    f"Chunk name {key_name} does not match dataset dimensions for inline array {key_parent}"
                )
            return None, None, inline_array.chunk_bytes

        # If this is a scalar, then the data should have been inline
        if h5_item.ndim == 0:
            raise Exception(f"No inline data for scalar dataset {key_parent}")

        # Get the chunk coords from the file name
        chunk_name_parts = key_name.split(".")
        if len(chunk_name_parts) != h5_item.ndim:
            raise Exception(f"Chunk name {key_name} does not match dataset dimensions")
        chunk_coords = tuple(int(x) for x in chunk_name_parts)
        for i, c in enumerate(chunk_coords):
            if c < 0 or c >= h5_item.shape[i]:
                raise Exception(
                    f"Chunk coordinates {chunk_coords} out of range for dataset {key_parent} with dtype {h5_item.dtype}"
                )
        if h5_item.chunks is not None:
            # Get the byte range in the file for the chunk.
            try:
                if isinstance(h5_item, SplitDatasetH5Item):
                    byte_offset, byte_count = h5_item.get_chunk_byte_range(chunk_coords)
                else:
                    byte_offset, byte_count = _get_chunk_byte_range(h5_item, chunk_coords)
            except Exception as e:
                raise Exception(
                    f"Error getting byte range for chunk {key_parent}/{key_name}. Shape: {h5_item.shape}, Chunks: {h5_item.chunks}, Chunk coords: {chunk_coords}: {e}"
                )
        else:
            # In this case (contiguous dataset), we need to check that the chunk
            # coordinates are (0, 0, 0, ...)
            if chunk_coords != (0,) * h5_item.ndim:
                raise Exception(
                    f"Chunk coordinates {chunk_coords} are not (0, 0, 0, ...) for contiguous dataset {key_parent} with dtype {h5_item.dtype} and shape {h5_item.shape}"
                )
            if isinstance(h5_item, SplitDatasetH5Item):
                raise Exception(f'Unexpected SplitDatasetH5Item for contiguous dataset {key_parent}')
            # Get the byte range in the file for the contiguous dataset
            byte_offset, byte_count = _get_byte_range_for_contiguous_dataset(h5_item)
        return byte_offset, byte_count, None

    def _add_chunk_info_to_refs(self, key_parent: str, add_ref: Callable, add_ref_chunk: Callable):
        if self._h5f is None:
            raise Exception("Store is closed")
        h5_item = self._h5f.get('/' + key_parent, None)
        assert isinstance(h5_item, h5py.Dataset)

        if self._split_datasets.get(key_parent, None) is not None:
            h5_item = self._split_datasets[key_parent]

        # If the shape is (0,), (0, 0), (0, 0, 0), etc., then do not add any chunk references
        if np.prod(h5_item.shape) == 0:
            return

        # For the case of a scalar dataset, we need to check a few things
        if h5_item.ndim == 0:
            if h5_item.chunks is not None:
                raise Exception(
                    f"Unable to handle case where chunks is not None but ndim is 0 for dataset {key_parent}"
                )

        inline_array = self._get_inline_array(key_parent, h5_item)
        if inline_array.is_inline:
            key_name = inline_array.chunk_fname
            inline_data = inline_array.chunk_bytes
            add_ref(f"{key_parent}/{key_name}", inline_data)
            return

        # If this is a scalar, then the data should have been inline
        if h5_item.ndim == 0:
            raise Exception(f"No inline data for scalar dataset {key_parent}")

        if h5_item.chunks is not None:
            # Set up progress bar for manual updates because h5py chunk_iter used in _apply_to_all_chunk_info
            # does not provide a way to hook in a progress bar
            # We use max number of chunks instead of actual number of chunks because get_num_chunks is slow
            # for remote datasets.
            num_chunks = _get_max_num_chunks(shape=h5_item.shape, chunk_size=h5_item.chunks)  # NOTE: unallocated chunks are counted
            pbar = tqdm(
                total=num_chunks,
                desc=f"Writing chunk info for {key_parent}",
                leave=True,
                delay=2  # do not show progress bar until 2 seconds have passed
            )

            chunk_size = h5_item.chunks

            if isinstance(h5_item, SplitDatasetH5Item):
                assert h5_item._num_chunks is not None, "Unexpected: _num_chunks is None"
                for i in range(h5_item._num_chunks):
                    chunk_coords = (i,) + (0,) * (h5_item.ndim - 1)
                    byte_offset, byte_count = h5_item.get_chunk_byte_range(chunk_coords)
                    key_name = ".".join([str(x) for x in chunk_coords])
                    add_ref_chunk(f"{key_parent}/{key_name}", (self._url, byte_offset, byte_count))
                    pbar.update()
            else:
                def store_chunk_info(chunk_info):
                    # Get the byte range in the file for each chunk.
                    chunk_offset: Tuple[int, ...] = chunk_info.chunk_offset
                    byte_offset = chunk_info.byte_offset
                    byte_count = chunk_info.size
                    key_name = ".".join([str(a // b) for a, b in zip(chunk_offset, chunk_size)])
                    add_ref_chunk(f"{key_parent}/{key_name}", (self._url, byte_offset, byte_count))
                    pbar.update()

                _apply_to_all_chunk_info(h5_item, store_chunk_info)

            pbar.close()
        else:
            # Get the byte range in the file for the contiguous dataset
            assert not isinstance(h5_item, SplitDatasetH5Item), "Unexpected SplitDatasetH5Item for contiguous dataset"
            byte_offset, byte_count = _get_byte_range_for_contiguous_dataset(h5_item)
            key_name = ".".join("0" for _ in range(h5_item.ndim))
            add_ref_chunk(f"{key_parent}/{key_name}", (self._url, byte_offset, byte_count))

    def _get_external_array_link(self, parent_key: str, h5_item: Union[h5py.Dataset, SplitDatasetH5Item]):
        # First check the memory cache
        if parent_key in self._external_array_links:
            return self._external_array_links[parent_key]
        # Important to set it to None so that we don't keep checking it
        self._external_array_links[parent_key] = None
        if h5_item.chunks and self._opts.num_dataset_chunks_threshold is not None:
            # We compute the expected number of chunks using the shape and chunks
            # and compare it to the threshold. If it's greater, then we create an
            # external array link.
            shape = h5_item.shape
            chunks = h5_item.chunks
            chunk_coords_shape = [
                (shape[i] + chunks[i] - 1) // chunks[i] if chunks[i] != 0 else 0
                for i in range(len(shape))
            ]
            num_chunks = int(np.prod(chunk_coords_shape))
            if num_chunks > self._opts.num_dataset_chunks_threshold:
                if self._url is not None:
                    self._external_array_links[parent_key] = {
                        "link_type": "hdf5_dataset",
                        "url": self._url,
                        "name": parent_key,
                    }
                else:
                    raise Exception(
                        f"Unable to create external array link for {parent_key}: url is not set"
                    )
        return self._external_array_links[parent_key]

    def listdir(self, path: str = "") -> List[str]:
        # This function is used by zarr. We need to return the names of the
        # subdirectories of the given path in the store. We should not return
        # the names of files.
        if self._h5f is None:
            raise Exception("Store is closed")
        try:
            item = self._h5f['/' + path]
        except KeyError:
            return []
        if isinstance(item, h5py.Group):
            # check whether it's a soft link
            link = self._h5f.get('/' + path, getlink=True) if path != '' else None
            if isinstance(link, h5py.SoftLink):
                # in this case we don't return any keys because the keys should
                # be in the target of the soft link
                return []
            # We will have one subdir for each key in the group
            ret = []
            for k in item.keys():
                ret.append(k)
            return ret
        elif isinstance(item, h5py.Dataset):
            # Datasets do not have subdirectories
            return []
        else:
            return []

    def write_reference_file_system(self, output_file_name: str):
        """Write a reference file system corresponding to this store to a file.

        This can then be loaded using LindiH5pyFile.from_lindi_file(file_name)
        """

        if not output_file_name.endswith(".lindi.json"):
            raise Exception("The output file name must end with .lindi.json")

        rfs = self.to_reference_file_system()
        _write_rfs_to_file(rfs=rfs, output_file_name=output_file_name)

    def to_reference_file_system(self) -> dict:
        """Create a reference file system corresponding to this store.

        This can then be loaded using LindiH5pyFile.from_reference_file_system(obj)
        """
        if self._h5f is None:
            raise Exception("Store is closed")
        if self._url is None:
            raise Exception("You must specify a url to create a reference file system")
        ret = {"refs": {}, "version": 1}

        def _add_ref(key: str, content: Union[bytes, None]):
            if content is None:
                raise Exception(f"Unable to get content for key {key}")
            if content.startswith(b"base64:"):
                # This is the rare case where the content actually starts with "base64:"
                # which is confusing. Not sure when this would happen, but it could.
                # TODO: needs a unit test
                ret["refs"][key] = (b"base64:" + base64.b64encode(content)).decode(
                    "ascii"
                )
            else:
                # This is the usual case. It will raise a UnicodeDecodeError if the
                # content is not valid ASCII, in which case the content will be
                # base64 encoded.
                try:
                    ret["refs"][key] = content.decode("ascii")
                except UnicodeDecodeError:
                    # If the content is not valid ASCII, then we base64 encode it. The
                    # reference file system reader will know what to do with it.
                    ret["refs"][key] = (b"base64:" + base64.b64encode(content)).decode(
                        "ascii"
                    )

        def _add_ref_chunk(key: str, data: Tuple[str, int, int]):
            assert data[1] is not None, \
                f"{key} chunk data is invalid. Element at index 1 cannot be None: {data}"
            assert data[2] is not None, \
                f"{key} chunk data is invalid. Element at index 2 cannot be None: {data}"
            ret["refs"][key] = list(data)  # downstream expects a list like on read from a JSON file

        def _process_group(key, item: h5py.Group):
            if isinstance(item, h5py.Group):
                # Add the .zattrs and .zgroup files for the group
                zattrs_bytes = self.get(_join(key, ".zattrs"))
                if zattrs_bytes != b"{}":  # don't include empty zattrs
                    _add_ref(_join(key, ".zattrs"), self.get(_join(key, ".zattrs")))
                _add_ref(_join(key, ".zgroup"), self.get(_join(key, ".zgroup")))
                # check if this is a soft link
                link = item.file.get('/' + key, getlink=True) if key != '' else None
                if isinstance(link, h5py.SoftLink):
                    # if it's a soft link, we don't include the keys because
                    # they should be in the target of the soft link
                    return
                for k in item.keys():
                    subitem = item[k]
                    if isinstance(subitem, h5py.Group):
                        # recursively process subgroups
                        _process_group(_join(key, k), subitem)
                    elif isinstance(subitem, h5py.Dataset):
                        _process_dataset(_join(key, k), subitem)

        def _process_dataset(key, item: h5py.Dataset):
            # Add the .zattrs and .zarray files for the dataset
            zattrs_bytes = self[f"{key}/.zattrs"]
            assert zattrs_bytes is not None
            if zattrs_bytes != b"{}":  # don't include empty zattrs
                _add_ref(f"{key}/.zattrs", zattrs_bytes)

            # check if this is a soft link
            link = item.file.get('/' + key, getlink=True) if key != '' else None
            if isinstance(link, h5py.SoftLink):
                # if it's a soft link, we create a zgroup and don't include
                # the .zarray or array chunks because those should be in the
                # target of the soft link
                _add_ref(_join(key, ".zgroup"), self.get(_join(key, ".zgroup")))
                return

            zarray_bytes = self.get(f"{key}/.zarray")
            assert zarray_bytes is not None
            _add_ref(f"{key}/.zarray", zarray_bytes)

            zattrs_dict = json.loads(zattrs_bytes.decode("utf-8"))
            external_array_link = zattrs_dict.get(
                "_EXTERNAL_ARRAY_LINK", None
            )
            if external_array_link is None:
                # Only add chunk references for datasets without an external array link
                self._add_chunk_info_to_refs(key, _add_ref, _add_ref_chunk)

        # Process the groups recursively starting with the root group
        _process_group("", self._h5f)

        LindiReferenceFileSystemStore.replace_meta_file_contents_with_dicts_in_rfs(ret)
        LindiReferenceFileSystemStore.use_templates_in_rfs(ret)
        return ret


class InlineArray:
    def __init__(self, h5_dataset: Union[h5py.Dataset, SplitDatasetH5Item]):
        self._additional_zarr_attributes = {}
        if h5_dataset.shape == ():
            self._additional_zarr_attributes["_SCALAR"] = True
            self._is_inline = True
            ...
        elif h5_dataset.dtype.kind in ['i', 'u', 'f']:  # integer or float
            if h5_dataset.size and h5_dataset.size < 1000:
                self._is_inline = True
            else:
                self._is_inline = False
        else:
            self._is_inline = True
            if h5_dataset.dtype.kind == "V" and h5_dataset.dtype.fields is not None:  # compound type
                compound_dtype = []
                for name in h5_dataset.dtype.names:
                    tt = h5_dataset.dtype[name]
                    if tt == h5py.special_dtype(ref=h5py.Reference):
                        tt = "<REFERENCE>"
                    compound_dtype.append((name, str(tt)))
                # For example: [['x', 'uint32'], ['y', 'uint32'], ['weight', 'float32']]
                self._additional_zarr_attributes["_COMPOUND_DTYPE"] = compound_dtype
        if self._is_inline:
            if isinstance(h5_dataset, SplitDatasetH5Item):
                raise Exception('SplitDatasetH5Item should not be an inline array')
            memory_store = MemoryStore()
            dummy_group = zarr.group(store=memory_store)
            size_is_zero = np.prod(h5_dataset.shape) == 0
            if isinstance(h5_dataset, SplitDatasetH5Item):
                h5_item = h5_dataset._h5_item
            else:
                h5_item = h5_dataset
            create_zarr_dataset_from_h5_data(
                zarr_parent_group=dummy_group,
                name='X',
                # For inline data it's important for now that we enforce a
                # single chunk because the rest of the code assumes a single
                # chunk for inline data. The assumption is that the inline
                # arrays are not going to be very large.
                h5_chunks=h5_dataset.shape if h5_dataset.shape != () and not size_is_zero else None,
                label=f'{h5_dataset.name}',
                h5_shape=h5_dataset.shape,
                h5_dtype=h5_dataset.dtype,
                h5f=h5_item.file,
                h5_data=h5_item[...]
            )
            self._zarray_bytes = reformat_json(memory_store['X/.zarray'])
            if not size_is_zero:
                if h5_dataset.ndim == 0:
                    chunk_fname = '0'
                else:
                    chunk_fname = '.'.join(['0'] * h5_dataset.ndim)
                self._chunk_fname = chunk_fname
                self._chunk_bytes = memory_store[f'X/{chunk_fname}']
            else:
                self._chunk_fname = None
                self._chunk_bytes = None
        else:
            self._zarray_bytes = None
            self._chunk_fname = None
            self._chunk_bytes = None

    @property
    def is_inline(self):
        return self._is_inline

    @property
    def additional_zarr_attrs(self):
        return self._additional_zarr_attributes

    @property
    def zarray_bytes(self):
        return self._zarray_bytes

    @property
    def chunk_fname(self):
        return self._chunk_fname

    @property
    def chunk_bytes(self):
        return self._chunk_bytes
