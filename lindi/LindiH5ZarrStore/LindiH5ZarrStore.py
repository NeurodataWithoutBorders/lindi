import json
import base64
from typing import Union, List, IO, Any, Dict, Literal
from dataclasses import dataclass
import numpy as np
import zarr
import remfile
from zarr.storage import Store, MemoryStore
import h5py
from ._zarr_info_for_h5_dataset import _zarr_info_for_h5_dataset
from ._util import (
    _read_bytes,
    _get_chunk_byte_range,
    _get_byte_range_for_contiguous_dataset,
)
from ._h5_attr_to_zarr_attr import _h5_attr_to_zarr_attr
from ._utils import _join, _get_chunk_names_for_dataset, _reformat_json


@dataclass
class LindiH5ZarrStoreOpts:
    """
    Options for the LindiH5ZarrStore class.

    Attributes:
        num_dataset_chunks_threshold (Union[int, None]): For each dataset in the
        HDF5 file, if the number of chunks is greater than this threshold, then
        the dataset will be represented as an external array link. If None, then
        the threshold is not used. Default is 1000.
    """
    num_dataset_chunks_threshold: Union[int, None] = 1000


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
        _entities_to_close: List[Any]
    ):
        """
        Do not call the constructor directly. Instead, use the from_file class
        method.
        """
        self._file: Union[IO, Any, None] = _file
        self._h5f: Union[h5py.File, None] = h5py.File(_file, "r")
        self._url = _url
        self._opts = _opts
        self._entities_to_close = _entities_to_close + [self._h5f]

        # Some datasets do not correspond to traditional chunked datasets. For
        # those datasets, we need to store the inline data so that we can return
        # it when the chunk is requested. We store the inline data in a
        # dictionary with the dataset name as the key. The values are the bytes.
        self._inline_data_for_arrays: Dict[str, bytes] = {}

        self._external_array_links: Dict[str, Union[dict, None]] = {}

    @staticmethod
    def from_file(
        hdf5_file_name_or_url: str,
        *,
        opts: LindiH5ZarrStoreOpts = LindiH5ZarrStoreOpts(),
        url: Union[str, None] = None,
    ):
        """
        Create a LindiH5ZarrStore from a file or url pointing to an HDF5 file.

        Parameters
        ----------
        hdf5_file_name_or_url : str
            The name of the HDF5 file or a URL to the HDF5 file.
        opts : LindiH5ZarrStoreOpts
            Options for the store.
        url : str or None
            If hdf5_file_name_or_url is a local file name, then this can
            optionally be set to the URL of the remote file to be used when
            creating references. If None, and the hdf5_file_name_or_url is a
            local file name, then you will need to set
            opts.num_dataset_chunks_threshold to None, and you will not be able
            to use the to_reference_file_system method.
        """
        if hdf5_file_name_or_url.startswith(
            "http://"
        ) or hdf5_file_name_or_url.startswith("https://"):
            # note that the remfile.File object does not need to be closed
            remf = remfile.File(hdf5_file_name_or_url, verbose=False)
            return LindiH5ZarrStore(_file=remf, _url=hdf5_file_name_or_url, _opts=opts, _entities_to_close=[])
        else:
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
            external_array_link = self._get_external_array_link(key_parent, h5_item)
            if external_array_link is not None:
                # The chunk files do not exist for external array links
                return False
            if h5_item.ndim == 0:
                return key_name == "0"
            chunk_name_parts = key_name.split(".")
            if len(chunk_name_parts) != h5_item.ndim:
                return False
            shape = h5_item.shape
            chunks = h5_item.chunks or shape
            chunk_coords_shape = [
                (shape[i] + chunks[i] - 1) // chunks[i] for i in range(len(shape))
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
        if isinstance(h5_item, h5py.Group) and parent_key != '':
            link = self._h5f.get('/' + parent_key, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                raise Exception(f"External links not supported: {parent_key}")
            elif isinstance(link, h5py.SoftLink):
                # if it's a soft link, we return a special attribute and ignore
                # the rest of the attributes because they should be stored in
                # the target of the soft link
                return _reformat_json(json.dumps({
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
            v2 = _h5_attr_to_zarr_attr(v, label=f"{parent_key} {k}", h5f=self._h5f)
            if v2 is not None:
                dummy_group.attrs[k] = v2
        if isinstance(h5_item, h5py.Dataset):
            if h5_item.ndim == 0:
                dummy_group.attrs["_SCALAR"] = True
            if h5_item.dtype.kind == "V":  # compound type
                compound_dtype = [
                    [name, str(h5_item.dtype[name])]
                    for name in h5_item.dtype.names
                ]
                # For example: [['x', 'uint32'], ['y', 'uint32'], ['weight', 'float32']]
                dummy_group.attrs["_COMPOUND_DTYPE"] = compound_dtype
            external_array_link = self._get_external_array_link(parent_key, h5_item)
            if external_array_link is not None:
                dummy_group.attrs["_EXTERNAL_ARRAY_LINK"] = external_array_link
        zattrs_content = _reformat_json(memory_store.get(".zattrs"))
        if zattrs_content is not None:
            return zattrs_content
        else:
            # No attributes, so we return an empty JSON object
            return "{}".encode("utf-8")

    def _get_zgroup_bytes(self, parent_key: str):
        """Get the .zgroup JSON text for a group"""
        if self._h5f is None:
            raise Exception("Store is closed")
        h5_item = self._h5f.get('/' + parent_key, None)
        if not isinstance(h5_item, h5py.Group):
            raise Exception(f"Item {parent_key} is not a group")
        # We create a dummy zarr group and then get the .zgroup JSON text
        # from it.
        memory_store = MemoryStore()
        zarr.group(store=memory_store)
        return _reformat_json(memory_store.get(".zgroup"))

    def _get_zarray_bytes(self, parent_key: str):
        """Get the .zarray JSON text for a dataset"""
        if self._h5f is None:
            raise Exception("Store is closed")
        h5_item = self._h5f.get('/' + parent_key, None)
        if not isinstance(h5_item, h5py.Dataset):
            raise Exception(f"Item {parent_key} is not a dataset")
        # get the shape, chunks, dtype, and filters from the h5 dataset
        info = _zarr_info_for_h5_dataset(h5_item)
        if info.inline_data is not None:
            self._inline_data_for_arrays[parent_key] = info.inline_data
        # We create a dummy zarr dataset with the appropriate shape, chunks,
        # dtype, and filters and then copy the .zarray JSON text from it
        memory_store = MemoryStore()
        dummy_group = zarr.group(store=memory_store)
        # Importantly, I'm pretty sure this doesn't actually create the
        # chunks in the memory store. That's important because we just need
        # to get the .zarray JSON text from the dummy group.
        dummy_group.create_dataset(
            name="dummy_array",
            shape=info.shape,
            chunks=info.chunks,
            dtype=info.dtype,
            compressor=None,
            order="C",
            fill_value=info.fill_value,
            filters=info.filters,
            object_codec=info.object_codec,
        )
        zarray_text = _reformat_json(memory_store.get("dummy_array/.zarray"))

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
            buf = _read_bytes(self._file, byte_offset, byte_count)
            return buf

    def _get_chunk_file_bytes_data(self, key_parent: str, key_name: str):
        if self._h5f is None:
            raise Exception("Store is closed")
        h5_item = self._h5f.get('/' + key_parent, None)
        if not isinstance(h5_item, h5py.Dataset):
            raise Exception(f"Item {key_parent} is not a dataset")

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
                    f"Chunk name {key_name} does not match dataset dimensions"
                )

        # Check whether we have inline data for this array. This would be set
        # when we created the .zarray JSON text for the dataset. Note that this
        # means that the .zarray file needs to be read before the chunk files,
        # which should always be the case (I assume).
        if key_parent in self._inline_data_for_arrays:
            x = self._inline_data_for_arrays[key_parent]
            if isinstance(x, bytes):
                return None, None, x
            else:
                raise Exception(
                    f"Inline data for dataset {key_parent} is not bytes. It is {type(x)}"
                )

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
                    f"Chunk coordinates {chunk_coords} out of range for dataset {key_parent}"
                )
        if h5_item.chunks is not None:
            # Get the byte range in the file for the chunk.
            byte_offset, byte_count = _get_chunk_byte_range(h5_item, chunk_coords)
        else:
            # In this case (contiguous dataset), we need to check that the chunk
            # coordinates are (0, 0, 0, ...)
            if chunk_coords != (0,) * h5_item.ndim:
                raise Exception(
                    f"Chunk coordinates {chunk_coords} are not (0, 0, 0, ...) for contiguous dataset {key_parent}"
                )
            # Get the byte range in the file for the contiguous dataset
            byte_offset, byte_count = _get_byte_range_for_contiguous_dataset(h5_item)
        return byte_offset, byte_count, None

    def _get_external_array_link(self, parent_key: str, h5_item: h5py.Dataset):
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
                (shape[i] + chunks[i] - 1) // chunks[i] for i in range(len(shape))
            ]
            num_chunks = np.prod(chunk_coords_shape)
            if num_chunks > self._opts.num_dataset_chunks_threshold:
                if self._url is not None:
                    self._external_array_links[parent_key] = {
                        "link_type": "hdf5_dataset",
                        "url": self._url,  # url is not going to be null based on the check in __init__
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

    def to_file(self, file_name: str, *, file_type: Literal["zarr.json"] = "zarr.json"):
        """Write a reference file system cooresponding to this store to a file.

        This can then be loaded using LindiH5pyFile.from_reference_file_system(file_name)
        """
        if file_type != "zarr.json":
            raise Exception(f"Unsupported file type: {file_type}")

        ret = self.to_reference_file_system()
        with open(file_name, "w") as f:
            json.dump(ret, f, indent=2)

    def to_reference_file_system(self) -> dict:
        """Create a reference file system cooresponding to this store.

        This can then be loaded using LindiH5pyFile.from_reference_file_system(obj)
        """
        if self._h5f is None:
            raise Exception("Store is closed")
        if self._url is None:
            raise Exception("You must specify a url to create a reference file system")
        ret = {"refs": {}, "version": 1}

        # TODO: use templates to decrease the size of the JSON

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
                        _process_dataset(_join(key, k))

        def _process_dataset(key):
            # Add the .zattrs and .zarray files for the dataset
            zattrs_bytes = self.get(f"{key}/.zattrs")
            assert zattrs_bytes is not None
            if zattrs_bytes != b"{}":  # don't include empty zattrs
                _add_ref(f"{key}/.zattrs", zattrs_bytes)
            zattrs_dict = json.loads(zattrs_bytes.decode("utf-8"))
            external_array_link = zattrs_dict.get(
                "_EXTERNAL_ARRAY_LINK", None
            )
            zarray_bytes = self.get(f"{key}/.zarray")
            assert zarray_bytes is not None
            _add_ref(f"{key}/.zarray", zarray_bytes)
            zarray_dict = json.loads(zarray_bytes.decode("utf-8"))

            if external_array_link is None:
                # Only add chunk references for datasets without an external array link
                shape = zarray_dict["shape"]
                chunks = zarray_dict.get("chunks", None)
                chunk_coords_shape = [
                    (shape[i] + chunks[i] - 1) // chunks[i]
                    for i in range(len(shape))
                ]
                # For example, chunk_names could be ['0', '1', '2', ...]
                # or ['0.0', '0.1', '0.2', ...]
                chunk_names = _get_chunk_names_for_dataset(
                    chunk_coords_shape
                )
                for chunk_name in chunk_names:
                    byte_offset, byte_count, inline_data = (
                        self._get_chunk_file_bytes_data(key, chunk_name)
                    )
                    if inline_data is not None:
                        # The data is inline for this chunk
                        _add_ref(f"{key}/{chunk_name}", inline_data)
                    else:
                        # In this case we reference a chunk of data in a separate file
                        assert byte_offset is not None
                        assert byte_count is not None
                        ret["refs"][f"{key}/{chunk_name}"] = [
                            self._url,
                            byte_offset,
                            byte_count,
                        ]

        # Process the groups recursively starting with the root group
        _process_group("", self._h5f)
        return ret
