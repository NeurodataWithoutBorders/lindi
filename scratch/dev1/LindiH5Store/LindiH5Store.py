import json
from typing import Union, List, IO, Any, Dict
import numpy as np
import zarr
from zarr.storage import Store, MemoryStore
import h5py
from zarr.errors import ReadOnlyError
from ._zarr_info_for_h5_dataset import _zarr_info_for_h5_dataset
from ._util import _get_h5_item, _read_bytes, _get_chunk_byte_range, _get_byte_range_for_contiguous_dataset
from ._h5_attr_to_zarr_attr import _h5_attr_to_zarr_attr


class LindiH5Store(Store):
    """A zarr store that provides a read-only view of an HDF5 file.

    Parameters
    ----------
    file : file-like object
        The HDF5 file to read from (not h5py.File object, but a file-like
        object, e.g. a file opened with open()). The reason for this is that we
        read chunks directly from the file rather than using h5py.
    """
    def __init__(self, *,
                 file: Union[IO, Any],
                 num_dataset_chunks_threshold: Union[int, None] = None,
                 url: Union[str, None] = None,
                 ):
        self.file = file
        self.h5f = h5py.File(file, 'r')
        self._url = url

        # Some datasets do not correspond to traditional chunked datasets. For
        # those datasets, we need to store the inline data so that we can
        # return it when the chunk is requested. We store the inline data in a
        # dictionary with the dataset name as the key. The values are of type
        # bytes.
        self._inline_data_for_arrays: Dict[str, bytes] = {}

        self._num_dataset_chunks_threshold = num_dataset_chunks_threshold
        self._external_array_links: Dict[str, Union[dict, None]] = {}
        if num_dataset_chunks_threshold is not None:
            if self._url is None:
                raise Exception('You must specify a url if num_dataset_chunks_threshold is not None')

    def __getitem__(self, key):
        """Get an item from the store (required by base class)."""
        parts = key.split('/')
        if len(parts) == 0:
            raise KeyError(key)
        key_name = parts[-1]
        key_parent = '/'.join(parts[:-1])
        if key_name == '.zattrs':
            # Get the attributes of a group or dataset
            return self._get_zattrs_bytes(key_parent)
        elif key_name == '.zgroup':
            # Get the .zgroup JSON text for a group
            return self._get_zgroup_bytes(key_parent)
        elif key_name == '.zarray':
            # Get the .zarray JSON text for a dataset
            return self._get_zarray_bytes(key_parent)
        else:
            # Otherwise, we assume it is a chunk file
            return self._get_chunk_file_bytes(key_parent=key_parent, key_name=key_name)

    def __contains__(self, key):
        """Check if a key is in the store (used by zarr)."""
        # it would be nice if we didn't have to repeat the logic from __getitem__
        parts = key.split('/')
        if len(parts) == 0:
            return False
        key_name = parts[-1]
        key_parent = '/'.join(parts[:-1])
        if key_name == '.zattrs':
            h5_item = _get_h5_item(self.h5f, key_parent)
            return isinstance(h5_item, h5py.Group)
        elif key_name == '.zgroup':
            h5_item = _get_h5_item(self.h5f, key_parent)
            return isinstance(h5_item, h5py.Group)
        elif key_name == '.zarray':
            h5_item = _get_h5_item(self.h5f, key_parent)
            return isinstance(h5_item, h5py.Dataset)
        else:
            # a chunk file
            h5_item = _get_h5_item(self.h5f, key_parent)
            if not isinstance(h5_item, h5py.Dataset):
                return False
            external_array_link = self._get_external_array_link(key_parent, h5_item)
            if external_array_link is not None:
                # The chunk files do not exist for external array links
                return False
            if h5_item.ndim == 0:
                return key_name == '0'
            chunk_name_parts = key_name.split('.')
            if len(chunk_name_parts) != h5_item.ndim:
                return False
            chunk_coords = tuple(int(x) for x in chunk_name_parts)
            for i, c in enumerate(chunk_coords):
                if c < 0 or c >= h5_item.shape[i]:
                    return False
            return True

    def __delitem__(self, key):
        raise ReadOnlyError()

    def __setitem__(self, key, value):
        raise ReadOnlyError()

    def __iter__(self):
        raise Exception('Not implemented')

    def __len__(self):
        raise Exception('Not implemented')

    def _get_zattrs_bytes(self, parent_key: str):
        """Get the attributes of a group or dataset"""
        h5_item = _get_h5_item(self.h5f, parent_key)
        # We create a dummy zarr group and copy the attributes to it. That way
        # we know that zarr has accepted them and they are serialized in the
        # correct format.
        memory_store = MemoryStore()
        dummy_group = zarr.group(store=memory_store)
        for k, v in h5_item.attrs.items():
            v2 = _h5_attr_to_zarr_attr(v, label=f'{parent_key} {k}', h5f=self.h5f)
            if v2 is not None:
                dummy_group.attrs[k] = v2
        if isinstance(h5_item, h5py.Dataset):
            dummy_group.attrs['_ARRAY_DIMENSIONS'] = h5_item.shape
            external_array_link = self._get_external_array_link(parent_key, h5_item)
            if external_array_link is not None:
                dummy_group.attrs['_EXTERNAL_ARRAY_LINK'] = external_array_link
        zattrs_content = memory_store.get('.zattrs')
        if zattrs_content is not None:
            return zattrs_content
        else:
            # No attributes, so we return an empty JSON object
            return '{}'.encode('utf-8')

    def _get_zgroup_bytes(self, parent_key: str):
        """Get the .zgroup JSON text for a group"""
        h5_item = _get_h5_item(self.h5f, parent_key)
        if not isinstance(h5_item, h5py.Group):
            raise Exception(f'Item {parent_key} is not a group')
        # We create a dummy zarr group and then get the .zgroup JSON text
        # from it.
        memory_store = MemoryStore()
        zarr.group(store=memory_store)
        return memory_store.get('.zgroup')

    def _get_zarray_bytes(self, parent_key: str):
        """Get the .zarray JSON text for a dataset"""
        h5_item = _get_h5_item(self.h5f, parent_key)
        if not isinstance(h5_item, h5py.Dataset):
            raise Exception(f'Item {parent_key} is not a dataset')
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
            name='dummy_array',
            shape=info.shape,
            chunks=info.chunks,
            dtype=info.dtype,
            compressor=None,
            order='C',
            fill_value=info.fill_value,
            filters=info.filters,
            object_codec=info.object_codec
        )
        zarray_text = memory_store.get('dummy_array/.zarray')

        return zarray_text

    def _get_chunk_file_bytes(self, key_parent: str, key_name: str):
        byte_offset, byte_count, inline_data = self._get_chunk_file_bytes_data(key_parent, key_name)
        if inline_data is not None:
            return inline_data
        else:
            assert byte_offset is not None
            assert byte_count is not None
            buf = _read_bytes(self.file, byte_offset, byte_count)
            return buf

    def _get_chunk_file_bytes_data(self, key_parent: str, key_name: str):
        h5_item = _get_h5_item(self.h5f, key_parent)
        if not isinstance(h5_item, h5py.Dataset):
            raise Exception(f'Item {key_parent} is not a dataset')

        external_array_link = self._get_external_array_link(key_parent, h5_item)
        if external_array_link is not None:
            raise Exception(f'Chunk file {key_parent}/{key_name} is not present because this is an external array link.')

        # For the case of a scalar dataset, we need to check a few things
        if h5_item.ndim == 0:
            if h5_item.chunks is not None:
                raise Exception(f'Unable to handle case where chunks is not None but ndim is 0 for dataset {key_parent}')
            if key_name != '0':
                raise Exception(f'Chunk name {key_name} does not match dataset dimensions')

        if key_parent in self._inline_data_for_arrays:
            x = self._inline_data_for_arrays[key_parent]
            if isinstance(x, bytes):
                return None, None, x
            elif isinstance(x, str):
                return None, None, x.encode('utf-8')
            else:
                raise Exception(f'Inline data for dataset {key_parent} is not bytes or str. It is {type(x)}')

        if h5_item.ndim == 0:
            raise Exception(f'No inline data for scalar dataset {key_parent}')

        # Get the chunk coords from the file name
        chunk_name_parts = key_name.split('.')
        if len(chunk_name_parts) != h5_item.ndim:
            raise Exception(f'Chunk name {key_name} does not match dataset dimensions')
        chunk_coords = tuple(int(x) for x in chunk_name_parts)
        for i, c in enumerate(chunk_coords):
            if c < 0 or c >= h5_item.shape[i]:
                raise Exception(f'Chunk coordinates {chunk_coords} out of range for dataset {key_parent}')
        if h5_item.chunks is not None:
            # Get the byte range in the file for the chunk. We do it this way
            # rather than reading from the h5py dataset Because we want to
            # ensure that we are reading the exact bytes.
            byte_offset, byte_count = _get_chunk_byte_range(h5_item, chunk_coords)
        else:
            # in this case (contiguous dataset), we need to check that the chunk
            # coordinates are (0, 0, 0, ...)
            if chunk_coords != (0,) * h5_item.ndim:
                raise Exception(f'Chunk coordinates {chunk_coords} are not (0, 0, 0, ...) for contiguous dataset {key_parent}')
            # Get the byte range in the file for the contiguous dataset
            byte_offset, byte_count = _get_byte_range_for_contiguous_dataset(h5_item)
        return byte_offset, byte_count, None

    def _get_external_array_link(self, parent_key: str, h5_item: h5py.Dataset):
        if parent_key in self._external_array_links:
            return self._external_array_links[parent_key]
        self._external_array_links[parent_key] = None  # important to set to None so that we don't keep checking
        if h5_item.chunks and self._num_dataset_chunks_threshold is not None:
            shape = h5_item.shape
            chunks = h5_item.chunks
            chunk_coords_shape = [
                (shape[i] + chunks[i] - 1) // chunks[i]
                for i in range(len(shape))
            ]
            num_chunks = np.prod(chunk_coords_shape)
            if num_chunks > self._num_dataset_chunks_threshold:
                self._external_array_links[parent_key] = {
                    'link_type': 'hdf5_object',
                    'url': self._url,  # url is not going to be null based on the check in __init__
                    'name': parent_key
                }
        return self._external_array_links[parent_key]

    def listdir(self, path: str = "") -> List[str]:
        try:
            item = _get_h5_item(self.h5f, path)
        except KeyError:
            return []
        if isinstance(item, h5py.Group):
            ret = []
            for k in item.keys():
                ret.append(k)
            return ret
        elif isinstance(item, h5py.Dataset):
            ret = []
            return ret
        else:
            return []

    def create_reference_file_system(self):
        if self._url is None:
            raise Exception('You must specify a url to create a reference file system')
        ret = {
            'refs': {},
            'version': 1
        }

        def _add_ref(key: str, content: Union[bytes, None]):
            if content is None:
                raise Exception(f'Unable to get content for key {key}')
            try:
                ret['refs'][key] = content.decode('ascii')
            except UnicodeDecodeError:
                import base64
                # from kerchunk
                ret['refs'][key] = (b"base64:" + base64.b64encode(content)).decode("ascii")

        def _process_group(key, item: h5py.Group):
            if isinstance(item, h5py.Group):
                _add_ref(f'{key}/.zattrs', self.get(f'{key}/.zattrs'))
                _add_ref(f'{key}/.zgroup', self.get(f'{key}/.zgroup'))
                for k in item.keys():
                    subitem = item[k]
                    if isinstance(subitem, h5py.Group):
                        _process_group(f'{key}/{k}', subitem)
                    elif isinstance(subitem, h5py.Dataset):
                        subkey = f'{key}/{k}'
                        zattrs_bytes = self.get(f'{subkey}/.zattrs')
                        assert zattrs_bytes is not None
                        _add_ref(f'{subkey}/.zattrs', zattrs_bytes)
                        zattrs_dict = json.loads(zattrs_bytes.decode('utf-8'))
                        external_array_link = zattrs_dict.get('_EXTERNAL_ARRAY_LINK', None)
                        zarray_bytes = self.get(f'{subkey}/.zarray')
                        assert zarray_bytes is not None
                        _add_ref(f'{subkey}/.zarray', zarray_bytes)
                        zarray_dict = json.loads(zarray_bytes.decode('utf-8'))
                        if external_array_link is None:  # only process chunks for non-external array links
                            shape = zarray_dict['shape']
                            chunks = zarray_dict.get('chunks', None)
                            chunk_coords_shape = [
                                (shape[i] + chunks[i] - 1) // chunks[i]
                                for i in range(len(shape))
                            ]
                            chunk_names = _get_chunk_names_for_dataset(chunk_coords_shape)
                            for chunk_name in chunk_names:
                                byte_offset, byte_count, inline_data = self._get_chunk_file_bytes_data(subkey, chunk_name)
                                if inline_data is not None:
                                    _add_ref(f'{subkey}/{chunk_name}', inline_data)
                                else:
                                    assert byte_offset is not None
                                    assert byte_count is not None
                                    ret['refs'][f'{subkey}/{chunk_name}'] = [
                                        self._url,
                                        byte_offset,
                                        byte_count
                                    ]

        _process_group('', self.h5f)
        return ret


def _get_chunk_names_for_dataset(chunk_coords_shape: List[int]) -> List[str]:
    """Get the chunk names for a dataset with the given chunk coords shape."""
    ndim = len(chunk_coords_shape)
    if ndim == 0:
        return ['0']
    elif ndim == 1:
        return [str(i) for i in range(chunk_coords_shape[0])]
    else:
        names0 = _get_chunk_names_for_dataset(chunk_coords_shape[1:])
        names = []
        for i in range(chunk_coords_shape[0]):
            for name0 in names0:
                names.append(f'{i}.{name0}')
        return names
