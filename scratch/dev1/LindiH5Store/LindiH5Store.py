import json
import struct
from typing import Union, List, IO, Any, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import zarr
import numcodecs
from zarr.storage import Store, MemoryStore
import h5py
from zarr.errors import ReadOnlyError
from numcodecs.abc import Codec
from ._h5_filters_to_codecs import _h5_filters_to_codecs


class LindiH5Store(Store):
    """A zarr store that provides a read-only view of an HDF5 file.

    Parameters
    ----------
    file : file-like object
        The HDF5 file to read from (not h5py.File object, but a file-like
        object, e.g. a file opened with open()). The reason for this is that we
        read chunks directly from the file rather than using h5py.
    """
    def __init__(self, file: Union[IO, Any]):
        self.file = file
        self.h5f = h5py.File(file, 'r')

        # Some datasets do not correspond to traditional chunked datasets. For
        # those datasets, we need to store the inline data so that we can
        # return it when the chunk is requested. We store the inline data in a
        # dictionary with the dataset name as the key. The values are of type
        # bytes.
        self._inline_data_for_arrays: Dict[str, bytes] = {}

    def __getitem__(self, key):
        """Get an item from the store (required by base class)."""
        parts = key.split('/')
        if len(parts) == 0:
            raise KeyError(key)
        last_part = parts[-1]
        parent_key = '/'.join(parts[:-1])
        if last_part == '.zattrs':
            """Get the attributes of a group or dataset"""
            h5_item = _get_h5_item(self.h5f, parent_key)
            # We create a dummy zarr group and copy the attributes to it. That
            # way we know that zarr has accepted them and they are serialized in
            # the correct format.
            memory_store = MemoryStore()
            dummy_group = zarr.group(store=memory_store)
            for k, v in h5_item.attrs.items():
                v2 = _attr_h5_to_zarr(v, label=f'{parent_key} {k}')
                if v2 is not None:
                    dummy_group.attrs[k] = v2
            if isinstance(h5_item, h5py.Dataset):
                dummy_group.attrs['_ARRAY_DIMENSIONS'] = h5_item.shape
            zattrs_content = memory_store.get('.zattrs')
            if zattrs_content is not None:
                return zattrs_content
            else:
                # No attributes, so we return an empty JSON object
                return '{}'.encode('utf-8')
        elif last_part == '.zgroup':
            """Get the .zgroup JSON text for a group"""
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Group):
                raise Exception(f'Item {parent_key} is not a group')
            # We create a dummy zarr group and then get the .zgroup JSON text
            # from it.
            memory_store = MemoryStore()
            dummy_group = zarr.group(store=memory_store)
            return memory_store.get('.zgroup')
        elif last_part == '.zarray':
            """Get the .zarray JSON text for a dataset"""
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Dataset):
                return ''
            # get the shape, chunks, dtype, and filters from the h5 dataset
            info = _h5_dataset_to_zarr_info(h5_item)
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
        else:
            # Otherwise, we assume it is a chunk file
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Dataset):
                raise Exception(f'Item {parent_key} is not a dataset')

            # handle the case of ndim = 0 (scalar dataset)
            if h5_item.ndim == 0:
                if h5_item.chunks is not None:
                    raise Exception(f'Unable to handle case where chunks is not None but ndim is 0 for dataset {parent_key}')
                if last_part != '0':
                    raise Exception(f'Chunk name {last_part} does not match dataset dimensions')
                if parent_key in self._inline_data_for_arrays:
                    x = self._inline_data_for_arrays[parent_key]
                    if isinstance(x, bytes):
                        return x
                    elif isinstance(x, str):
                        return x.encode('utf-8')
                    else:
                        raise Exception(f'Inline data for dataset {parent_key} is not bytes or str. It is {type(x)}')
                byte_offset, byte_count = _get_byte_range_for_contiguous_dataset(h5_item)
                buf = _read_bytes(self.file, byte_offset, byte_count)
                return buf

            # Get the chunk coords from the file name
            chunk_name_parts = last_part.split('.')
            if len(chunk_name_parts) != h5_item.ndim:
                raise Exception(f'Chunk name {last_part} does not match dataset dimensions')
            chunk_coords = tuple(int(x) for x in chunk_name_parts)
            for i, c in enumerate(chunk_coords):
                if c < 0 or c >= h5_item.shape[i]:
                    raise Exception(f'Chunk coordinates {chunk_coords} out of range for dataset {parent_key}')
            # Get the byte range in the file for the chunk We do it this way
            # rather than reading from the h5py dataset Because we want to
            # ensure that we are reading the exact bytes
            byte_offset, byte_count = _get_chunk_byte_range(h5_item, chunk_coords)
            print(f'For chunk {key}, byte_offset={byte_offset}, byte_count={byte_count}')
            buf = _read_bytes(self.file, byte_offset, byte_count)
            return buf

    def __contains__(self, key):
        """Check if a key is in the store (used by zarr)."""
        # it would be nice if we didn't have to repeat the logic from __getitem__
        parts = key.split('/')
        if len(parts) == 0:
            return False
        last_part = parts[-1]
        parent_key = '/'.join(parts[:-1])
        if last_part == '.zattrs':
            h5_item = _get_h5_item(self.h5f, parent_key)
            return isinstance(h5_item, h5py.Group)
        elif last_part == '.zgroup':
            h5_item = _get_h5_item(self.h5f, parent_key)
            return isinstance(h5_item, h5py.Group)
        elif last_part == '.zarray':
            h5_item = _get_h5_item(self.h5f, parent_key)
            return isinstance(h5_item, h5py.Dataset)
        else:
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Dataset):
                return False
            if h5_item.ndim == 0:
                return last_part == '0'
            chunk_name_parts = last_part.split('.')
            if len(chunk_name_parts) != h5_item.ndim:
                return False
            chunk_coords = tuple(int(x) for x in chunk_name_parts)
            for i, c in enumerate(chunk_coords):
                if c < 0 or c >= h5_item.shape[i]:
                    return False
            return True

    # We use keys2 instead of keys because of linter complaining
    def keys2(self):
        # I think visititems is inefficient on h5py - not sure though. That's
        # why I'm using this approach
        stack: List[str] = []
        stack.append('')
        while len(stack) > 0:
            current_key = stack.pop()
            item = _get_h5_item(self.h5f, current_key)
            if isinstance(item, h5py.Group):
                for k in item.keys():
                    stack.append(f'{current_key}/{k}')
            yield current_key

    def __delitem__(self, key):
        raise ReadOnlyError()

    def __setitem__(self, key, value):
        raise ReadOnlyError()

    def __iter__(self):
        return self.keys2()

    def __len__(self):
        """Get the number of items in the store (used by zarr)."""
        # Not sure why this is needed. It seems unfortunate because this could
        # be time-consuming. However, it may only be called in certain
        # circumstances.
        return sum(1 for _ in self.keys2())


def _get_h5_item(h5f: h5py.File, key: str):
    """Get an item from the h5 file, given its key."""
    return h5f['/' + key]


def _attr_h5_to_zarr(attr, *, label: str = ''):
    """Convert an attribute from h5py to a format that zarr can accept."""
    if isinstance(attr, bytes):
        return attr.decode('utf-8')  # is this reversible?
    elif isinstance(attr, (int, float, str)):
        return attr
    elif isinstance(attr, list):
        return [_attr_h5_to_zarr(x) for x in attr]
    elif isinstance(attr, dict):
        return {k: _attr_h5_to_zarr(v) for k, v in attr.items()}
    elif isinstance(attr, h5py.Reference):
        print(f'Warning: attribute of type h5py.Reference not handled: {label}')
        return None
    else:
        print(f'Warning: attribute of type {type(attr)} not handled: {label}')
        raise NotImplementedError()


@dataclass
class DatasetZarrInfo:
    shape: Tuple[int]
    chunks: Union[None, Tuple[int]]
    dtype: Any
    filters: Union[List[Codec], None]
    fill_value: Any
    object_codec: Union[None, Codec]
    inline_data: Union[bytes, None]


def _h5_dataset_to_zarr_info(h5_dataset: h5py.Dataset) -> DatasetZarrInfo:
    """Get the shape, chunks, dtype, and filters from an h5py dataset."""
    shape = h5_dataset.shape
    dtype = h5_dataset.dtype

    if len(shape) == 0:
        # scalar dataset
        value = h5_dataset[()]
        # zarr doesn't support scalar datasets, so we make an array of shape (1,)
        # and the _ARRAY_DIMENSIONS attribute will be set to [] to indicate that
        # it is a scalar dataset

        # Let's handle all the possible types explicitly
        numeric_format_str = _get_numeric_format_str(dtype)
        if numeric_format_str is not None:
            # Handle the simple numeric types
            inline_data = struct.pack(numeric_format_str, value)
            return DatasetZarrInfo(
                shape=(1,),
                chunks=None,
                dtype=dtype,
                filters=None,
                fill_value=0,
                object_codec=None,
                inline_data=inline_data
            )
        elif dtype == object:
            # For type object, we are going to use the JSON codec
            # which requires inline data of the form [[val], '|O', [1]]
            if isinstance(value, (bytes, str)):
                if isinstance(value, bytes):
                    value = value.decode()
                return DatasetZarrInfo(
                    shape=(1,),
                    chunks=None,
                    dtype=dtype,
                    filters=None,
                    fill_value=' ',
                    object_codec=numcodecs.JSON(),
                    inline_data=json.dumps([value, '|O', [1]]).encode('utf-8')
                )
            else:
                raise Exception(f'Not yet implemented (1): object scalar dataset with value {value} and dtype {dtype}')
        else:
            raise Exception(f'Cannot handle scalar dataset {h5_dataset.name} with dtype {dtype}')
    else:
        # not a scalar dataset
        if dtype.kind in 'SU':  # byte string or unicode string
            raise Exception(f'Not yet implemented (2): dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')
        elif dtype.kind == 'O':
            # For type object, we are going to use the JSON codec
            # which requires inline data of the form [[some, nested, array], '|O', [n1, n2, ...]]
            object_codec = numcodecs.JSON()
            data = h5_dataset[:]
            data_vec_view = data.ravel()
            for i, val in enumerate(data_vec_view):
                if isinstance(val, bytes):
                    data_vec_view[i] = val.decode()
                elif isinstance(val, str):
                    data_vec_view[i] = val
                elif isinstance(val, h5py.h5r.Reference):
                    print(f'Warning: reference in dataset {h5_dataset.name} not handled')
                    data_vec_view[i] = None
                else:
                    raise Exception(f'Cannot handle dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')
            inline_data = json.dumps([data.tolist(), '|O', list(shape)]).encode('utf-8')
            return DatasetZarrInfo(
                shape=shape,
                chunks=None,
                dtype=dtype,
                filters=None,
                fill_value=' ',  # not sure what to put here
                object_codec=object_codec,
                inline_data=inline_data
            )
        elif dtype.kind in ['i', 'u', 'f']:  # integer, unsigned integer, float
            # This is the normal case of a chunked dataset with a numeric dtype
            filters = _h5_filters_to_codecs(h5_dataset)
            return DatasetZarrInfo(
                shape=shape,
                chunks=h5_dataset.chunks,
                dtype=dtype,
                filters=filters,
                fill_value=h5_dataset.fillvalue,
                object_codec=None,
                inline_data=None
            )
        else:
            raise Exception(f'Not yet implemented (3): dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')


def _get_numeric_format_str(dtype: Any) -> Union[str, None]:
    """Get the format string for a numeric dtype."""
    if dtype == np.int8:
        return '<b'
    elif dtype == np.uint8:
        return '<B'
    elif dtype == np.int16:
        return '<h'
    elif dtype == np.uint16:
        return '<H'
    elif dtype == np.int32:
        return '<i'
    elif dtype == np.uint32:
        return '<I'
    elif dtype == np.int64:
        return '<q'
    elif dtype == np.uint64:
        return '<Q'
    elif dtype == np.float32:
        return '<f'
    elif dtype == np.float64:
        return '<d'
    else:
        return None


def _read_bytes(file: IO, offset: int, count: int):
    """Read a range of bytes from a file-like object."""
    file.seek(offset)
    return file.read(count)


def _get_chunk_byte_range(h5_dataset: h5py.Dataset, chunk_coords: tuple) -> tuple:
    """Get the byte range in the file for a chunk of an h5py dataset."""
    shape = h5_dataset.shape
    chunk_shape = h5_dataset.chunks
    assert chunk_shape is not None
    chunk_coords_shape = [shape[i] // chunk_shape[i] for i in range(len(shape))]
    ndim = h5_dataset.ndim
    assert len(chunk_coords) == ndim
    chunk_index = 0
    for i in range(ndim):
        chunk_index += chunk_coords[i] * np.prod(chunk_coords_shape[i + 1:])
    return _get_chunk_byte_range_for_chunk_index(h5_dataset, chunk_index)


def _get_chunk_byte_range_for_chunk_index(h5_dataset: h5py.Dataset, chunk_index: int) -> tuple:
    # got hints from kerchunk source code
    dsid = h5_dataset.id
    chunk_info = dsid.get_chunk_info(chunk_index)
    byte_offset = chunk_info.byte_offset
    byte_count = chunk_info.size
    return byte_offset, byte_count


def _get_byte_range_for_contiguous_dataset(h5_dataset: h5py.Dataset) -> tuple:
    """Get the byte range in the file for a contiguous dataset."""
    # got hints from kerchunk source code
    dsid = h5_dataset.id
    byte_offset = dsid.get_offset()
    byte_count = dsid.get_storage_size()
    return byte_offset, byte_count
