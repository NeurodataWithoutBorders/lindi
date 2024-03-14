import json
import struct
from typing import Union, List, IO, Any
import numpy as np
import zarr
import numcodecs
from numcodecs.abc import Codec
from zarr.storage import Store, MemoryStore
import h5py
from zarr.errors import ReadOnlyError


class LindiH5Store(Store):
    """A zarr store that reads from an HDF5 file using h5py.

    This store is read-only.

    Parameters
    ----------
    file : file-like object
        The HDF5 file to read from (not h5py.File object, but a file-like object, e.g. a file opened with open()).
    """
    def __init__(self, file: Union[IO, Any]):
        self.file = file
        self.h5f = h5py.File(file, 'r')
        self._inline_data_for_arrays = {}

    def __getitem__(self, key):
        """Get an item from the store (required by zarr.Store)"""
        parts = key.split('/')
        if len(parts) == 0:
            raise KeyError(key)
        last_part = parts[-1]
        parent_key = '/'.join(parts[:-1])
        if last_part == '.zattrs':
            """Get the attributes of a group or dataset"""
            h5_item = _get_h5_item(self.h5f, parent_key)
            # We create a dummy zarr group and copy the attributes to it That
            #   way we know that zarr has accepted them and they are serialized
            #   in the correct format
            memory_store = MemoryStore()
            dummy_group = zarr.group(store=memory_store)
            for k, v in h5_item.attrs.items():
                v2 = _attr_h5_to_zarr(v, label=f'{parent_key} {k}')
                if v2 is not None:
                    dummy_group.attrs[k] = v2
            if isinstance(h5_item, h5py.Dataset):
                dummy_group.attrs['_ARRAY_DIMENSIONS'] = h5_item.shape
            zattrs_text = memory_store.get('.zattrs')
            if zattrs_text is not None:
                return zattrs_text
            else:
                # No attributes, so we return an empty JSON object
                return '{}'.encode('utf-8')
        elif last_part == '.zgroup':
            """Get the .zgroup JSON text for a group"""
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Group):
                raise Exception(f'Item {parent_key} is not a group')
            # We create a dummy zarr group and then get the .zgroup JSON text
            #   from it
            memory_store = MemoryStore()
            dummy_group = zarr.group(store=memory_store)
            return memory_store.get('.zgroup')
        elif last_part == '.zarray':
            """Get the .zarray JSON text for a dataset"""
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Dataset):
                return ''
            # get the shape, chunks, dtype, and filters from the h5 dataset
            shape, chunks, dtype, filters, fill_value, object_codec, inline_data = _h5_dataset_to_zarr_info(h5_item)
            if inline_data is not None:
                self._inline_data_for_arrays[parent_key] = inline_data
            # We create a dummy zarr dataset with the appropriate shape, chunks,
            #   dtype, and filters and then copy the .zarray JSON text from it
            memory_store = MemoryStore()
            dummy_group = zarr.group(store=memory_store)
            dummy_group.create_dataset(
                name='dummy_array',
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=None,
                order='C',
                fill_value=fill_value,
                filters=filters,
                object_codec=object_codec
            )
            zarray_text = memory_store.get('dummy_array/.zarray')
            return zarray_text
        else:
            # Otherwise, we assume it is a chunk file
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Dataset):
                raise Exception(f'Item {parent_key} is not a dataset')

            # handle the case of ndim = 0
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
            #   rather than reading from the h5py dataset Because we want to
            #   ensure that we are reading the exact bytes
            byte_offset, byte_count = _get_chunk_byte_range(h5_item, chunk_coords)
            buf = _read_bytes(self.file, byte_offset, byte_count)
            return buf

    def __contains__(self, key):
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
        # I think visititems is inefficient on h5py - not sure though
        #   That's why I'm using this approach
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


def _h5_dataset_to_zarr_info(h5_dataset: h5py.Dataset):
    """Get the shape, chunks, dtype, and filters from an h5py dataset."""
    # This function needs to be expanded a lot to handle all the possible
    #   cases
    shape = h5_dataset.shape
    chunks = h5_dataset.chunks
    dtype = h5_dataset.dtype
    filters = []
    fill_value = None
    object_codec = None
    inline_data = None

    if len(shape) == 0:
        # scalar dataset
        value = h5_dataset[()]
        # zarr doesn't support scalar datasets, so we make an array of shape (1,)
        shape = (1,)
        chunks = (1,)
        if dtype == np.int8:
            inline_data = struct.pack('<b', value)
            fill_value = 0
        elif dtype == np.uint8:
            inline_data = struct.pack('<B', value)
            fill_value = 0
        elif dtype == np.int16:
            inline_data = struct.pack('<h', value)
            fill_value = 0
        elif dtype == np.uint16:
            inline_data = struct.pack('<H', value)
            fill_value = 0
        elif dtype == np.int32:
            inline_data = struct.pack('<i', value)
            fill_value = 0
        elif dtype == np.uint32:
            inline_data = struct.pack('<I', value)
            fill_value = 0
        elif dtype == np.int64:
            inline_data = struct.pack('<q', value)
            fill_value = 0
        elif dtype == np.uint64:
            inline_data = struct.pack('<Q', value)
            fill_value = 0
        elif dtype == np.float32:
            inline_data = struct.pack('<f', value)
            fill_value = 0
        elif dtype == np.float64:
            inline_data = struct.pack('<d', value)
            fill_value = 0
        elif dtype == object:
            if isinstance(value, str):
                object_codec = numcodecs.JSON()
                inline_data = json.dumps([value, '|O', [1]])
                fill_value = ' '
            elif isinstance(value, bytes):
                object_codec = numcodecs.JSON()
                inline_data = json.dumps([value.decode('utf-8'), '|O', [1]])
                fill_value = ' '
            else:
                raise Exception(f'Cannot handle scalar dataset {h5_dataset.name} with dtype object and value {value} of type {type(value)}')
        else:
            raise Exception(f'Cannot handle scalar dataset {h5_dataset.name} with dtype {dtype}')
    else:
        filters = _decode_filters(h5_dataset)

        if dtype.kind in 'SU':  # byte string or unicode string
            fill_value = h5_dataset.fillvalue or ' '  # this is from kerchunk code
        elif dtype.kind == 'O':
            # This is the kerchunk "embed" case
            object_codec = numcodecs.JSON()
            if np.isscalar(h5_dataset):
                data = str(h5_dataset)
            elif h5_dataset.ndim == 0:
                # data = np.array(h5_dataset).tolist().decode()  # this is from kerchunk, but I don't know what it's doing
                a = np.array(h5_dataset).tolist()
                if isinstance(a, bytes):
                    data = a.decode()
                else:
                    raise Exception(f'Cannot handle dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')
            else:
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
    return shape, chunks, dtype, filters, fill_value, object_codec, inline_data


def _read_bytes(file: IO, offset: int, count: int):
    """Read a range of bytes from a file-like object."""
    file.seek(offset)
    return file.read(count)


# This _decode_filters adapted from kerchunk source code
# https://github.com/fsspec/kerchunk
# Copyright (c) 2020 Intake
# MIT License
def _decode_filters(h5obj: h5py.Dataset) -> Union[List[Codec], None]:
    """Decode HDF5 filters to numcodecs filters."""
    if h5obj.scaleoffset:
        raise RuntimeError(
            f"{h5obj.name} uses HDF5 scaleoffset filter - not supported"
        )
    if h5obj.compression in ("szip", "lzf"):
        raise RuntimeError(
            f"{h5obj.name} uses szip or lzf compression - not supported"
        )
    filters = []
    if h5obj.shuffle and h5obj.dtype.kind != "O":
        # cannot use shuffle if we materialised objects
        filters.append(numcodecs.Shuffle(elementsize=h5obj.dtype.itemsize))
    for filter_id, properties in h5obj._filters.items():
        if str(filter_id) == "32001":
            blosc_compressors = (
                "blosclz",
                "lz4",
                "lz4hc",
                "snappy",
                "zlib",
                "zstd",
            )
            (
                _1,
                _2,
                bytes_per_num,
                total_bytes,
                clevel,
                shuffle,
                compressor,
            ) = properties
            pars = dict(
                blocksize=total_bytes,
                clevel=clevel,
                shuffle=shuffle,
                cname=blosc_compressors[compressor],
            )
            filters.append(numcodecs.Blosc(**pars))
        elif str(filter_id) == "32015":
            filters.append(numcodecs.Zstd(level=properties[0]))
        elif str(filter_id) == "gzip":
            filters.append(numcodecs.Zlib(level=properties))
        elif str(filter_id) == "32004":
            raise RuntimeError(
                f"{h5obj.name} uses lz4 compression - not supported"
            )
        elif str(filter_id) == "32008":
            raise RuntimeError(
                f"{h5obj.name} uses bitshuffle compression - not supported"
            )
        elif str(filter_id) == "shuffle":
            # already handled before this loop
            pass
        else:
            raise RuntimeError(
                f"{h5obj.name} uses filter id {filter_id} with properties {properties},"
                f" not supported."
            )
    return filters


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


def test_lindi_h5_store():
    """Test the LindiH5Store class."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_h5_fname = f'{tmpdir}/test.h5'
        with h5py.File(test_h5_fname, 'w') as h5f:
            h5f.create_dataset('data', data=np.arange(100).reshape(10, 10), chunks=(5, 5))
        with open(test_h5_fname, 'rb') as f:
            h5f = h5py.File(f, 'r')
            store = LindiH5Store(f)
            root = zarr.open_group(store)
            data = root['data']
            A1x = h5f['data']
            assert isinstance(A1x, h5py.Dataset)
            A1 = A1x[:]
            A2 = data[:]
            assert isinstance(A2, np.ndarray)
            assert np.array_equal(A1, A2)


if __name__ == '__main__':
    test_lindi_h5_store()
