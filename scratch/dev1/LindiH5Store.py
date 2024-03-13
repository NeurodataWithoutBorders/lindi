from typing import Union, List, IO
import json
import numpy as np
import zarr
import numcodecs
from numcodecs.abc import Codec
from zarr.storage import Store, MemoryStore
import h5py
from zarr.errors import ReadOnlyError


class LindiH5Store(Store):
    def __init__(self, file: IO):
        self.file = file
        self.h5f = h5py.File(file, 'r')

    def __getitem__(self, key):
        parts = key.split('/')
        if len(parts) == 0:
            raise KeyError(key)
        last_part = parts[-1]
        parent_key = '/'.join(parts[:-1])
        if last_part == '.zattrs':
            h5_item = _get_h5_item(self.h5f, parent_key)
            memory_store = MemoryStore()
            dummy_group = zarr.group(store=memory_store)
            for k, v in h5_item.attrs.items():
                dummy_group.attrs[k] = _attr_h5_to_zarr(v)
            return memory_store.get('.zattrs')
        elif last_part == '.zgroup':
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Group):
                raise Exception(f'Item {parent_key} is not a group')
            zgroup = {
                "zarr_format": 2
            }
            return json.dumps(zgroup)
        elif last_part == '.zarray':
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Dataset):
                return ''
            shape, chunks, dtype, filters = _h5_dataset_to_zarr_info(h5_item)
            memory_store = MemoryStore()
            dummy_group = zarr.group(store=memory_store)
            dummy_group.create_dataset(
                name='dummy_array',
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=None,
                order='C',
                filters=filters
            )
            zarray_text = memory_store.get('dummy_array/.zarray')
            return zarray_text
        else:
            # Assume it's a chunk file
            h5_item = _get_h5_item(self.h5f, parent_key)
            if not isinstance(h5_item, h5py.Dataset):
                raise Exception(f'Item {parent_key} is not a dataset')
            chunk_name_parts = last_part.split('.')
            if len(chunk_name_parts) != h5_item.ndim:
                raise Exception(f'Chunk name {last_part} does not match dataset dimensions')
            chunk_coords = tuple(int(x) for x in chunk_name_parts)
            for i, c in enumerate(chunk_coords):
                if c < 0 or c >= h5_item.shape[i]:
                    raise Exception(f'Chunk coordinates {chunk_coords} out of range for dataset {parent_key}')
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
            chunk_name_parts = last_part.split('.')
            if len(chunk_name_parts) != h5_item.ndim:
                return False
            chunk_coords = tuple(int(x) for x in chunk_name_parts)
            for i, c in enumerate(chunk_coords):
                if c < 0 or c >= h5_item.shape[i]:
                    return False
            return True

    def keys2(self):
        # I think visititems is inefficient on h5py - not sure though
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

    def getsize(self, path):
        raise NotImplementedError()

    def listdir(self, path=None):
        item = _get_h5_item(self.h5f, path)
        if not isinstance(item, h5py.Group):
            raise Exception(f'Item {path} is not a group')
        return list(item.keys())

    def rmdir(self, path: str = "") -> None:
        raise ReadOnlyError()


def _get_h5_item(h5f: h5py.File, key: str):
    return h5f['/' + key]


def _attr_h5_to_zarr(attr):
    if isinstance(attr, bytes):
        return attr.decode('utf-8')  # is this reversible?
    elif isinstance(attr, (int, float, str)):
        return attr
    elif isinstance(attr, list):
        return [_attr_h5_to_zarr(x) for x in attr]
    elif isinstance(attr, dict):
        return {k: _attr_h5_to_zarr(v) for k, v in attr.items()}
    else:
        raise NotImplementedError()


def _h5_dataset_to_zarr_info(h5_dataset: h5py.Dataset):
    shape = h5_dataset.shape
    chunks = h5_dataset.chunks
    dtype = h5_dataset.dtype
    filters = _decode_filters(h5_dataset)
    return shape, chunks, dtype, filters


def _read_bytes(file: IO, offset: int, count: int):
    # file is a file-like object
    file.seek(offset)
    return file.read(count)


# This _decode_filters adapted from kerchunk source code
# https://github.com/fsspec/kerchunk
# Copyright (c) 2020 Intake
# MIT License
def _decode_filters(h5obj: h5py.Dataset) -> Union[List[Codec], None]:
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
    shape = h5_dataset.shape
    chunk_shape = h5_dataset.chunks
    assert chunk_shape is not None
    chunk_coords_shape = [shape[i] // chunk_shape[i] for i in range(len(shape))]
    ndim = h5_dataset.ndim
    assert len(chunk_coords) == ndim
    chunk_index = 0
    for i in range(ndim):
        chunk_index += chunk_coords[i] * np.prod(chunk_coords_shape[i + 1:])
    # got hints from kerchunk source code
    dsid = h5_dataset.id
    chunk_info = dsid.get_chunk_info(chunk_index)
    byte_offset = chunk_info.byte_offset
    byte_count = chunk_info.size
    return byte_offset, byte_count


def test_lindi_h5_store():
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
