from typing import IO
import numpy as np
import h5py


def _read_bytes(file: IO, offset: int, count: int):
    """Read a range of bytes from a file-like object."""
    file.seek(offset)
    return file.read(count)


def _get_chunk_byte_range(h5_dataset: h5py.Dataset, chunk_coords: tuple) -> tuple:
    """Get the byte range in the file for a chunk of an h5py dataset.

    This involves some low-level functions from the h5py library. First we need
    to get the chunk index. Then we call _get_chunk_byte_range_for_chunk_index.
    """
    shape = h5_dataset.shape
    chunk_shape = h5_dataset.chunks
    assert chunk_shape is not None

    chunk_coords_shape = [
        (shape[i] + chunk_shape[i] - 1) // chunk_shape[i]
        for i in range(len(shape))
    ]
    ndim = h5_dataset.ndim
    assert len(chunk_coords) == ndim
    chunk_index = 0
    for i in range(ndim):
        chunk_index += int(chunk_coords[i] * np.prod(chunk_coords_shape[i + 1:]))
    return _get_chunk_byte_range_for_chunk_index(h5_dataset, chunk_index)


def _get_chunk_byte_range_for_chunk_index(h5_dataset: h5py.Dataset, chunk_index: int) -> tuple:
    """Get the byte range in the file for a chunk of an h5py dataset.

    This involves some low-level functions from the h5py library.
    """
    # got hints from kerchunk source code
    dsid = h5_dataset.id
    chunk_info = dsid.get_chunk_info(chunk_index)
    byte_offset = chunk_info.byte_offset
    byte_count = chunk_info.size
    return byte_offset, byte_count


def _get_byte_range_for_contiguous_dataset(h5_dataset: h5py.Dataset) -> tuple:
    """Get the byte range in the file for a contiguous dataset.

    This is the case where no chunking is used. Then all the data is stored
    contiguously in the file.
    """
    # got hints from kerchunk source code
    dsid = h5_dataset.id
    byte_offset = dsid.get_offset()
    byte_count = dsid.get_storage_size()
    return byte_offset, byte_count
