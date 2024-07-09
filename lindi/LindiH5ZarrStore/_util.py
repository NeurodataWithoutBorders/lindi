from typing import IO, List, Callable
import json
import numpy as np
import h5py
import math
import warnings


def _read_bytes(file: IO, offset: int, count: int):
    """Read a range of bytes from a file-like object."""
    file.seek(offset)
    return file.read(count)


def _get_max_num_chunks(*, shape, chunk_size):
    """Get the maximum number of chunks in an h5py dataset.

    This is similar to h5_dataset.id.get_num_chunks() but significantly faster. It does not account for
    whether some chunks are allocated.
    """
    assert chunk_size is not None
    if np.prod(chunk_size) == 0:
        return 0
    return math.prod([math.ceil(a / b) for a, b in zip(shape, chunk_size)])


def _apply_to_all_chunk_info(h5_dataset: h5py.Dataset, callback: Callable):
    """Apply the callback function to each chunk of an h5py dataset.
    The chunks are iterated in order such that the last dimension changes the fastest,
    e.g., chunk coordinates could be:
    [0, 0, 0], [0, 0, 1], [0, 0, 2], ..., [0, 1, 0], [0, 1, 1], [0, 1, 2], ..., [1, 0, 0], [1, 0, 1], [1, 0, 2], ...

    This method tries to use the `chunk_iter` method if it is available. The `chunk_iter` method requires
    HDF5 1.12.3 and above. If it is not available, this method falls back to the `get_chunk_info` method,
    which is significantly slower and not recommended if the dataset has many chunks.

    `chunk_iter` takes 1-5 seconds for all chunks for a dataset with 1e6 chunks.
    `get_chunk_info` takes about 0.2 seconds per chunk for a dataset with 1e6 chunks.

    NOTE: This method might be very slow if the dataset is stored remotely.
    """
    assert h5_dataset.chunks is not None
    dsid = h5_dataset.id
    try:
        dsid.chunk_iter(callback)
    except AttributeError:
        # chunk_iter is not available
        num_chunks = dsid.get_num_chunks()  # NOTE: this can be slow for remote datasets with many chunks
        if num_chunks > 100:
            warnings.warn(
                f"Dataset {h5_dataset.name} has {num_chunks} chunks. Using get_chunk_info is slow. "
                f"Consider upgrading to HDF5 1.12.3 or above for faster performance."
            )
        for index in range(num_chunks):
            chunk_info = dsid.get_chunk_info(index)
            callback(chunk_info)


def _get_chunk_byte_range(h5_dataset: h5py.Dataset, chunk_coords: tuple) -> tuple:
    """Get the byte range in the file for a chunk of an h5py dataset.

    This involves some low-level functions from the h5py library. First we need
    to get the chunk index. Then we call _get_chunk_byte_range_for_chunk_index.
    """
    shape = h5_dataset.shape
    chunk_shape = h5_dataset.chunks
    assert chunk_shape is not None

    chunk_coords_shape = [
        # the shape could be zero -- for example dandiset 000559 - acquisition/depth_video/data has shape [0, 0, 0]
        (shape[i] + chunk_shape[i] - 1) // chunk_shape[i] if chunk_shape[i] != 0 else 0
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

    This involves some low-level functions from the h5py library. Use _apply_to_all_chunk_info instead of
    calling this repeatedly for many chunks of the same dataset.
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


def _join(a: str, b: str) -> str:
    if a == "":
        return b
    else:
        return f"{a}/{b}"


# NOTE: this is no longer used
def _get_chunk_names_for_dataset(chunk_coords_shape: List[int]) -> List[str]:
    """Get the chunk names for a dataset with the given chunk coords shape.

    For example: _get_chunk_names_for_dataset([1, 2, 3]) returns
    ['0.0.0', '0.0.1', '0.0.2', '0.1.0', '0.1.1', '0.1.2']
    """
    ndim = len(chunk_coords_shape)
    if ndim == 0:
        return ["0"]
    elif ndim == 1:
        return [str(i) for i in range(chunk_coords_shape[0])]
    else:
        names0 = _get_chunk_names_for_dataset(chunk_coords_shape[1:])
        names = []
        for i in range(chunk_coords_shape[0]):
            for name0 in names0:
                names.append(f"{i}.{name0}")
        return names


def _write_rfs_to_file(*, rfs: dict, output_file_name: str):
    """Write a reference file system to a file.
    """
    with open(output_file_name, "w") as f:
        json.dump(rfs, f, indent=2, sort_keys=True)
