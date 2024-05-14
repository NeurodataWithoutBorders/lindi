from typing import IO, List, Union
import json
import numpy as np
import h5py


def _read_bytes(file: IO, offset: int, count: int):
    """Read a range of bytes from a file-like object."""
    file.seek(offset)
    return file.read(count)


def _get_all_chunk_info(h5_dataset: h5py.Dataset) -> Union[list, None]:
    """Get the chunk info for all the chunks of an h5py dataset as a list of StoreInfo objects.
    The chunks are in order such that the last dimension changes the fastest, e.g., chunk coordinates could be:
    [0, 0, 0], [0, 0, 1], [0, 0, 2], ..., [0, 1, 0], [0, 1, 1], [0, 1, 2], ..., [1, 0, 0], [1, 0, 1], [1, 0, 2], ...

    Use stinfo[i].byte_offset and stinfo[i].size to get the byte range in the file for the i-th chunk.

    Requires HDF5 1.12.3 and above. If the chunk_iter method is not available, return None.

    This takes 1-5 seconds for a dataset with 1e6 chunks.

    This might be very slow if the dataset is stored remotely.
    """
    stinfo = list()
    dsid = h5_dataset.id
    try:
        dsid.chunk_iter(stinfo.append)
    except AttributeError:
        # chunk_iter is not available
        return None
    return stinfo


def _get_chunk_index(h5_dataset: h5py.Dataset, chunk_coords: tuple) -> int:
    """Get the chunk index for a chunk of an h5py dataset.

    This involves some low-level functions from the h5py library.
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
    return chunk_index


def _get_chunk_byte_range(h5_dataset: h5py.Dataset, chunk_coords: tuple) -> tuple:
    """Get the byte range in the file for a chunk of an h5py dataset.

    This involves some low-level functions from the h5py library. First we need
    to get the chunk index. Then we call _get_chunk_byte_range_for_chunk_index.
    """
    chunk_index = _get_chunk_index(h5_dataset, chunk_coords)
    return _get_chunk_byte_range_for_chunk_index(h5_dataset, chunk_index)


def _get_chunk_byte_range_for_chunk_index(h5_dataset: h5py.Dataset, chunk_index: int) -> tuple:
    """Get the byte range in the file for a chunk of an h5py dataset.

    This involves some low-level functions from the h5py library. Use _get_all_chunk_info instead of
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
