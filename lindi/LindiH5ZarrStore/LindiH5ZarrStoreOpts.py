from typing import Union
from dataclasses import dataclass


@dataclass(frozen=True)
class LindiH5ZarrStoreOpts:
    """
    Options for the LindiH5ZarrStore class.

    Attributes:
        num_dataset_chunks_threshold (Union[int, None]): For each dataset in the
        HDF5 file, if the number of chunks is greater than this threshold, then
        the dataset will be represented as an external array link. If None, then
        no datasets will be represented as external array links (equivalent to a
        threshold of 0). Default is 1000.

        contiguous_dataset_max_chunk_size (Union[int, None]): For large
        contiguous arrays in the hdf5 file that are not chunked, this option
        specifies the maximum size in bytes of the zarr chunks that will be
        created. If None, then the entire array will be represented as a single
        chunk. Default is 1000 * 1000 * 20
    """
    num_dataset_chunks_threshold: Union[int, None] = 1000
    contiguous_dataset_max_chunk_size: Union[int, None] = 1000 * 1000 * 20
