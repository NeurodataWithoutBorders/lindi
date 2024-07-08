from typing import Union
from dataclasses import dataclass


@dataclass(frozen=True)
class LindiH5ZarrStoreOpts:
    """
    Options for the LindiH5ZarrStore class.

    Attributes:
        num_dataset_chunks_threshold (Union[int, None]): For each dataset in the
        HDF5 file, if the number of chunks is greater than this threshold, then
        the dataset will be represented as an external array link. Default is 1000.

        single_chunk_size_threshold (Union[int, None]): For each dataset in the
        HDF5 file, if the dataset is a single chunk and the size of the chunk
        is greater than this threshold, then the dataset will be represented as
        an external array link. Default is 1000 * 1000 * 100.
    """
    num_dataset_chunks_threshold: Union[int, None] = 1000
    single_chunk_size_threshold: Union[int, None] = 1000 * 1000 * 100
