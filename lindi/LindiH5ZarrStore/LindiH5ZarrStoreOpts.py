from typing import Union
from dataclasses import dataclass


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
