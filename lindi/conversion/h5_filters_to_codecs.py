from typing import List, Union
import h5py
import numcodecs
from numcodecs.abc import Codec

# The purpose of _h5_filters_to_codecs it to translate the filters that are
# defined on an HDF5 dataset into numcodecs filters for use with Zarr so that
# the raw data chunks can stay within the the HDF5 file and be read by Zarr
# without having to copy/convert the data.


# This is adapted from _decode_filters from kerchunk source
# https://github.com/fsspec/kerchunk
# Copyright (c) 2020 Intake
# MIT License
def h5_filters_to_codecs(h5obj: h5py.Dataset) -> Union[List[Codec], None]:
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
        elif str(filter_id) == "fletcher32":
            # added by lindi (not in kerchunk) -- required by dandiset 000117
            filters.append(numcodecs.Fletcher32())
        else:
            raise RuntimeError(
                f"{h5obj.name} uses filter id {filter_id} with properties {properties},"
                f" not supported."
            )
    return filters
