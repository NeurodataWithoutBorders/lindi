# LINDI - Linked Data Interface

[![latest-release](https://img.shields.io/pypi/v/lindi.svg)](https://pypi.org/project/lindi)
![tests](https://github.com/neurodatawithoutborders/lindi/actions/workflows/integration_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/neurodatawithoutborders/lindi/branch/main/graph/badge.svg?token=xxxx)](https://codecov.io/gh/neurodatawithoutborders/lindi)

:warning: Please note, LINDI is currently under development and should not yet be used in practice.

LINDI is a Python library that facilitates handling NWB (Neurodata Without Borders) files in an efficient, flexible manner, especially when dealing with large datasets on remote servers. The goal is to enable composition of NWB files by integrating data from multiple sources without the need to copy or move large datasets.

LINDI features include:

- A specification for representing arbitrary HDF5 files as Zarr stores. This handles scalar datasets, references, soft links, and compound data types for datasets.
- A Zarr wrapper for remote or local HDF5 files (LindiH5ZarrStore). This involves pointers to remote files for remote data chunks.
- A function for generating a reference file system .zarr.json file from a Zarr store. This is inspired by [kerchunk](https://github.com/fsspec/kerchunk).
- An h5py-like interface for accessing these Zarr stores that can be used with [pynwb](https://pynwb.readthedocs.io/en/stable/).

This project was inspired by [kerchunk](https://github.com/fsspec/kerchunk) and [hdmf-zarr](https://hdmf-zarr.readthedocs.io/en/latest/index.html) and depends on [zarr](https://zarr.readthedocs.io/en/stable/), [h5py](https://www.h5py.org/), [remfile](https://github.com/magland/remfile) and [numcodecs](https://numcodecs.readthedocs.io/en/stable/).

## Installation

For now, install from source. Clone this repo and then

```bash
cd lindi
pip install -e .
```

## Example usage

```python
# examples/example1.py

import json
import pynwb
import lindi

# Define the URL for a remote NWB file
h5_url = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"

# Create a read-only Zarr store as a wrapper for the h5 file
store = lindi.LindiH5ZarrStore.from_file(h5_url)

# Generate a reference file system
rfs = store.to_reference_file_system()

# Save it to a file for later use
with open("example.zarr.json", "w") as f:
    json.dump(rfs, f, indent=2)

# Create an h5py-like client from the reference file system
client = lindi.LindiH5pyFile.from_reference_file_system(rfs)

# Open using pynwb
with pynwb.NWBHDF5IO(file=client, mode="r") as io:
    nwbfile = io.read()
    print(nwbfile)
```

Or if you already have a .zarr.json file prepared (loading is much faster)

```python
# examples/example2.py

import pynwb
import lindi

# Define the URL for a remote .zarr.json file
url = 'https://kerchunk.neurosift.org/dandi/dandisets/000939/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/zarr.json'

# Load the h5py-like client from the reference file system
client = lindi.LindiH5pyFile.from_reference_file_system(url)

# Open using pynwb
with pynwb.NWBHDF5IO(file=client, mode="r") as io:
    nwbfile = io.read()
    print(nwbfile)
```

## Mixing and matching data from multiple sources

Once we have NWB files represented by relatively small reference file systems (e.g., .zarr.json files), we can begin to mix and match data from multiple sources. More on this to come.

## For developers

[Special Zarr annotations used by LINDI](docs/special_zarr_annotations.md)

## License

See [LICENSE](LICENSE).
