# LINDI - Linked Data Interface

[![latest-release](https://img.shields.io/pypi/v/lindi.svg)](https://pypi.org/project/lindi)
![tests](https://github.com/neurodatawithoutborders/lindi/actions/workflows/integration_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/neurodatawithoutborders/lindi/branch/main/graph/badge.svg?token=xxxx)](https://codecov.io/gh/neurodatawithoutborders/lindi)

:warning: Please note, LINDI is currently under development and should not yet be used in practice.

LINDI is a cloud-friendly file format and Python library for working with scientific data, especially Neurodata Without Borders (NWB) datasets. It is an alternative to HDF5 and Zarr, but is compatible with both, with features that make it particularly well-suited for linking to remote datasets in the cloud such as those stored on [DANDI Archive](https://www.dandiarchive.org/).

**What is a LINDI file?**

You can think of a LINDI file as a differently-formatted HDF5 file that is cloud-friendly and capable of linking to data chunks in remote files (such as on DANDI Archive).

There are two types of LINDI files: JSON/text format (.lindi.json) and binary format (.lindi or .lindi.tar). In the JSON format, the hierarchical group structure, attributes, a small datasets are all stored in a JSON structure, with references to larger data chunks stored in external files. The binary format is a .tar file that contains this JSON file as well as optional internal data chunks that can be referenced by the JSON file in addition to the external chunks. The advantage of the JSON LINDI format is that it is human-readable and easily inspected and edited. The advantage of the binary LINDI format is that it can contain internal data chunks. Both formats are cloud-friendly in that they can be efficiently downloaded from cloud storage with random access.

**What are the main use cases?**

One use case is to represent a NWB file on DANDI using a condensed JSON file so that the entire group structure can be downloaded in a single request. For example, eurosift uses pre-generated LINDI JSON files to efficiently load NWB files from DANDI.

Another use case is to create amended NWB files that add additional data objects to existing NWB files without redundantly storing the entire NWB file. This is done by creating a binary LINDI file that references the original NWB file and adds additional data objects that are stored as internal data chunks.

**Why not use Zarr?**

Zarr provides a cloud-friendly alternative to HDF5, but an important limitation is that Zarr archives often contain thousands of individual files making it cumbersome to manage. LINDI files are more like HDF5 in that they favor the single index approach, but are just as cloud-friendly as Zarr. A second limitation of Zarr is that there is currently no mechanism for referencing chunks in external datasets.

**Why not use HDF5?**

HDF5 is not cloud-friendly in that if you have a remote HDF5 file, many small requests are required to obtain metadata before larger data chunks can be downloaded. Both JSON and binary LINDI files solve this problem by storing the entire group structure in a single JSON structure that can be downloaded in a single request. Furthermore, as with HDF5, there is no built-in mechanism for referencing chunks in external datasets.

**Does LINDI use Zarr?**

Yes, LINDI uses the Zarr format to store data, including attributes and group hierarchies. But instead of using directories and files, it stores all of the data in a single JSON data structure, with references to large data chunks, which can either be found in remote files (e.g., in a HDF5 NWB file on DANDI) or in internal data chunks in the binary LINDI file. However, NWB depends on certain HDF5 features that are not supported by Zarr, so LINDI also provides mechanism for representing these features in Zarr.

**Is tar format really cloud-friendly**

With LINDI, yes. TODO: discuss

## Installation

```bash
pip install lindi
```

Or from source

```bash
cd lindi
pip install -e .
```

## Usage

**Creating and reading a LINDI file**

The simplest way to start is to use it like HDF5.

```python
import lindi

# Create a new lindi.json file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi.json', mode='w') as f:
    f.attrs['attr1'] = 'value1'
    f.attrs['attr2'] = 7
    ds = f.create_dataset('dataset1', shape=(10,), dtype='f')
    ds[...] = 12

# Later read the file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi.json', mode='r') as f:
    print(f.attrs['attr1'])
    print(f.attrs['attr2'])
    print(f['dataset1'][...])
```

You can inspect the example.lindi.json file to get an idea of how the data are stored. If you are familiar with the internal Zarr format, you will recognize the .group and .zarray files and the layout of the chunks.

Because the above dataset is very small, it can all fit reasonably inside the JSON file. For storing larger arrays (the usual case) it is better to use the binary format. Just leave off the .json extension.

```python
import numpy as np
import lindi

# Create a new lindi binary file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi', mode='w') as f:
    f.attrs['attr1'] = 'value1'
    f.attrs['attr2'] = 7
    ds = f.create_dataset('dataset1', shape=(1000, 1000), dtype='f')
    ds[...] = np.random.rand(1000, 1000)

# Later read the file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi', mode='r') as f:
    print(f.attrs['attr1'])
    print(f.attrs['attr2'])
    print(f['dataset1'][...])
```

**Loading a remote NWB file from DANDI**

```python
import json
import pynwb
import lindi

# Define the URL for a remote NWB file
h5_url = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"

# Load as LINDI and view using pynwb
f = lindi.LindiH5pyFile.from_hdf5_file(h5_url)
with pynwb.NWBHDF5IO(file=f, mode="r") as io:
    nwbfile = io.read()
    print('NWB via LINDI')
    print(nwbfile)

    print('Electrode group at shank0:')
    print(nwbfile.electrode_groups["shank0"])  # type: ignore

    print('Electrode group at index 0:')
    print(nwbfile.electrodes.group[0])  # type: ignore

# Save as LINDI JSON
f.write_lindi_file('example.nwb.lindi.json')

# Later, read directly from the LINDI JSON file
g = lindi.LindiH5pyFile.from_lindi_file('example.nwb.lindi.json')
with pynwb.NWBHDF5IO(file=g, mode="r") as io:
    nwbfile = io.read()
    print('')
    print('NWB from LINDI JSON:')
    print(nwbfile)

    print('Electrode group at shank0:')
    print(nwbfile.electrode_groups["shank0"])  # type: ignore

    print('Electrode group at index 0:')
    print(nwbfile.electrodes.group[0])  # type: ignore
```

## Amending an NWB file

Basically you save the remote NWB as a local binary LINDI file, and then add additional data objects to it.

TODO: finish this section

## Notes

This project was inspired by [kerchunk](https://github.com/fsspec/kerchunk) and [hdmf-zarr](https://hdmf-zarr.readthedocs.io/en/latest/index.html) and depends on [zarr](https://zarr.readthedocs.io/en/stable/), [h5py](https://www.h5py.org/) and [numcodecs](https://numcodecs.readthedocs.io/en/stable/).

## For developers

[Special Zarr annotations used by LINDI](docs/special_zarr_annotations.md)

## License

See [LICENSE](LICENSE).
