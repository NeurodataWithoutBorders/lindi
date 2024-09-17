# LINDI - Linked Data Interface

[![latest-release](https://img.shields.io/pypi/v/lindi.svg)](https://pypi.org/project/lindi)
![tests](https://github.com/neurodatawithoutborders/lindi/actions/workflows/integration_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/neurodatawithoutborders/lindi/branch/main/graph/badge.svg?token=xxxx)](https://codecov.io/gh/neurodatawithoutborders/lindi)

:warning: Please note, LINDI is currently under development and should not yet be used in practice.

LINDI is a cloud-friendly file format and Python library designed for managing scientific data, especially Neurodata Without Borders (NWB) datasets. It offers an alternative to [HDF5](https://docs.hdfgroup.org/hdf5/v1_14/_intro_h_d_f5.html) and [Zarr](https://zarr.dev/), maintaining compatibility with both, while providing features tailored for linking to remote datasets stored in the cloud, such as those on the [DANDI Archive](https://www.dandiarchive.org/). LINDI's unique structure and capabilities make it particularly well-suited for efficient data access and management in cloud environments.

**What is a LINDI file?**

A LINDI file is a cloud-friendly format for storing scientific data, designed to be compatible with HDF5 and Zarr while offering unique advantages. It comes in two types: JSON/text format (.lindi.json) and binary format (.lindi.tar).

In the JSON format, the hierarchical group structure, attributes, and small datasets are stored in a JSON structure, with references to larger data chunks stored in external files (inspired by [kerchunk](https://github.com/fsspec/kerchunk)). This format is human-readable and easily inspected and edited. On the other hand, the binary format is a .tar file that contains the JSON file along with optional internal data chunks referenced by the JSON file, in addition to external chunks. This format allows for efficient cloud storage and random access.

The main advantage of the JSON LINDI format is its readability and ease of modification, while the binary LINDI format offers the ability to include internal data chunks, providing flexibility in data storage and retrieval. Both formats are optimized for cloud use, enabling efficient downloading and access from cloud storage.

**What are the main use cases?**

LINDI files are particularly useful in the following scenarios:

**Efficient NWB File Representation on DANDI**: A LINDI JSON file can represent an NWB file stored on the DANDI Archive (or other remote system). By downloading a condensed JSON file, the entire group structure can be retrieved in a single request, facilitating efficient loading of NWB files. For instance, [Neurosift](https://github.com/flatironinstitute/neurosift) utilizes pre-generated LINDI JSON files to streamline the loading process of NWB files from DANDI.

**Creating Amended NWB Files**: LINDI allows for the creation of amended NWB files that add new data objects to existing NWB files without duplicating the entire file. This is achieved by generating a binary LINDI file that references the original NWB file and includes additional data objects stored as internal data chunks. This approach saves storage space and reduces redundancy.

**Why not use Zarr?**

While Zarr is a cloud-friendly alternative to HDF5, it has notable limitations. Zarr archives often consist of thousands of individual files, making them cumbersome to manage. In contrast, LINDI files adopt a single file approach similar to HDF5, enhancing manageability while retaining cloud-friendliness. Another limitation of Zarr is the lack of a mechanism to reference data chunks in external datasets as LINDI has. Additionally, Zarr does not support certain features utilized by PyNWB, such as compound data types and references, which are supported by both HDF5 and LINDI.

**Why not use HDF5?**

HDF5 is not well-suited for cloud environments because accessing a remote HDF5 file often requires a large number of small requests to retrieve metadata before larger data chunks can be downloaded. LINDI addresses this by storing the entire group structure in a single JSON file, which can be downloaded in one request. Additionally, HDF5 lacks a built-in mechanism for referencing data chunks in external datasets. Furthermore, HDF5 does not support custom Python codecs, a feature available in both Zarr and LINDI. These advantages make LINDI a more efficient and versatile option for cloud-based data storage and access.

**Does LINDI use Zarr?**

Yes, LINDI leverages the Zarr format to store data, including attributes and group hierarchies. However, instead of using directories and files like Zarr, LINDI stores all data within a single JSON structure. This structure includes references to large data chunks, which can reside in remote files (e.g., an HDF5 NWB file on DANDI) or within internal data chunks in the binary LINDI file. Although NWB relies on certain HDF5 features not supported by Zarr, LINDI provides mechanisms to represent these features in Zarr, ensuring compatibility and extending functionality.

**Is tar format really cloud-friendly**

With LINDI, yes. See [docs/tar.md](docs/tar.md) for details.

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
with lindi.LindiH5pyFile.from_lindi_file('example.lindi.tar', mode='w') as f:
    f.attrs['attr1'] = 'value1'
    f.attrs['attr2'] = 7
    ds = f.create_dataset('dataset1', shape=(1000, 1000), dtype='f')
    ds[...] = np.random.rand(1000, 1000)

# Later read the file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi.tar', mode='r') as f:
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
