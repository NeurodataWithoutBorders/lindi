# LINDI - Linked Data Interface

[![latest-release](https://img.shields.io/pypi/v/lindi.svg)](https://pypi.org/project/lindi)
![tests](https://github.com/neurodatawithoutborders/lindi/actions/workflows/integration_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/neurodatawithoutborders/lindi/branch/main/graph/badge.svg?token=xxxx)](https://codecov.io/gh/neurodatawithoutborders/lindi)

LINDI is a cloud-friendly file format and Python library designed for managing scientific data, especially Neurodata Without Borders (NWB) datasets. It offers an alternative to [HDF5](https://docs.hdfgroup.org/hdf5/v1_14/_intro_h_d_f5.html) and [Zarr](https://zarr.dev/), maintaining compatibility with both, while providing features tailored for linking to remote datasets stored in the cloud, such as those on the [DANDI Archive](https://www.dandiarchive.org/). LINDI's unique structure and capabilities make it particularly well-suited for efficient data access and management in cloud environments.

**What is a LINDI file?**

A LINDI file is a cloud-friendly format for storing scientific data, designed to be compatible with HDF5 and Zarr while offering unique advantages. It comes in three types which are representations of the same underlying data: JSON/text format (.lindi.json), binary format (.lindi.tar), and directory format (.lindi.d).

In the JSON format, the hierarchical group structure, attributes, and small datasets are stored in a JSON structure, with references to larger data chunks stored in external files (inspired by [kerchunk](https://github.com/fsspec/kerchunk)). This format is human-readable and easily inspected and edited.

The binary format is a .tar file that contains the JSON file (lindi.json) along with optional internal data chunks referenced by the JSON file, in addition to external chunks. This format can be used to create a new NWB file that builds on an existing NWB file without duplicating it and adds new data objects (see below).

The directory format is similar to the .tar format but it stores the lindi.json and the binary chunks in a directory rather than in a .tar.

**What are the main use cases?**

LINDI files are particularly useful in the following scenarios:

**Efficient NWB File Representation on DANDI**: A LINDI JSON file can represent an NWB file stored on the DANDI Archive (or other remote system). By downloading a condensed JSON file, the entire group structure can be retrieved in a single request, facilitating efficient loading of NWB files. For instance, [Neurosift](https://github.com/flatironinstitute/neurosift) utilizes pre-generated LINDI JSON files to streamline the loading process of NWB files from DANDI ([here is an example](https://neurosift.app/?p=/nwb&url=https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/&dandisetId=000409&dandisetVersion=draft)).

**Creating Amended NWB Files**: LINDI allows for the creation of amended NWB files that add new data objects to existing NWB files without duplicating the entire file. This is achieved by generating a binary or directory LINDI file that references the original NWB file and includes additional data objects stored as internal data chunks. This approach saves storage space by reducing redundancy and establishing dependencies between NWB files.

**Why not use Zarr?**

When comparing LINDI to Zarr it should be noted that LINDI files are in fact valid Zarr archives that can be accessed via the Zarr API. Indeed, a LINDI file is a special type of Zarr store that allows for external links to chunks (see [kerchunk](https://github.com/fsspec/kerchunk)) and special conventions for representing HDF5 features used by NWB that are not natively supported in Zarr.

Traditional Zarr directory stores have some limitations. First, Zarr archives often consist of tens of thousands of individual files, making them cumbersome to manage. In contrast, LINDI adopts a single file approach similar to HDF5, enhancing manageability while retaining cloud-friendliness. Another limitation (as mentioned) is the lack of a mechanism to reference data chunks in external datasets as LINDI has. Finally, Zarr does not natively support certain features utilized by NWB, such as compound data types and references. These are supported by both HDF5 and LINDI.

**Why not use HDF5?**

HDF5 is not well-suited for cloud environments because accessing a remote HDF5 file often requires a large number of small requests to retrieve metadata before larger data chunks can be downloaded. LINDI addresses this by storing the entire group structure in a single JSON file, which can be downloaded in one request. Additionally, HDF5 lacks a built-in mechanism for referencing data chunks in external datasets. Furthermore, HDF5 does not support custom Python codecs, a feature available in both Zarr and LINDI.

**Is tar format really cloud-friendly?**

With LINDI, yes. See [docs/tar.md](docs/tar.md) for details.

## Installation

```bash
pip install --upgrade lindi
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

You can inspect the example.lindi.json file to get an idea of how data are stored. If you are familiar with the internal Zarr format, you will recognize the .group and .zarray files and the layout of the chunks. [Here is an example](https://lindi.neurosift.org/dandi/dandisets/000409/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/nwb.lindi.json) of a LINDI JSON file that represents an NWB file stored on DANDI.

Because the above dataset is very small, it can all fit reasonably inside the JSON file. For storing larger arrays (the usual case) it is better to use the binary or directory format.

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

With LINDI, it is easy to load an NWB file stored on DANDI. The following example demonstrates how to load an NWB file from DANDI, view it using the pynwb library, and save it as a relatively smaller .lindi.json file. The LINDI JSON file can then be read directly to access the NWB file.

```python
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
f.close()

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

One of the main use cases of LINDI is to create amended NWB files that add new data objects to existing NWB files without duplicating the entire file. This is achieved by generating a binary or directory LINDI file that references the original NWB file and includes additional data objects stored as internal data chunks.

```python
import numpy as np
import pynwb
from pynwb.file import TimeSeries
import lindi

# Load the remote NWB file from DANDI
h5_url = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"
f = lindi.LindiH5pyFile.from_hdf5_file(h5_url)

# Write to a local .lindi.tar file
f.write_lindi_file('example.nwb.lindi.tar')
f.close()

# Open with pynwb and add new data
g = lindi.LindiH5pyFile.from_lindi_file('example.nwb.lindi.tar', mode='r+')
with pynwb.NWBHDF5IO(file=g, mode="a") as io:
    nwbfile = io.read()
    timeseries_test = TimeSeries(
        name="test",
        data=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
        rate=1.,
        unit='s'
    )
    ts = nwbfile.processing['behavior'].add(timeseries_test)  # type: ignore
    io.write(nwbfile)  # type: ignore

# Later on, you can read the file again
h = lindi.LindiH5pyFile.from_lindi_file('example.nwb.lindi.tar')
with pynwb.NWBHDF5IO(file=h, mode="r") as io:
    nwbfile = io.read()
    test_timeseries = nwbfile.processing['behavior']['test']  # type: ignore
    print(test_timeseries)
```

## Notes

This project was inspired by [kerchunk](https://github.com/fsspec/kerchunk) and [hdmf-zarr](https://hdmf-zarr.readthedocs.io/en/latest/index.html).

## For developers

[Special Zarr annotations used by LINDI](docs/special_zarr_annotations.md)

## License

See [LICENSE](LICENSE).
