# LINDI - Linked Data Interface

[![latest-release](https://img.shields.io/pypi/v/lindi.svg)](https://pypi.org/project/lindi)
![tests](https://github.com/neurodatawithoutborders/lindi/actions/workflows/integration_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/neurodatawithoutborders/lindi/branch/main/graph/badge.svg?token=xxxx)](https://codecov.io/gh/neurodatawithoutborders/lindi)

:warning: Please note, LINDI is currently under development and should not yet be used in practice.

**HDF5 as Zarr as JSON for NWB**

LINDI provides a JSON representation of NWB (Neurodata Without Borders) data where the large data chunks are stored separately from the main metadata. This enables efficient storage, composition, and sharing of NWB files on cloud systems such as [DANDI](https://www.dandiarchive.org/) without duplicating the large data blobs.

LINDI provides:

- A specification for representing arbitrary HDF5 files as Zarr stores. This handles scalar datasets, references, soft links, and compound data types for datasets.
- A Zarr wrapper for remote or local HDF5 files (LindiH5ZarrStore).
- A mechanism for creating .lindi.json (or .nwb.lindi.json) files that reference data chunks in external files, inspired by [kerchunk](https://github.com/fsspec/kerchunk).
- An h5py-like interface for reading from and writing to these data sources that can be used with [pynwb](https://pynwb.readthedocs.io/en/stable/).
- A mechanism for uploading and downloading these data sources to and from cloud storage, including DANDI.

This project was inspired by [kerchunk](https://github.com/fsspec/kerchunk) and [hdmf-zarr](https://hdmf-zarr.readthedocs.io/en/latest/index.html) and depends on [zarr](https://zarr.readthedocs.io/en/stable/), [h5py](https://www.h5py.org/) and [numcodecs](https://numcodecs.readthedocs.io/en/stable/).

## Installation

```bash
pip install lindi
```

Or from source

```bash
cd lindi
pip install -e .
```

## Use cases

* Lazy-load a remote NWB/HDF5 file for efficient access to metadata and data.
* Represent a remote NWB/HDF5 file as a .nwb.lindi.json file.
* Read a local or remote .nwb.lindi.json file using pynwb or other tools.
* Edit a .nwb.lindi.json file using pynwb or other tools.
* Add datasets to a .nwb.lindi.json file using a local staging area.
* Upload a .nwb.lindi.json file with staged datasets to a cloud storage service such as DANDI.

### Lazy-load a remote NWB/HDF5 file for efficient access to metadata and data

```python
import pynwb
import lindi

# URL of the remote NWB file
h5_url = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"

# Set up a local cache
local_cache = lindi.LocalCache(cache_dir='lindi_cache')

# Create the h5py-like client
client = lindi.LindiH5pyFile.from_hdf5_file(h5_url, local_cache=local_cache)

# Open using pynwb
with pynwb.NWBHDF5IO(file=client, mode="r") as io:
    nwbfile = io.read()
    print(nwbfile)

# The downloaded data will be cached locally, so subsequent reads will be faster
```

### Represent a remote NWB/HDF5 file as a .nwb.lindi.json file

```python
import json
import lindi

# URL of the remote NWB file
h5_url = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"

# Create the h5py-like client
client = lindi.LindiH5pyFile.from_hdf5_file(h5_url)

# Generate a reference file system
rfs = client.to_reference_file_system()

# Save it to a file for later use
with open("example.lindi.json", "w") as f:
    json.dump(rfs, f, indent=2)

# See the next example for how to read this file
```

### Read a local or remote .nwb.lindi.json file using pynwb or other tools

```python
import pynwb
import lindi

# URL of the remote .nwb.lindi.json file
url = 'https://kerchunk.neurosift.org/dandi/dandisets/000939/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/zarr.json'

# Load the h5py-like client
client = lindi.LindiH5pyFile.from_lindi_file(url)

# Open using pynwb
with pynwb.NWBHDF5IO(file=client, mode="r") as io:
    nwbfile = io.read()
    print(nwbfile)
```

### Edit a .nwb.lindi.json file using pynwb or other tools

```python
import json
import lindi

# URL of the remote .nwb.lindi.json file
url = 'https://lindi.neurosift.org/dandi/dandisets/000939/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/zarr.json'

# Load the h5py-like client for the reference file system
# in read-write mode
client = lindi.LindiH5pyFile.from_reference_file_system(url, mode="r+")

# Edit an attribute
client.attrs['new_attribute'] = 'new_value'

# Save the changes to a new .nwb.lindi.json file
rfs_new = client.to_reference_file_system()
with open('new.nwb.lindi.json', 'w') as f:
    f.write(json.dumps(rfs_new, indent=2, sort_keys=True))
```

### Add datasets to a .nwb.lindi.json file using a local staging area

```python
import lindi

# URL of the remote .nwb.lindi.json file
url = 'https://lindi.neurosift.org/dandi/dandisets/000939/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/zarr.json'

# Load the h5py-like client for the reference file system
# in read-write mode with a staging area
with lindi.StagingArea.create(base_dir='lindi_staging') as staging_area:
    client = lindi.LindiH5pyFile.from_lindi_file(
        url,
        mode="r+",
        staging_area=staging_area
    )
    # add datasets to client using pynwb or other tools
    # upload the changes to the remote .nwb.lindi.json file
```

### Upload a .nwb.lindi.json file with staged datasets to a cloud storage service such as DANDI

See [this example](https://github.com/magland/lindi-dandi/blob/main/devel/lindi_test_2.py).

## For developers

[Special Zarr annotations used by LINDI](docs/special_zarr_annotations.md)

## License

See [LICENSE](LICENSE).
