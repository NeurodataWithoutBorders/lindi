import json
import pytest
import lindi


@pytest.mark.network
def test_remote_data_1():
    import pynwb

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


@pytest.mark.network
def test_remote_data_2():
    import pynwb

    # Define the URL for a remote .zarr.json file
    url = 'https://kerchunk.neurosift.org/dandi/dandisets/000939/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/zarr.json'

    # Load the h5py-like client from the reference file system
    client = lindi.LindiH5pyFile.from_reference_file_system(url)

    # Open using pynwb
    with pynwb.NWBHDF5IO(file=client, mode="r") as io:
        nwbfile = io.read()
        print(nwbfile)
