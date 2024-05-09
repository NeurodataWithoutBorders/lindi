import json
import pytest
import lindi
from utils import arrays_are_equal


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
    with open("example.nwb.lindi.json", "w") as f:
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

    # Define the URL for a remote .nwb.lindi.json file
    url = 'https://lindi.neurosift.org/dandi/dandisets/000939/assets/56d875d6-a705-48d3-944c-53394a389c85/nwb.lindi.json'

    # Load the h5py-like client from the reference file system
    client = lindi.LindiH5pyFile.from_reference_file_system(url)

    # Open using pynwb
    with pynwb.NWBHDF5IO(file=client, mode="r") as io:
        nwbfile = io.read()
        print(nwbfile)


@pytest.mark.network
def test_remote_data_rfs_copy():
    # Test that we can copy datasets and groups from one reference file system to another
    # and the data itself is not copied, only the references.
    url = 'https://lindi.neurosift.org/dandi/dandisets/000939/assets/56d875d6-a705-48d3-944c-53394a389c85/nwb.lindi.json'

    client = lindi.LindiH5pyFile.from_reference_file_system(url)

    rfs2 = {'refs': {
        '.zgroup': '{"zarr_format": 2}',
    }}
    client2 = lindi.LindiH5pyFile.from_reference_file_system(rfs2)

    # This first dataset is a 2D array with chunks
    ds = client['processing/behavior/Position/position/data']
    assert isinstance(ds, lindi.LindiH5pyDataset)
    assert ds.shape == (360867, 2)

    client.copy('processing/behavior/Position/position/data', client2, 'copied_data1')
    aa = rfs2['refs']['copied_data1/.zarray']
    assert isinstance(aa, str)
    assert 'copied_data1/0.0' in rfs2['refs']
    bb = rfs2['refs']['copied_data1/0.0']
    assert isinstance(bb, list)  # make sure it is a reference, not the actual data

    ds2 = client2['copied_data1']
    assert isinstance(ds2, lindi.LindiH5pyDataset)
    assert arrays_are_equal(ds[()], ds2[()])  # make sure the data is the same

    # This next dataset has an _EXTERNAL_ARRAY_LINK which means it has a pointer
    # to a dataset in a remote h5py
    ds = client['processing/ecephys/LFP/LFP/data']
    assert isinstance(ds, lindi.LindiH5pyDataset)
    assert ds.shape == (17647830, 64)

    client.copy('processing/ecephys/LFP/LFP/data', client2, 'copied_data2')
    aa = rfs2['refs']['copied_data2/.zarray']
    assert isinstance(aa, str)
    assert 'copied_data2/0.0' not in rfs2['refs']  # make sure the chunks were not copied

    ds2 = client2['copied_data2']
    assert isinstance(ds2, lindi.LindiH5pyDataset)
    assert arrays_are_equal(ds[100000:100010], ds2[100000:100010])


if __name__ == "__main__":
    test_remote_data_rfs_copy()
