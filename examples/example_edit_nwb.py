import lindi
import h5py
import pynwb


# Define the URL for a remote .zarr.json file
url = 'https://kerchunk.neurosift.org/dandi/dandisets/000939/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/zarr.json'

# Load the h5py-like client from the reference file system
client = lindi.LindiH5pyFile.from_reference_file_system(url)

# modify the age of the subject
subject = client['general']['subject']  # type: ignore
assert isinstance(subject, h5py.Group)
del subject['age']  # type: ignore
subject.create_dataset('age', data=b'3')

# Create a new reference file system
rfs_new = client.to_reference_file_system()

# Optionally write to a file
# import json
# with open('new.zarr.json', 'w') as f:
#     json.dump(rfs_new, f)

# Load a new h5py-like client from the new reference file system
client_new = lindi.LindiH5pyFile.from_reference_file_system(rfs_new)

# Open using pynwb and verify that the subject age has been updated
with pynwb.NWBHDF5IO(file=client, mode="r") as io:
    nwbfile = io.read()
    print(nwbfile)
