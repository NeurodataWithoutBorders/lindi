import pynwb
import lindi

# Define the URL for a remote .nwb.lindi.json file
url = 'https://kerchunk.neurosift.org/dandi/dandisets/000939/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/zarr.json'

# Load the h5py-like client from the reference file system
client = lindi.LindiH5pyFile.from_lindi_file(url)

# Open using pynwb
with pynwb.NWBHDF5IO(file=client, mode="r") as io:
    nwbfile = io.read()
    print(nwbfile)
