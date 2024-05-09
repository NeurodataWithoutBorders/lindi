import pynwb
import lindi

# Define the URL for a remote .nwb.lindi.json file
url = 'https://lindi.neurosift.org/dandi/dandisets/000939/assets/56d875d6-a705-48d3-944c-53394a389c85/nwb.lindi.json'

# Load the h5py-like client from the reference file system
client = lindi.LindiH5pyFile.from_lindi_file(url)

# Open using pynwb
with pynwb.NWBHDF5IO(file=client, mode="r") as io:
    nwbfile = io.read()
    print(nwbfile)
