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
