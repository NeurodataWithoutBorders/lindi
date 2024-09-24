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
