import lindi
import pynwb
from pynwb import TimeSeries
from pynwb.epoch import TimeIntervals

# URL of the remote .nwb.lindi.json file
url = 'https://lindi.neurosift.org/dandi/dandisets/000713/assets/b2391922-c9a6-43f9-8b92-043be4015e56/nwb.lindi.json'

# Load the h5py-like client for the reference file system
# in read-write mode with a staging area
with lindi.StagingArea.create(base_dir='lindi_staging') as staging_area:
    client = lindi.LindiH5pyFile.from_lindi_file(
        url,
        mode="r+",
        staging_area=staging_area
    )

    with pynwb.NWBHDF5IO(file=client, mode="r+") as io:
        nwbfile = io.read()

        timeseries = TimeSeries("test_timeseries", description="test", unit="unit", data=[1, 2, 3], timestamps=[1, 2, 3])
        nwbfile.add_acquisition(timeseries)

        intervals = TimeIntervals("test_timeintervals", description="test")
        intervals.add_row(start_time=1.0, stop_time=2.0)
        nwbfile.add_time_intervals(intervals)

        io.write(nwbfile)

    client.write_lindi_file('example.nwb.lindi.json')
    # you can open the example.nwb.lindi.json file and confirm that the new timeseries
    # and time intervals table are present, under the "refs" key at the keys
    # "acquisition/test_timeseries" and "acquisition/test_timeintervals"

# upload example.nwb.lindi.json to DANDI...