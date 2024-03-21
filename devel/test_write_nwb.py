from typing import Any

from datetime import datetime
from uuid import uuid4
import numpy as np
from dateutil.tz import tzlocal
from pynwb import NWBHDF5IO, NWBFile, H5DataIO
from pynwb.ecephys import LFP, ElectricalSeries
import zarr
import lindi

nwbfile: Any = NWBFile(
    session_description="my first synthetic recording",
    identifier=str(uuid4()),
    session_start_time=datetime.now(tzlocal()),
    experimenter=[
        "Baggins, Bilbo",
    ],
    lab="Bag End Laboratory",
    institution="University of Middle Earth at the Shire",
    experiment_description="I went on an adventure to reclaim vast treasures.",
    session_id="LONELYMTN001",
)

device = nwbfile.create_device(
    name="array", description="the best array", manufacturer="Probe Company 9000"
)

nwbfile.add_electrode_column(name="label", description="label of electrode")

nshanks = 4
nchannels_per_shank = 3
electrode_counter = 0

for ishank in range(nshanks):
    # create an electrode group for this shank
    electrode_group = nwbfile.create_electrode_group(
        name="shank{}".format(ishank),
        description="electrode group for shank {}".format(ishank),
        device=device,
        location="brain area",
    )
    # add electrodes to the electrode table
    for ielec in range(nchannels_per_shank):
        nwbfile.add_electrode(
            group=electrode_group,
            label="shank{}elec{}".format(ishank, ielec),
            location="brain area",
        )
        electrode_counter += 1

all_table_region = nwbfile.create_electrode_table_region(
    region=list(range(electrode_counter)),  # reference row indices 0 to N-1
    description="all electrodes",
)

raw_data = np.random.randn(300000, 100)
raw_electrical_series = ElectricalSeries(
    name="ElectricalSeries",
    data=H5DataIO(data=raw_data, chunks=(100000, 100)),  # type: ignore
    electrodes=all_table_region,
    starting_time=0.0,  # timestamp of the first sample in seconds relative to the session start time
    rate=20000.0,  # in Hz
)

nwbfile.add_acquisition(raw_electrical_series)

lfp_data = np.random.randn(50, 12)
lfp_electrical_series = ElectricalSeries(
    name="ElectricalSeries",
    data=lfp_data,
    electrodes=all_table_region,
    starting_time=0.0,
    rate=200.0,
)

lfp = LFP(electrical_series=lfp_electrical_series)

ecephys_module = nwbfile.create_processing_module(
    name="ecephys", description="processed extracellular electrophysiology data"
)
ecephys_module.add(lfp)

nwbfile.add_unit_column(name="quality", description="sorting quality")

firing_rate = 20
n_units = 10
res = 1000
duration = 20
for n_units_per_shank in range(n_units):
    spike_times = (
        np.where(np.random.rand((res * duration)) < (firing_rate / res))[0] / res
    )
    nwbfile.add_unit(spike_times=spike_times, quality="good")

# with tempfile.TemporaryDirectory() as tmpdir:
tmpdir = '.'
dirname = f'{tmpdir}/test.nwb'
store = zarr.DirectoryStore(dirname)
# create a top-level group
root = zarr.group(store=store, overwrite=True)
client = lindi.LindiH5pyFile.from_zarr_store(store, mode='a')
with NWBHDF5IO(file=client, mode='w') as io:
    io.write(nwbfile)  # type: ignore

store2 = zarr.DirectoryStore(dirname)
client2 = lindi.LindiH5pyFile.from_zarr_store(store2, mode='r')
with NWBHDF5IO(file=client2, mode='r') as io:
    nwbfile2 = io.read()  # type: ignore
    print(nwbfile2)
