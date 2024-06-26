from typing import Any
import shutil
import os
import zarr
import pynwb
import lindi


def example_create_zarr_nwb():
    zarr_dirname = 'example_nwb.zarr'
    if os.path.exists(zarr_dirname):
        shutil.rmtree(zarr_dirname)

    nwbfile = _create_sample_nwb_file()

    store = zarr.DirectoryStore(zarr_dirname)
    zarr.group(store=store)  # create a root group
    with lindi.LindiH5pyFile.from_zarr_store(store, mode='r+') as client:
        with pynwb.NWBHDF5IO(file=client, mode='r+') as io:
            io.write(nwbfile)  # type: ignore


def _create_sample_nwb_file():
    from datetime import datetime
    from uuid import uuid4

    import numpy as np
    from dateutil.tz import tzlocal

    from pynwb import NWBFile
    from pynwb.ecephys import LFP, ElectricalSeries

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

    raw_data = np.random.randn(50, 12)
    raw_electrical_series = ElectricalSeries(
        name="ElectricalSeries",
        data=raw_data,
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

    return nwbfile


if __name__ == '__main__':
    example_create_zarr_nwb()
