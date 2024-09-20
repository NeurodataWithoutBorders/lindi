import os
import tempfile
import pytest
import numpy as np
import lindi


def test_create_and_read_lindi_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f'{tmpdir}/example.lindi.json'
        # Create a new lindi.json file
        with lindi.LindiH5pyFile.from_lindi_file(fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
            f.attrs['attr2'] = 7
            ds = f.create_dataset('dataset1', shape=(10,), dtype='f')
            ds[...] = 12

        # Later read the file
        with lindi.LindiH5pyFile.from_lindi_file(fname, mode='r') as f:
            assert f.attrs['attr1'] == 'value1'
            assert f.attrs['attr2'] == 7
            ds = f['dataset1']
            assert isinstance(ds, lindi.LindiH5pyDataset)
            assert ds.shape == (10,)
            for i in range(10):
                assert ds[i] == 12


def test_create_and_read_lindi_tar():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f'{tmpdir}/example.lindi.tar'
        # Create a new lindi.json file
        with lindi.LindiH5pyFile.from_lindi_file(fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
            f.attrs['attr2'] = 7
            ds = f.create_dataset('dataset1', shape=(10,), dtype='f')
            ds[...] = 12

        # Later read the file
        with lindi.LindiH5pyFile.from_lindi_file(fname, mode='r') as f:
            assert f.attrs['attr1'] == 'value1'
            assert f.attrs['attr2'] == 7
            ds = f['dataset1']
            assert isinstance(ds, lindi.LindiH5pyDataset)
            assert ds.shape == (10,)
            for i in range(10):
                assert ds[i] == 12


def test_create_and_read_lindi_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        dirname = f'{tmpdir}/example.lindi.d'
        # Create a new lindi.json file
        with lindi.LindiH5pyFile.from_lindi_file(dirname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
            f.attrs['attr2'] = 7
            ds = f.create_dataset('dataset1', shape=(10,), dtype='f')
            ds[...] = 12

        # verify that it's a directory
        assert os.path.isdir(dirname)

        # Later read the file
        with lindi.LindiH5pyFile.from_lindi_file(dirname, mode='r') as f:
            assert f.attrs['attr1'] == 'value1'
            assert f.attrs['attr2'] == 7
            ds = f['dataset1']
            assert isinstance(ds, lindi.LindiH5pyDataset)
            assert ds.shape == (10,)
            for i in range(10):
                assert ds[i] == 12


@pytest.mark.network
def test_represent_remote_nwb_as_lindi_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f'{tmpdir}/example.nwb.lindi.json'
        # Define the URL for a remote NWB file
        h5_url = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"

        # Load as LINDI and view using pynwb
        f = lindi.LindiH5pyFile.from_hdf5_file(h5_url)

        # Save as LINDI JSON
        f.write_lindi_file(fname)
        f.flush()

        # Later, read directly from the LINDI JSON file
        g = lindi.LindiH5pyFile.from_lindi_file(fname)

        # Later, read directly from the LINDI JSON file
        for k, v in f.attrs.items():
            v2 = g.attrs[k]
            if isinstance(v, lindi.LindiH5pyReference):
                assert isinstance(v2, lindi.LindiH5pyReference)
            else:
                assert v == v2

        f.close()
        g.close()


@pytest.mark.network
def test_amend_remote_nwb_as_lindi_tar():
    import pynwb
    from pynwb.file import TimeSeries
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f'{tmpdir}/example.nwb.lindi.tar'
        # Load the remote NWB file from DANDI
        h5_url = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"
        f = lindi.LindiH5pyFile.from_hdf5_file(h5_url)

        # Write to a local .lindi.tar file
        f.write_lindi_file(fname)
        f.close()

        # Open with pynwb and add new data
        g = lindi.LindiH5pyFile.from_lindi_file(fname, mode='r+')
        with pynwb.NWBHDF5IO(file=g, mode="a") as io:
            nwbfile = io.read()
            timeseries_test = TimeSeries(
                name="test",
                data=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                rate=1.,
                unit='s'
            )
            nwbfile.processing['behavior'].add(timeseries_test)  # type: ignore
            io.write(nwbfile)  # type: ignore

        # Later on, you can read the file again
        h = lindi.LindiH5pyFile.from_lindi_file(fname)
        with pynwb.NWBHDF5IO(file=h, mode="r") as io:
            nwbfile = io.read()
            test_timeseries = nwbfile.processing['behavior']['test']  # type: ignore
            assert test_timeseries.data.shape == (9,)
            for i in range(9):
                assert test_timeseries.data[i] == i + 1
