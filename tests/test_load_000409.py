import tempfile
import pytest
import lindi


@pytest.mark.network
def test_load_000409():
    import pynwb
    # https://neurosift.app/?p=/nwb&url=https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/&dandisetId=000409&dandisetVersion=draft
    url = 'https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/'
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f'{tmpdir}/example.nwb.lindi.json'
        with lindi.LindiH5pyFile.from_hdf5_file(url) as f:
            f.write_lindi_file(fname)
        f = lindi.LindiH5pyFile.from_lindi_file(fname, mode='r')
        with pynwb.NWBHDF5IO(file=f, mode='r') as io:
            nwb = io.read()
            print(nwb)
            X = nwb.acquisition['ElectricalSeriesAp']  # type: ignore
            a = X.data[:1000]
            assert a.shape == (1000, X.data.shape[1])
            units = nwb.units  # type: ignore
            units_ids = units['id']
            assert len(units_ids) == 590


if __name__ == '__main__':
    test_load_000409()
