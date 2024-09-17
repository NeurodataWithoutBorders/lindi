import pytest
import lindi
import h5py


@pytest.mark.network
def test_split_contiguous_dataset():
    # https://neurosift.app/?p=/nwb&dandisetId=000935&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/e18e787a-544a-438e-8396-f396efb3bd3d/download/
    h5_url = "https://api.dandiarchive.org/api/assets/e18e787a-544a-438e-8396-f396efb3bd3d/download/"

    opts = lindi.LindiH5ZarrStoreOpts(
        contiguous_dataset_max_chunk_size=1000 * 1000 * 17
    )
    x = lindi.LindiH5pyFile.from_hdf5_file(h5_url, zarr_store_opts=opts)
    d = x['acquisition/ElectricalSeries/data']
    assert isinstance(d, h5py.Dataset)
    print(d.shape)
    assert d[0][0] == 6.736724784228119e-06
    assert d[10 * 1000 * 1000][0] == -1.0145925267155008e-06
    rfs = x.to_reference_file_system()
    zarray = rfs['refs']['acquisition/ElectricalSeries/data/.zarray']
    assert zarray['chunks'] == [66406, 32]
    aa = rfs['refs']['acquisition/ElectricalSeries/data/5.0']
    assert aa[1] == 2415072880
    assert aa[2] == 16999936


if __name__ == "__main__":
    test_split_contiguous_dataset()
