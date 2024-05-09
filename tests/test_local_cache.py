import tempfile
import json
import time
import pytest
import lindi


@pytest.mark.network
def test_remote_data_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        local_cache = lindi.LocalCache(cache_dir=tmpdir + '/local_cache')
        for passnum in range(2):
            import pynwb

            # Define the URL for a remote NWB file
            h5_url = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"

            # Create a read-only Zarr store as a wrapper for the h5 file
            store = lindi.LindiH5ZarrStore.from_file(h5_url)

            # Generate a reference file system
            rfs = store.to_reference_file_system()

            # Save it to a file for later use
            with open("example.nwb.lindi.json", "w") as f:
                json.dump(rfs, f, indent=2)

            # Create an h5py-like client from the reference file system
            client = lindi.LindiH5pyFile.from_reference_file_system(rfs, local_cache=local_cache)

            # Open using pynwb
            timer = time.time()
            with pynwb.NWBHDF5IO(file=client, mode="r") as io:
                nwbfile = io.read()
                print(nwbfile)
            x = client["/processing/ecephys/LFP/LFP/data"]
            assert isinstance(x, lindi.LindiH5pyDataset)
            x[:1000]
            elapsed = time.time() - timer
            print('Elapsed time:', elapsed)
            if passnum == 0:
                elapsed_0 = elapsed
            if passnum == 1:
                elapsed_1 = elapsed
                assert elapsed_1 < elapsed_0 * 0.3  # type: ignore


if __name__ == "__main__":
    test_remote_data_1()
