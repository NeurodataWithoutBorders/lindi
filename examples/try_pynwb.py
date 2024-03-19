import json
import lindi
import pynwb


# https://neurosift.app/?p=/nwb&dandisetId=000717&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/
h5_url = "https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/"


def try_pynwb():
    # Create the read-only store for the h5 file
    store = lindi.LindiH5ZarrStore.from_file(h5_url)

    # Create the reference file system object
    rfs = store.to_reference_file_system()

    # For reference, save it to a file
    with open("example.zarr.json", "w") as f:
        json.dump(rfs, f, indent=2)

    # Create the client from the reference file system
    client = lindi.LindiZarrWrapper.from_reference_file_system(rfs)

    # Try to read using pynwb
    with pynwb.NWBHDF5IO(file=client, mode="r") as io:
        nwbfile = io.read()
        print(nwbfile)


if __name__ == "__main__":
    try_pynwb()
