import lindi
import pynwb


# https://neurosift.app/?p=/nwb&dandisetId=000717&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/
h5_url = "https://api.dandiarchive.org/api/assets/3d12a902-139a-4c1a-8fd0-0a7faf2fb223/download/"


def try_pynwb():
    store = lindi.LindiH5Store.from_file(h5_url)
    rfs = store.to_reference_file_system()
    client = lindi.LindiClient.from_reference_file_system(rfs)
    print(client)

    with pynwb.NWBHDF5IO(file=client, mode="r") as io:
        nwbfile = io.read()
        print(nwbfile)


if __name__ == "__main__":
    try_pynwb()