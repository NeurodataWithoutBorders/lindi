import numpy as np
import h5py
import remfile


# https://neurosift.app/?p=/nwb&dandisetId=000776&dandisetVersion=draft&url=https://api.dandiarchive.org/api/assets/54895119-f739-4544-973e-a9341a5c66ad/download/
h5_url = "https://api.dandiarchive.org/api/assets/54895119-f739-4544-973e-a9341a5c66ad/download/"


def demonstrate_slow_get_chunk_info():
    # Open the remote file using remfile. We use verbose option to see the
    # activity of the download. Don't be confused about when remfile says
    # "loading 2 chunks" - those are different chunks than hdf5 dataset chunks.
    remf = remfile.File(h5_url, verbose=True)

    h5f = h5py.File(remf, "r")
    dset = h5f["/acquisition/CalciumImageSeries/data"]
    assert isinstance(dset, h5py.Dataset)
    shape = dset.shape
    chunk_shape = dset.chunks
    assert chunk_shape is not None
    print(f"shape: {shape}")  # (128000, 212, 322, 2)
    print(f"chunk_shape: {chunk_shape}")  # (3, 53, 81, 1)
    chunk_coord_shape = [
        (shape[i] + chunk_shape[i] - 1) // chunk_shape[i] for i in range(len(shape))
    ]
    print(f"chunk_coord_shape: {chunk_coord_shape}")  # [42667, 4, 4, 2]
    num_chunks = np.prod(chunk_coord_shape)
    print(f"Number of chunks: {num_chunks}")  # 1365344 - around 1.3 million

    dsid = dset.id
    print(
        "Getting chunk info for chunk 0 (this takes a very long time because I think it is iterating through all the chunks)"
    )
    info = dsid.get_chunk_info(0)
    print(info)


if __name__ == "__main__":
    demonstrate_slow_get_chunk_info()
