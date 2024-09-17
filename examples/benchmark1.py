import os
import h5py
import numpy as np
import time
import lindi
import gzip
import zarr
import numcodecs


def create_dataset(size):
    return np.random.rand(size)


def benchmark_h5py(file_path, num_small_datasets, num_large_datasets, small_size, large_size, chunks, compression, mode):
    start_time = time.time()

    if mode == 'dat':
        with open(file_path, 'wb') as f:
            # Write small datasets
            print('Writing small datasets')
            for i in range(num_small_datasets):
                data = create_dataset(small_size)
                f.write(data.tobytes())

            # Write large datasets
            print('Writing large datasets')
            for i in range(num_large_datasets):
                data = create_dataset(large_size)
                if compression == 'gzip':
                    data_zipped = gzip.compress(data.tobytes(), compresslevel=4)
                    f.write(data_zipped)
                elif compression is None:
                    f.write(data.tobytes())
                else:
                    raise ValueError(f"Unknown compressor: {compression}")
    elif mode == 'zarr':
        if os.path.exists(file_path):
            import shutil
            shutil.rmtree(file_path)
        store = zarr.DirectoryStore(file_path)
        root = zarr.group(store)

        if compression == 'gzip':
            compressor = numcodecs.GZip(level=4)
        else:
            compressor = None

        # Write small datasets
        print('Writing small datasets')
        for i in range(num_small_datasets):
            data = create_dataset(small_size)
            root.create_dataset(f'small_dataset_{i}', data=data)

        # Write large datasets
        print('Writing large datasets')
        for i in range(num_large_datasets):
            data = create_dataset(large_size)
            root.create_dataset(f'large_dataset_{i}', data=data, chunks=chunks, compressor=compressor)
    else:
        if mode == 'h5':
            f = h5py.File(file_path, 'w')
        else:
            f = lindi.LindiH5pyFile.from_lindi_file(file_path, mode='w')

        # Write small datasets
        print('Writing small datasets')
        for i in range(num_small_datasets):
            data = create_dataset(small_size)
            ds = f.create_dataset(f'small_dataset_{i}', data=data)
            ds.attrs['attr1'] = 1

        # Write large datasets
        print('Writing large datasets')
        for i in range(num_large_datasets):
            data = create_dataset(large_size)
            ds = f.create_dataset(f'large_dataset_{i}', data=data, chunks=chunks, compression=compression)
            ds.attrs['attr1'] = 1

        f.close()

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate total data size
    total_size = (num_small_datasets * small_size + num_large_datasets * large_size) * 8  # 8 bytes per float64
    total_size_gb = total_size / (1024 ** 3)

    print("Benchmark Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total data size: {total_size_gb:.2f} GB")
    print(f"Write speed: {total_size_gb / total_time:.2f} GB/s")

    h5py_file_size = os.path.getsize(file_path) / (1024 ** 3)
    print(f"File size: {h5py_file_size:.2f} GB")

    return total_time, total_size_gb


if __name__ == "__main__":
    file_path_h5 = "benchmark.h5"
    file_path_lindi = "benchmark.lindi.tar"
    file_path_dat = "benchmark.dat"
    file_path_zarr = "benchmark.zarr"
    num_small_datasets = 0
    num_large_datasets = 5
    small_size = 1000
    large_size = 100000000
    compression = None  # 'gzip' or None
    chunks = (large_size / 20,)

    print('Lindi Benchmark')
    lindi_time, total_size = benchmark_h5py(file_path_lindi, num_small_datasets, num_large_datasets, small_size, large_size, chunks=chunks, compression=compression, mode='lindi')
    print('')
    print('Zarr Benchmark')
    lindi_time, total_size = benchmark_h5py(file_path_zarr, num_small_datasets, num_large_datasets, small_size, large_size, chunks=chunks, compression=compression, mode='zarr')
    print('')
    print('H5PY Benchmark')
    h5py_time, total_size = benchmark_h5py(file_path_h5, num_small_datasets, num_large_datasets, small_size, large_size, chunks=chunks, compression=compression, mode='h5')
    print('')
    print('DAT Benchmark')
    dat, total_size = benchmark_h5py(file_path_dat, num_small_datasets, num_large_datasets, small_size, large_size, chunks=chunks, compression=compression, mode='dat')

    import shutil
    shutil.copyfile(file_path_lindi, file_path_lindi + '.tar')
