import tempfile
import os
import numpy as np
import lindi
import shutil


def test_staging_area():
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_area = lindi.StagingArea.create(base_dir=tmpdir + '/staging_area')
        client = lindi.LindiH5pyFile.from_reference_file_system(None, mode='r+', staging_area=staging_area)
        X = np.random.randn(1000, 1000).astype(np.float32)
        client.create_dataset('large_array', data=X, chunks=(400, 400))
        total_size = _get_total_size_of_directory(tmpdir)
        assert total_size >= X.nbytes * 0.5, f'{total_size} < {X.nbytes} * 0.5'  # take into consideration compression
        rfs = client.to_reference_file_system()
        client2 = lindi.LindiH5pyFile.from_reference_file_system(rfs, mode='r')
        assert isinstance(client2, lindi.LindiH5pyFile)
        X1 = client['large_array']
        assert isinstance(X1, lindi.LindiH5pyDataset)
        X2 = client2['large_array']
        assert isinstance(X2, lindi.LindiH5pyDataset)
        assert np.allclose(X1[:], X2[:])

        upload_dir = f'{tmpdir}/upload_dir'
        os.makedirs(upload_dir, exist_ok=True)
        output_fname = f'{tmpdir}/output.lindi.json'

        def on_upload_blob(fname: str):
            random_fname = f'{upload_dir}/{_random_string(10)}'
            shutil.copy(fname, random_fname)
            return random_fname

        def on_upload_main(fname: str):
            shutil.copy(fname, output_fname)
            return output_fname

        assert client.staging_store
        client.upload(
            on_upload_blob=on_upload_blob,
            on_upload_main=on_upload_main
        )

        client3 = lindi.LindiH5pyFile.from_lindi_file(output_fname, mode='r')
        X3 = client3['large_array']
        assert isinstance(X3, lindi.LindiH5pyDataset)
        assert np.allclose(X1[:], X3[:])


def _get_total_size_of_directory(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def _random_string(n):
    import random
    import string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


if __name__ == '__main__':
    test_staging_area()
