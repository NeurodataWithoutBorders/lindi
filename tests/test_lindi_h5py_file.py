import tempfile
import os
import pytest
import h5py
import lindi
from .utils import assert_h5py_files_equal


def test_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_fname = f'{tmpdir}/test.h5'
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        lindi_tar_fname = f'{tmpdir}/test.lindi.tar'
        lindi_d_fname = f'{tmpdir}/test.lindi.d'

        create_example_h5_file(h5_fname)

        with lindi.LindiH5pyFile.from_hdf5_file(h5_fname, url=h5_fname) as f:
            f.write_lindi_file(lindi_json_fname)

        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
            f.write_lindi_file(lindi_tar_fname)

        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
            f.write_lindi_file(lindi_d_fname)

        assert os.path.isdir(lindi_d_fname)

        with h5py.File(h5_fname, 'r') as h5f:
            with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
                assert_h5py_files_equal(h5f, f, skip_large_datasets=False)

            with lindi.LindiH5pyFile.from_lindi_file(lindi_tar_fname) as f:
                assert_h5py_files_equal(h5f, f, skip_large_datasets=False)

            with lindi.LindiH5pyFile.from_lindi_file(lindi_d_fname) as f:
                assert_h5py_files_equal(h5f, f, skip_large_datasets=False)


def test_fail_open_hdf5_in_write_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_fname = f'{tmpdir}/test.h5'
        create_example_h5_file(h5_fname)
        with lindi.LindiH5pyFile.from_hdf5_file(h5_fname, url=h5_fname, mode='r'):
            pass
        with pytest.raises(ValueError):
            with lindi.LindiH5pyFile.from_hdf5_file(h5_fname, url=h5_fname, mode='w'):
                pass


def test_create_new_lindi_json_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
            assert f.attrs['attr1'] == 'value1'


@pytest.mark.network
def test_load_remote_lindi_json_file():
    # https://neurosift.app/?p=/nwb&url=https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/&dandisetId=000409&dandisetVersion=draft
    url_lindi_json = 'https://lindi.neurosift.org/dandi/dandisets/000409/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/nwb.lindi.json'
    url_hdf5 = 'https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/'
    f1 = lindi.LindiH5pyFile.from_lindi_file(url_lindi_json)
    f2 = lindi.LindiH5pyFile.from_hdf5_file(url_hdf5)
    assert_h5py_files_equal(f1, f2, skip_large_datasets=True)


def test_fail_open_non_existing_file_for_reading():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            with lindi.LindiH5pyFile.from_lindi_file(f'{tmpdir}/non_existing_file.lindi.json', mode='r'):
                pass
        with pytest.raises(FileNotFoundError):
            with lindi.LindiH5pyFile.from_lindi_file(f'{tmpdir}/non_existing_file.lindi.json', mode='r+'):
                pass


def test_fail_open_existing_file_for_new_writing():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f'{tmpdir}/existing_file.lindi.json', 'w') as f:
            f.write('test')
        with pytest.raises(ValueError):
            with lindi.LindiH5pyFile.from_lindi_file(f'{tmpdir}/existing_file.lindi.json', mode='w-'):
                pass


def test_create_lindi_json_file_in_x_mode():
    # w- and x are equivalent
    for mode in ['w-', 'x']:
        with tempfile.TemporaryDirectory() as tmpdir:
            lindi_json_fname = f'{tmpdir}/test.lindi.json'
            with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode=mode) as f:  # type: ignore
                f.attrs['attr1'] = 'value1'
            with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
                assert f.attrs['attr1'] == 'value1'


def test_append_to_lindi_json_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='a') as f:
            f.attrs['attr1'] = 'value1'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='a') as f:
            f.attrs['attr2'] = 2
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
            assert f.attrs['attr1'] == 'value1'
            assert f.attrs['attr2'] == 2


def test_rfs_for_lindi_tar_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_tar_fname = f'{tmpdir}/test.lindi.tar'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_tar_fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_tar_fname) as f:
            assert f.attrs['attr1'] == 'value1'
            rfs = f.to_reference_file_system()
            assert rfs['refs']['.zattrs']['attr1'] == 'value1'


def test_fail_write_lindi_invalid_extension():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
        with pytest.raises(ValueError):
            with lindi.LindiH5pyFile.from_lindi_file(lindi_fname) as f:
                f.write_lindi_file(f'{tmpdir}/test.lindi.invalid_extension')


def test_fail_write_lindi_json_from_local_lindi_tar():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_tar_fname = f'{tmpdir}/test.lindi.tar'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_tar_fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
        with pytest.raises(ValueError):
            with lindi.LindiH5pyFile.from_lindi_file(lindi_tar_fname) as f:
                f.write_lindi_file(f'{tmpdir}/test.lindi.json')


@pytest.mark.network
def test_create_local_lindi_json_from_remote_lindi_tar():
    # This example will probably disappear in the future
    # and will need to be replaced with another example
    # https://neurosift.app/?p=/nwb&url=https://tempory.net/f/dendro/f/hello_world_service/hello_neurosift/spike_sorting_post_processing/FKu2zK3TAsehGtJJXjQa/output/post.nwb.lindi.tar&dandisetId=215561&dandisetVersion=draft&st=lindi
    url = 'https://tempory.net/f/dendro/f/hello_world_service/hello_neurosift/spike_sorting_post_processing/FKu2zK3TAsehGtJJXjQa/output/post.nwb.lindi.tar'
    with tempfile.TemporaryDirectory() as tmpdir:
        with lindi.LindiH5pyFile.from_lindi_file(url) as f:
            f.write_lindi_file(f'{tmpdir}/test.lindi.json')
        f1 = lindi.LindiH5pyFile.from_lindi_file(url)
        f2 = lindi.LindiH5pyFile.from_lindi_file(f'{tmpdir}/test.lindi.json')
        assert_h5py_files_equal(f1, f2, skip_large_datasets=True)


def test_write_lindi_json_with_generation_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname)
        f.write_lindi_file(f'{tmpdir}/test.lindi.json', generation_metadata={'test': 1})
        g = lindi.LindiH5pyFile.from_lindi_file(f'{tmpdir}/test.lindi.json')
        rfs = g.to_reference_file_system()
        assert rfs['generationMetadata']['test'] == 1


def test_misc_coverage():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname)
        assert f.filename == ''
        with pytest.raises(Exception):
            f.driver
        assert f.mode == 'r'
        with pytest.raises(Exception):
            f.libver
        with pytest.raises(Exception):
            f.userblock_size
        with pytest.raises(Exception):
            f.meta_block_size
        with pytest.raises(Exception):
            f.swmr_mode(1)
        assert isinstance(str(f), str)
        assert isinstance(repr(f), str)
        assert f.__bool__() is True
        assert f.__hash__()
        assert f.id
        assert f.file
        assert f.name
        # cannot get ref on readonly object
        with pytest.raises(ValueError):
            f.ref


def test_delete_group():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='w') as f:
            with pytest.raises(Exception):
                f.create_group('group1', track_order=True)
            f.require_group('group1')
            f.create_group('group2')
            f.require_group('group2')
            with pytest.raises(Exception):
                f.create_group('group2')
            f.create_group('group3')
            del f['group2']
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
            assert 'group1' in f
            assert 'group2' not in f
            assert 'group3' in f


def test_copy_lindi_to_lindi():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='w') as f:
            f.create_group('group1')
            group1 = f['group1']
            assert isinstance(group1, lindi.LindiH5pyGroup)
            group1.attrs['attr1'] = 'value1'
            group2 = group1.create_group('group2')
            group2.attrs['attr2'] = 2
            group2.create_dataset('dataset1', data=[1, 2, 3])
            group2.require_dataset('dataset1', shape=(3,), dtype=int)
            f.copy('group1', f, 'group3')
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname)
        assert 'group1' in f
        assert 'group2' in f['group1']  # type: ignore
        assert 'group3' in f
        assert 'group2' in f['group3']  # type: ignore
        assert f['group1'].attrs['attr1'] == 'value1'  # type: ignore
        assert f['group3'].attrs['attr1'] == 'value1'  # type: ignore
        assert f['group3']['group2'].attrs['attr2'] == 2  # type: ignore
        ds = f['group3']['group2']['dataset1']  # type: ignore
        assert isinstance(ds, lindi.LindiH5pyDataset)
        assert ds.shape == (3,)


def test_copy_lindi_to_hdf5():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        h5_fname = f'{tmpdir}/test.h5'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='w') as lindi_f:
            lindi_f.create_group('group1')
            group1 = lindi_f['group1']
            assert isinstance(group1, lindi.LindiH5pyGroup)
            group1.attrs['attr1'] = 'value1'
            group2 = group1.create_group('group2')
            group2.attrs['attr2'] = 2
            ds = group2.create_dataset('dataset1', data=[1, 2, 3])
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='r')
        with h5py.File(h5_fname, 'w') as h5f:
            with pytest.raises(Exception):
                f.copy('group1', h5f, 'group1_copy', shallow=True)
            with pytest.raises(Exception):
                f.copy('group1', h5f, 'group1_copy', expand_soft=True)
            with pytest.raises(Exception):
                f.copy('group1', h5f, 'group1_copy', expand_external=True)
            with pytest.raises(Exception):
                f.copy('group1', h5f, 'group1_copy', expand_refs=True)
            with pytest.raises(Exception):
                f.copy('group1', h5f, 'group1_copy', without_attrs=True)
            with pytest.raises(Exception):
                f.copy('group1', h5f, None)
            f.copy('group1', h5f, 'group1_copy')
        with h5py.File(h5_fname, 'r') as h5f:
            assert 'group1_copy' in h5f
            assert 'group2' in h5f['group1_copy']  # type: ignore
            assert h5f['group1_copy'].attrs['attr1'] == 'value1'
            assert h5f['group1_copy']['group2'].attrs['attr2'] == 2  # type: ignore
            ds = h5f['group1_copy']['group2']['dataset1']  # type: ignore
            assert isinstance(ds, h5py.Dataset)
            assert ds.shape == (3,)


def test_soft_link():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_fname = f'{tmpdir}/test.h5'
        with h5py.File(h5_fname, 'w') as h5f:
            group1 = h5f.create_group('group1')
            group1.attrs['attr1'] = 'value1'
            h5f['group_sl'] = h5py.SoftLink('group1')
        f = lindi.LindiH5pyFile.from_hdf5_file(h5_fname, url=h5_fname)
        f.write_lindi_file(f'{tmpdir}/test.lindi.json')
        f.close()
        g = lindi.LindiH5pyFile.from_lindi_file(f'{tmpdir}/test.lindi.json')
        assert 'group_sl' in g
        aa = g.get('group_sl', getlink=True)
        assert isinstance(aa, h5py.SoftLink)
        bb = g['group_sl']
        assert isinstance(bb, lindi.LindiH5pyGroup)
        assert bb.attrs['attr1'] == 'value1'


def test_reference():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_fname = f'{tmpdir}/test.h5'
        with h5py.File(h5_fname, 'w') as h5f:
            group1 = h5f.create_group('group1')
            dataset1 = group1.create_dataset('dataset1', data=[1, 2, 3])
            dataset1.attrs['attr1'] = 'value1'
            h5f.attrs['ref1'] = dataset1.ref
        f = lindi.LindiH5pyFile.from_hdf5_file(h5_fname, url=h5_fname)
        for k, _ in f.items():
            if k != 'group1':
                raise Exception(f'Unexpected key: {k}')
        for k in f:
            if k != 'group1':
                raise Exception(f'Unexpected key: {k}')
        f.write_lindi_file(f'{tmpdir}/test.lindi.json')
        f.close()
        g = lindi.LindiH5pyFile.from_lindi_file(f'{tmpdir}/test.lindi.json')
        ref1 = g.attrs['ref1']
        assert isinstance(ref1, h5py.Reference)
        with pytest.raises(Exception):
            g.get(ref1, getlink=True)
        with pytest.raises(Exception):
            g.get(ref1, getclass=True)
        b = g[ref1]
        assert isinstance(b, lindi.LindiH5pyDataset)
        assert b.attrs['attr1'] == 'value1'


def test_fail_attempt_write_in_read_only_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='w') as f:
            f.attrs['attr1'] = 'value1'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='r') as f:
            with pytest.raises(ValueError):
                f.attrs['attr2'] = 2
            with pytest.raises(ValueError):
                f.create_group('group1')
            with pytest.raises(ValueError):
                f.create_dataset('dataset1', data=[1, 2, 3])
            with pytest.raises(ValueError):
                f.require_group('group1')
            with pytest.raises(ValueError):
                f.require_dataset('dataset1', shape=(3,), dtype=int)


def test_create_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        lindi_json_fname = f'{tmpdir}/test.lindi.json'
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname, mode='w') as f:
            f.create_dataset('dataset1', data=[1, 2, 3])
            f.require_dataset('dataset1', shape=(3,), dtype=int)
        with lindi.LindiH5pyFile.from_lindi_file(lindi_json_fname) as f:
            assert 'dataset1' in f
            ds = f['dataset1']
            assert isinstance(ds, lindi.LindiH5pyDataset)
            assert ds.shape == (3,)


def create_example_h5_file(fname):
    with h5py.File(fname, 'w') as f:
        f.attrs['attr1'] = 'value1'
        f.attrs['attr2'] = 2
        f.create_dataset('dataset1', data=[1, 2, 3])
        f.create_group('group1')
        f.create_group('group2')
        group1 = f['group1']
        assert isinstance(group1, h5py.Group)
        group1.create_dataset('dataset2', data=[4, 5, 6])
