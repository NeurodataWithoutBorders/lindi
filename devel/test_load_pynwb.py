import os
import urllib.request
import numpy as np
import pynwb
import h5py
import lindi
import json
import remfile


def test_load_pynwb():
    # https://neurosift.app/?p=/nwb&dandisetId=000939&dandisetVersion=0.240318.1555&url=https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/
    url_nwb = "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/"
    url = "https://kerchunk.neurosift.org/dandi/dandisets/000939/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/zarr.json"

    thisdir = os.path.dirname(os.path.abspath(__file__))
    fname = thisdir + "/test.zarr.json"
    if not os.path.exists(fname):
        _download_file(url, fname)

    remf = remfile.File(url_nwb)
    h5f0 = h5py.File(remf, mode="r")
    h5f = lindi.LindiH5pyFile.from_h5py_file(h5f0)

    store = lindi.LindiH5ZarrStore.from_file(url_nwb, url=url_nwb)
    rfs = store.to_reference_file_system()
    with open("test_rfs.zarr.json", "w") as f:
        json.dump(rfs, f, indent=2)
    hf5_rfs = lindi.LindiH5pyFile.from_reference_file_system(rfs)

    _compare_h5py_files(h5f, hf5_rfs)

    with pynwb.NWBHDF5IO(file=hf5_rfs, mode="r") as io1:
        nwb = io1.read()
        print(nwb)
        for k in nwb.fields:
            print(
                f"________________________________ {k} __________________________________"
            )
            print(getattr(nwb, k))


def _compare_h5py_files(h5f1: h5py.File, h5f2: h5py.File):
    _compare_h5py_groups(h5f1, h5f2, label="root")


def _compare_attrs(attrs1, attrs2):
    for k, v in attrs1.items():
        if k not in attrs2:
            print(f"*************** Attribute {k} not found in h5f2")
        elif not _check_attrs_equal(attrs2[k], v, label=k):
            print(f"*************** Attribute {k} mismatch")
            print(f"  h5f1: {v}")
            print(f"  h5f2: {attrs2[k]}")
    for k, v in attrs2.items():
        if k not in attrs1:
            print(f"*************** Attribute {k} not found in h5f1")
        elif not _check_attrs_equal(attrs1[k], v, label=k):
            print(f"*************** Attribute {k} mismatch")
            print(f"  h5f1: {attrs1[k]}")
            print(f"  h5f2: {v}")


def _check_attrs_equal(a, b, label: str):
    if isinstance(a, h5py.Reference):
        if isinstance(b, h5py.Reference):
            return True
        else:
            print(f'Not a reference: {label}')
            print(type(b))
            return False
    elif isinstance(b, h5py.Reference):
        if isinstance(a, h5py.Reference):
            return True
        else:
            print(f'Not a reference: {label}')
            print(type(a))
            return False
    else:
        return _check_equal(a, b)


def _compare_h5py_groups(g1: h5py.Group, g2: h5py.Group, label: str):
    print(f"Comparing group {label}")
    _compare_attrs(g1.attrs, g2.attrs)
    for k in g1.keys():
        if k not in g2.keys():
            print(f"*************** Key {k} not found in h5f2")
        else:
            obj1 = g1[k]
            obj2 = g2[k]
            if isinstance(obj1, h5py.Group):
                if not isinstance(obj2, h5py.Group):
                    print(f"*************** Object {k} is not a group in h5f2")
                else:
                    _compare_h5py_groups(obj1, obj2, label=f"{label}/{k}")
            elif isinstance(obj1, h5py.Dataset):
                if not isinstance(obj2, h5py.Dataset):
                    print(f"*************** Object {k} is not a dataset in h5f2")
                else:
                    _compare_h5py_datasets(obj1, obj2, label=f"{label}/{k}")
            else:
                raise Exception(f"Unhandled type: {type(obj1)}")
            if isinstance(obj1, h5py.Group):
                obj1x = g1.get(k, getlink=True)
                obj2x = g2.get(k, getlink=True)
                if isinstance(obj1x, h5py.SoftLink) or isinstance(obj1x, lindi.LindiH5pySoftLink):
                    if isinstance(obj2x, h5py.SoftLink) or isinstance(obj2x, lindi.LindiH5pySoftLink):
                        pass
                    else:
                        print(f"*************** Link type mismatch for {k}")
                        print(type(obj1x))
                        print(type(obj2x))
                elif isinstance(obj1x, h5py.HardLink) or isinstance(obj1x, lindi.LindiH5pyHardLink):
                    if isinstance(obj2x, h5py.HardLink) or isinstance(obj2x, lindi.LindiH5pyHardLink):
                        pass
                    else:
                        print(f"*************** Hard link type mismatch for {k}")
                        print(type(obj1x))
                        print(type(obj2x))
                elif isinstance(obj2x, h5py.SoftLink) or isinstance(obj2x, lindi.LindiH5pySoftLink):
                    print(f"*************** Link type mismatch for {k}")
                    print(type(obj1x))
                    print(type(obj2x))
                elif isinstance(obj2x, h5py.HardLink) or isinstance(obj2x, lindi.LindiH5pyHardLink):
                    print(f"*************** Link type mismatch for {k}")
                    print(type(obj1x))
                    print(type(obj2x))
    for k in g2.keys():
        if k not in g1.keys():
            print(f"*************** Key {k} not found in h5f1")


def _compare_h5py_datasets(d1: h5py.Dataset, d2: h5py.Dataset, label: str):
    print(f"Comparing dataset {label}")
    _compare_attrs(d1.attrs, d2.attrs)
    if d1.shape != d2.shape:
        print("*************** Shape mismatch")
    if d1.size != d2.size:
        print("*************** Size mismatch")
    if d1.dtype != d2.dtype:
        print("*************** Dtype mismatch")
    if d1.nbytes != d2.nbytes:
        print("*************** Nbytes mismatch")
    if d1.name != d2.name:
        print("*************** Name mismatch")
    if d1.ndim != d2.ndim:
        print("*************** Ndim mismatch")
    if d1.maxshape != d2.maxshape:
        print("*************** Maxshape mismatch")
    if d1.size and d1.size < 100:
        if not _check_equal(d1[()], d2[()]):
            print("*************** Data mismatch")
            print(f"  h5f1: {d1[()].ravel()[:5]}")
            print(f"  h5f2: {d2[()].ravel()[:5]}")


def _download_file(url, fname):
    print(f"Downloading {url} to {fname}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(fname, "wb") as f:
            f.write(response.read())


def _check_equal(a, b):
    # allow comparison of bytes and strings
    if isinstance(a, str):
        a = a.encode()
    if isinstance(b, str):
        b = b.encode()

    # allow comparison of numpy scalars with python scalars
    if np.issubdtype(type(a), np.floating):
        a = float(a)
    if np.issubdtype(type(b), np.floating):
        b = float(b)
    if np.issubdtype(type(a), np.integer):
        a = int(a)
    if np.issubdtype(type(b), np.integer):
        b = int(b)

    # allow comparison of numpy arrays to python lists
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)

    if type(a) != type(b):  # noqa: E721
        return False

    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        return _check_arrays_equal(a, b)

    # test for NaNs (we need to use np.isnan because NaN != NaN in python)
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True

    return a == b


def _check_arrays_equal(a: np.ndarray, b: np.ndarray):
    # If it's an array of strings, we convert to an array of bytes
    a = np.array([x.encode() if type(x) in (str, np.str_) else x for x in a.ravel()]).reshape(
        a.shape
    )
    b = np.array([x.encode() if type(x) in (str, np.str_) else x for x in b.ravel()]).reshape(
        b.shape
    )
    # if this is numeric data we need to use allclose so that we can handle NaNs
    if np.issubdtype(a.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)
    else:
        return np.array_equal(a, b)


if __name__ == "__main__":
    test_load_pynwb()
