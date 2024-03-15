import numpy as np


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

    return a == b


def _check_arrays_equal(a: np.ndarray, b: np.ndarray):
    # If it's an array of strings, we convert to an array of bytes
    if a.dtype == object:
        # need to modify all the entries
        a = np.array([x.encode() if type(x) is str else x for x in a.ravel()]).reshape(
            a.shape
        )
    if b.dtype == object:
        b = np.array([x.encode() if type(x) is str else x for x in b.ravel()]).reshape(
            b.shape
        )
    # if this is numeric data we need to use allclose so that we can handle NaNs
    if np.issubdtype(a.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)
    else:
        return np.array_equal(a, b)
