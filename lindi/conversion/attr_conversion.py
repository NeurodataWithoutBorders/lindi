from typing import Any, Union
import numpy as np
import h5py
from .nan_inf_ninf import encode_nan_inf_ninf
from .h5_ref_to_zarr_attr import h5_ref_to_zarr_attr


def h5_to_zarr_attr(attr: Any, *, label: str = '', h5f: Union[h5py.File, None]):
    """Convert an attribute from h5py to a format that zarr can accept."""

    from ..LindiH5pyFile.LindiH5pyReference import LindiH5pyReference  # Avoid circular import

    # Do not allow these special strings in attributes
    special_strings = ['NaN', 'Infinity', '-Infinity']

    if isinstance(attr, list):
        list_dtype = _determine_list_dtype(attr)
        attr = np.array(attr, dtype=list_dtype)

    if isinstance(attr, str) and attr in special_strings:
        raise ValueError(f"Special string {attr} not allowed in attribute value at {label}")
    if attr is None:
        raise Exception(f"Unexpected h5 attribute: None at {label}")
    elif type(attr) in [int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
        return int(attr)
    elif isinstance(attr, (float, np.floating)):
        return encode_nan_inf_ninf(float(attr))
    elif isinstance(attr, (complex, np.complexfloating)):
        raise Exception(f"Complex number is not supported at {label}")
    elif type(attr) in [bool, np.bool_]:
        return bool(attr)
    elif isinstance(attr, (bool, list, tuple, dict, set)):
        raise Exception(f"Unexpected type for h5 attribute: {type(attr)} at {label}")
    elif isinstance(attr, str):
        return attr
    elif isinstance(attr, bytes):
        return attr.decode('utf-8')
    elif isinstance(attr, np.ndarray):
        if attr.dtype.kind in ['i', 'u']:
            return attr.tolist()  # this will be a nested list of type int
        elif attr.dtype.kind in ['f']:
            return encode_nan_inf_ninf(attr.tolist())  # this will be a nested list of type float
        elif attr.dtype.kind in ['c']:
            raise Exception(f"Arrays of complex numbers are not supported at {label}")
        elif attr.dtype.kind == 'b':
            return attr.tolist()  # this will be a nested list of type bool
        elif attr.dtype.kind == 'O':
            x = attr.tolist()
            if not _nested_list_has_all_strings(x):
                raise Exception(f"Not allowed for attribute: numpy array with dtype=object that contains non-string elements at {label}")
            return x
        elif attr.dtype.kind == 'U':
            return _decode_bytes_to_str_in_nested_list(attr.tolist())
        elif attr.dtype.kind == 'S':
            return _decode_bytes_to_str_in_nested_list(attr.tolist())
        else:
            raise Exception(f"Unexpected dtype for attribute numpy array: {attr.dtype} at {label}")
    elif isinstance(attr, LindiH5pyReference):
        return {
            '_REFERENCE': attr._obj
        }
    elif isinstance(attr, h5py.Reference):
        if h5f is None:
            raise Exception(f"h5f cannot be None when converting h5py.Reference to zarr attribute at {label}")
        return h5_ref_to_zarr_attr(attr, h5f=h5f)
    else:
        raise Exception(f"Unexpected type for h5 attribute: {type(attr)} at {label}")


def _decode_bytes_to_str_in_nested_list(x):
    if isinstance(x, bytes):
        return x.decode('utf-8')
    elif isinstance(x, str):
        return x
    elif isinstance(x, list):
        return [_decode_bytes_to_str_in_nested_list(y) for y in x]
    else:
        raise Exception("Unexpected type in _decode_bytes_to_str")


def zarr_to_h5_attr(attr: Any):
    """Convert an attribute from zarr to a format that h5py expects."""
    if isinstance(attr, str):
        return attr
    elif isinstance(attr, int):
        return attr
    elif isinstance(attr, float):
        return attr
    elif isinstance(attr, bool):
        return np.bool_(attr)
    elif isinstance(attr, list):
        if _nested_list_has_all_strings(attr):
            return np.array(attr, dtype='O')
        elif _nested_list_has_all_ints(attr):
            return np.array(attr, dtype='int64')
        elif _nested_list_has_all_floats_or_ints(attr):
            return np.array(attr, dtype='float64')
        elif _nested_list_has_all_bools(attr):
            return np.array(attr, dtype='bool')
        else:
            raise Exception("Nested list contains mixed types")
    else:
        raise Exception(f"Unexpected type in zarr attribute: {type(attr)}")


def _nested_list_has_all_strings(x):
    if isinstance(x, str):
        return True
    elif isinstance(x, list):
        return all(_nested_list_has_all_strings(y) for y in x)
    else:
        return False


def _nested_list_has_all_ints(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, list):
        return all(_nested_list_has_all_ints(y) for y in x)
    else:
        return False


def _nested_list_has_all_floats_or_ints(x):
    if isinstance(x, (int, float)):
        return True
    elif isinstance(x, list):
        return all(_nested_list_has_all_floats_or_ints(y) for y in x)
    else:
        return False


def _nested_list_has_all_bools(x):
    if isinstance(x, bool):
        return True
    elif isinstance(x, list):
        return all(_nested_list_has_all_bools(y) for y in x)
    else:
        return False


def _determine_list_dtype(x):
    x_flattened = _flatten_list(x)
    if len(x_flattened) == 0:
        return np.dtype(np.int64)
    if all(isinstance(i, int) for i in x_flattened):
        return np.int64
    elif all(isinstance(i, float) for i in x_flattened):
        return np.float64
    elif all(isinstance(i, bool) for i in x_flattened):
        return np.bool_
    elif all(isinstance(i, str) for i in x_flattened):
        return np.dtype('O')
    else:
        raise Exception("Mixed types in list")


def _flatten_list(x):
    if isinstance(x, list):
        return [a for i in x for a in _flatten_list(i)]
    else:
        return [x]
