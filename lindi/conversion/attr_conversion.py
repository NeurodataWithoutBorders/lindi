from typing import Any, Union
import numpy as np
import h5py
from .nan_inf_ninf import encode_nan_inf_ninf


def h5_to_zarr_attr(attr: Any, *, label: str = '', h5f: Union[h5py.File, None]):
    """Convert an attribute from h5py to a format that zarr can accept."""
    special_strings = ['NaN', 'Infinity', '-Infinity']
    if isinstance(attr, str) and attr in special_strings:
        raise ValueError(f"Special string {attr} not allowed in attribute value at {label}")
    if attr is None:
        raise Exception(f"Unexpected h5 attribute: None at {label}")
    elif type(attr) in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
        return int(attr)
    elif type(attr) in [np.float16, np.float32, np.float64, np.float128]:
        return encode_nan_inf_ninf(attr)
    elif type(attr) in [np.complex64, np.complex128, np.complex256]:
        raise Exception(f"Complex number is not supported at {label}")
    elif type(attr) in [np.bool_]:
        return bool(attr)
    elif isinstance(attr, (bool, list, tuple, bytes, dict, set)):
        raise Exception(f"Unexpected type for h5 attribute: {type(attr)} at {label}")
    elif isinstance(attr, str):
        return attr
    elif isinstance(attr, np.ndarray):
        if attr.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            return attr.tolist()  # this will be a nested list of type int
        elif attr.dtype in [np.float16, np.float32, np.float64, np.float128]:
            return encode_nan_inf_ninf(attr.tolist())  # this will be a nested list of type float
        elif attr.dtype in [np.complex64, np.complex128, np.complex256]:
            raise Exception(f"Arrays of complex numbers are not supported at {label}")
        elif attr.dtype in [np.bool_]:
            return attr.tolist()  # this will be a nested list of type bool
        elif attr.dtype.kind == 'O':
            x = attr.tolist()
            if not _nested_list_has_all_strings(x):
                raise Exception(f"Not allowed: numpy array with dtype=object that contains non-string elements at {label}")
            return x
        else:
            raise Exception(f"Unexpected dtype for numpy array: {attr.dtype} at {label}")
    elif isinstance(attr, h5py.Reference):
        if h5f is None:
            raise Exception(f"h5f cannot be None when converting h5py.Reference to zarr attribute at {label}")
        return _h5_ref_to_zarr_attr(attr, label=label + '._REFERENCE', h5f=h5f)
    else:
        raise Exception(f"Unexpected type for h5 attribute: {type(attr)} at {label}")


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


def _h5_ref_to_zarr_attr(ref: h5py.Reference, *, label: str = '', h5f: h5py.File):
    """Convert an h5py reference to a format that zarr can accept.

    The format is a dictionary with a single key, '_REFERENCE', whose value is
    another dictionary with the following keys:

    'object_id', 'path', 'source', 'source_object_id'

    * object_id is the object ID of the target object.
    * path is the path of the target object.
    * source is always '.', meaning that path is relative to the root of the
      file (I think)
    * source_object_id is the object ID of the source object.

    See
    https://hdmf-zarr.readthedocs.io/en/latest/storage.html#storing-object-references-in-attributes

    Note that we will also need to handle "region" references. I would propose
    another field in the value containing the region info. See
    https://hdmf-zarr.readthedocs.io/en/latest/storage.html#sec-zarr-storage-references-region
    """
    file_id = h5f.id

    # The get_name call can actually be quite slow. A possible way around this
    # is to do an initial pass through the file and build a map of object IDs to
    # paths. This would need to happen elsewhere in the code.
    deref_objname = h5py.h5r.get_name(ref, file_id)
    if deref_objname is None:
        raise ValueError(f"Could not dereference object with reference {ref}")
    deref_objname = deref_objname.decode("utf-8")

    dref_obj = h5f[deref_objname]
    object_id = dref_obj.attrs.get("object_id", None)

    # Here we assume that the file has a top-level attribute called "object_id".
    # This will be the case for files created by the LindiH5ZarrStore class.
    file_object_id = h5f.attrs.get("object_id", None)

    # See https://hdmf-zarr.readthedocs.io/en/latest/storage.html#storing-object-references-in-attributes
    value = {
        "object_id": object_id,
        "path": deref_objname,
        "source": ".",  # Are we always going to use the top-level object as the source?
        "source_object_id": file_object_id,
    }

    # We need this to be json serializable
    for k, v in value.items():
        if isinstance(v, bytes):
            value[k] = v.decode('utf-8')

    return {
        "_REFERENCE": value
    }
