from typing import Any
import numpy as np
import h5py


def _h5_attr_to_zarr_attr(attr: Any, *, label: str = '', h5f: h5py.File):
    """Convert an attribute from h5py to a format that zarr can accept.

    bytes -> decoded utf-8 string
    int, float, str -> unchanged
    list -> recursively convert each element
    dict -> recursively convert each value
    h5py.Reference -> convert to a reference object, see _h5_ref_to_zarr_attr

    Otherwise, raise NotImplementedError
    """

    # first disallow special strings
    special_strings = ['NaN', 'Infinity', '-Infinity']
    if isinstance(attr, str) and attr in special_strings:
        raise ValueError(f"Special string {attr} not allowed in attribute value at {label}")
    if isinstance(attr, bytes) and attr in [x.encode('utf-8') for x in special_strings]:
        raise ValueError(f"Special string {attr} not allowed in attribute value at {label}")

    if attr is None:
        return None
    elif isinstance(attr, bytes):
        return attr.decode('utf-8')
    elif isinstance(attr, (int, float, str)):
        return attr
    elif np.issubdtype(type(attr), np.integer):
        return int(attr)
    elif np.issubdtype(type(attr), np.floating):
        return float(attr)
    elif np.issubdtype(type(attr), np.bool_):
        return bool(attr)
    elif type(attr) is np.bytes_:
        return attr.tobytes().decode('utf-8')
    elif isinstance(attr, h5py.Reference):
        return _h5_ref_to_zarr_attr(attr, label=label + '._REFERENCE', h5f=h5f)
    elif isinstance(attr, list):
        return [_h5_attr_to_zarr_attr(x, label=label, h5f=h5f) for x in attr]
    elif isinstance(attr, dict):
        return {k: _h5_attr_to_zarr_attr(v, label=label, h5f=h5f) for k, v in attr.items()}
    elif isinstance(attr, np.ndarray):
        return _h5_attr_to_zarr_attr(attr.tolist(), label=label, h5f=h5f)
    else:
        print(f'Warning: attribute of type {type(attr)} not handled: {label}')
        raise NotImplementedError()


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

    # This is how hdmf_zarr does it, but I would propose to use a _REFERENCE key
    # instead. Note that we will also need to handle "region" references. I
    # would propose another field in the value containing the region info. See
    # https://hdmf-zarr.readthedocs.io/en/latest/storage.html#sec-zarr-storage-references-region

    # return {
    #     "zarr_dtype": "object",
    #     "value": value
    # }

    # important to run it through _h5_attr_to_zarr_attr to handle object IDs of
    # type bytes
    return _h5_attr_to_zarr_attr({
        "_REFERENCE": value
    }, label=label, h5f=h5f)
