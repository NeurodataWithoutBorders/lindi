from typing import Any
import h5py


def _h5_attr_to_zarr_attr(attr: Any, *, label: str = ''):
    """Convert an attribute from h5py to a format that zarr can accept."""
    if isinstance(attr, bytes):
        return attr.decode('utf-8')  # is this reversible?
    elif isinstance(attr, (int, float, str)):
        return attr
    elif isinstance(attr, list):
        return [_h5_attr_to_zarr_attr(x) for x in attr]
    elif isinstance(attr, dict):
        return {k: _h5_attr_to_zarr_attr(v) for k, v in attr.items()}
    elif isinstance(attr, h5py.Reference):
        return _h5_ref_to_zarr_attr(attr, label=label)
    else:
        print(f'Warning: attribute of type {type(attr)} not handled: {label}')
        raise NotImplementedError()


def _h5_ref_to_zarr_attr(ref: h5py.Reference, *, label: str = ''):
    file = ref.file
    file_id = file.id

    # The get_name call can actually be quite slow. A possible way around this
    # is to do an initial pass through the file and build a map of object IDs to
    # paths. This would need to happen elsewhere in the code.
    deref_objname = h5py.h5r.get_name(ref, file_id)
    deref_objname = deref_objname.decode("utf-8")

    object_id = ref.attrs.get("object_id", None)

    # Here we assume that the file has a top-level attribute called "object_id".
    # This will be the case for files created by the LindiH5Store class.
    file_object_id = file.attrs.get("object_id", None)

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

    return {
        "_REFERENCE": value
    }
