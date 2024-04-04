import h5py


def h5_ref_to_zarr_attr(ref: h5py.Reference, *, h5f: h5py.File):
    """Convert/encode an h5py reference to a format that zarr can accept.

    Parameters
    ----------
    ref : h5py.Reference
        The reference to convert.
    h5f : h5py.File
        The file that the reference is in.

    Returns
    -------
    dict
        The reference in a format that zarr can accept.

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
    dref_obj = h5f[ref]
    deref_objname = dref_obj.name

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
