# Special Zarr Annotations in LINDI

LINDI defines a set of special Zarr annotations to correspond with HDF5 features that are not natively supported in Zarr.

## Scalar Datasets

### `_SCALAR = True`

In HDF5, datasets can be scalar, but Zarr does not natively support scalar arrays. To overcome this limitation, LINDI represents scalar datasets as Zarr arrays with a shape of `(1,)` and marks them with the `_SCALAR = True` attribute.

## Soft Links

### `_SOFT_LINK = { 'path': '...' }`

Soft links in HDF5 are pointers to other groups within the file. LINDI utilizes the `_SOFT_LINK` attribute on a Zarr group to represent this relationship. The `path` key within the attribute specifies the target group within the Zarr structure. Soft link groups in Zarr should be otherwise empty, serving only as a reference to another location in the dataset.

Note that we do not currently support external links.

## References

```json
{
  "_REFERENCE": {
    "source": ".",
    "path": "...",
    "object_id": "...",
    "source_object_id": "...",
  }
}
```

- `source`: Always `.` for now, indicating that the reference is to an object within the same Zarr.
- `path`: Path to the target object within the Zarr.
- `object_id`: The object_id attribute of the target object (for validation).
- `source_object_id`: The object_id attribute of the source object (for validation).

The largely follows the [convention used by hdmf-zarr](https://hdmf-zarr.readthedocs.io/en/latest/storage.html#storing-object-references-in-attributes). 

HDF5 references can appear within both attributes and datasets. For attributes, the value of the attribute is a dict in the above form. For datasets, the value of an item within the dataset is a dict in the above form.

**Note**: Region references are not supported at this time and are planned for future implementation.

## Compound Data Types

### `_COMPOUND_DTYPE: [['x', 'int32'], ['y', 'float64'], ...]`

Zarr arrays can represent compound data types from HDF5 datasets. The `_COMPOUND_DTYPE` attribute on a Zarr array indicates this, listing each field's name and data type. The array data should be JSON encoded, aligning with the specified compound structure. The `h5py.Reference` type is also supported within these structures, enabling references within compound data types.

## External Array Links

### `_EXTERNAL_ARRAY_LINK = {'link_type': 'hdf5_dataset', 'url': '...', 'name': '...'}`

For datasets with an extensive number of chunks such that inclusion in the Zarr or reference file system is impractical, LINDI uses the `_EXTERNAL_ARRAY_LINK` attribute on a Zarr array. This attribute points to an external HDF5 file, specifying the `url` for remote access (or local path) and the `name` of the target dataset within that file. When slicing that dataset, the `LindiH5pyClient` will handle data retrieval, leveraging `h5py` and `remfile` for remote access.