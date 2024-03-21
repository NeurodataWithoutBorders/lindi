from typing import TYPE_CHECKING, Union, Any, Dict
import numpy as np
import h5py
import zarr
import remfile

from .LindiH5pyAttributes import LindiH5pyAttributes
from .LindiH5pyReference import LindiH5pyReference


if TYPE_CHECKING:
    from .LindiH5pyFile import LindiH5pyFile  # pragma: no cover


class LindiH5pyDatasetId:
    def __init__(self, _h5py_dataset_id):
        self._h5py_dataset_id = _h5py_dataset_id


# This is a global list of external hdf5 clients, which are used by
# possibly multiple LindiH5pyFile objects. The key is the URL of the
# external hdf5 file, and the value is the h5py.File object.
# TODO: figure out how to close these clients
_external_hdf5_clients: Dict[str, h5py.File] = {}


class LindiH5pyDataset(h5py.Dataset):
    def __init__(self, _dataset_object: Union[h5py.Dataset, zarr.Array], _file: "LindiH5pyFile"):
        self._dataset_object = _dataset_object
        self._file = _file

        # See if we have the _COMPOUND_DTYPE attribute, which signifies that
        # this is a compound dtype
        if isinstance(_dataset_object, zarr.Array):
            compound_dtype_obj = _dataset_object.attrs.get("_COMPOUND_DTYPE", None)
            if compound_dtype_obj is not None:
                # If we have a compound dtype, then create the numpy dtype
                self._compound_dtype = np.dtype(
                    [(compound_dtype_obj[i][0], compound_dtype_obj[i][1]) for i in range(len(compound_dtype_obj))]
                )
            else:
                self._compound_dtype = None
        else:
            self._compound_dtype = None

        # Check whether this is a scalar dataset
        if isinstance(_dataset_object, zarr.Array):
            self._is_scalar = self._dataset_object.attrs.get("_SCALAR", False)
        else:
            self._is_scalar = self._dataset_object.ndim == 0

    @property
    def id(self):
        if isinstance(self._dataset_object, h5py.Dataset):
            return LindiH5pyDatasetId(self._dataset_object.id)
        else:
            return LindiH5pyDatasetId(None)

    @property
    def shape(self):  # type: ignore
        if self._is_scalar:
            return ()
        return self._dataset_object.shape

    @property
    def size(self):
        if self._is_scalar:
            return 1
        return self._dataset_object.size

    @property
    def dtype(self):
        if self._compound_dtype is not None:
            return self._compound_dtype
        ret = self._dataset_object.dtype
        if ret.kind == 'O':
            if not ret.metadata:
                # The following correction is needed because of
                # this code in hdmf/backends/hdf5/h5tools.py:
                # def _check_str_dtype(self, h5obj):
                #     dtype = h5obj.dtype
                #     if dtype.kind == 'O':
                #         if dtype.metadata.get('vlen') == str and H5PY_3:
                #             return StrDataset(h5obj, None)
                #     return h5obj
                # We cannot have a dtype with kind 'O' and no metadata
                ret = np.dtype(str(ret), metadata={})
        return ret

    @property
    def nbytes(self):
        return self._dataset_object.nbytes

    @property
    def file(self):
        return self._file

    @property
    def name(self):
        return self._dataset_object.name

    @property
    def maxshape(self):
        # not sure what to return here, so let's return self.shape rather than self._h5py_dataset.maxshape
        # return self._h5py_dataset.maxshape
        return self.shape

    @property
    def ndim(self):
        if self._is_scalar:
            return 0
        return self._dataset_object.ndim

    @property
    def attrs(self):  # type: ignore
        if isinstance(self._dataset_object, h5py.Dataset):
            attrs_type = 'h5py'
        elif isinstance(self._dataset_object, zarr.Array):
            attrs_type = 'zarr'
        else:
            raise Exception(f'Unexpected dataset object type: {type(self._dataset_object)}')
        return LindiH5pyAttributes(self._dataset_object.attrs, attrs_type=attrs_type)

    def __getitem__(self, args, new_dtype=None):
        if isinstance(self._dataset_object, h5py.Dataset):
            ret = self._dataset_object.__getitem__(args, new_dtype)
        elif isinstance(self._dataset_object, zarr.Array):
            if new_dtype is not None:
                raise Exception("new_dtype is not supported for zarr.Array")
            ret = self._get_item_for_zarr(self._dataset_object, args)
            ret = _resolve_references(ret)
        else:
            raise Exception(f"Unexpected type: {type(self._dataset_object)}")
        return ret

    def _get_item_for_zarr(self, zarr_array: zarr.Array, selection: Any):
        # First check whether this is an external array link
        external_array_link = zarr_array.attrs.get("_EXTERNAL_ARRAY_LINK", None)
        if external_array_link and isinstance(external_array_link, dict):
            link_type = external_array_link.get("link_type", None)
            if link_type == 'hdf5_dataset':
                url = external_array_link.get("url", None)
                name = external_array_link.get("name", None)
                if url is not None and name is not None:
                    client = self._get_external_hdf5_client(url)
                    dataset = client[name]
                    assert isinstance(dataset, h5py.Dataset)
                    return dataset[selection]
        if self._compound_dtype is not None:
            # Compound dtype
            # In this case we index into the compound dtype using the name of the field
            # For example, if the dtype is [('x', 'f4'), ('y', 'f4')], then we can do
            # dataset['x'][0] to get the first x value
            assert self._compound_dtype.names is not None
            if isinstance(selection, str):
                # Find the index of this field in the compound dtype
                ind = self._compound_dtype.names.index(selection)
                # Get the dtype of this field
                dt = self._compound_dtype[ind]
                if dt == 'object':
                    dtype = h5py.Reference
                else:
                    dtype = np.dtype(dt)
                # Return a new object that can be sliced further
                # It's important that the return type is Any here, because otherwise we get linter problems
                ret = LindiH5pyDatasetCompoundFieldSelection(
                    dataset=self, ind=ind, dtype=dtype
                )
                return ret
            else:
                raise TypeError(
                    f"Compound dataset {self.name} does not support selection with {selection}"
                )

        # We use zarr's slicing, except in the case of a scalar dataset
        if self.ndim == 0:
            # make sure selection is ()
            if selection != ():
                raise TypeError(f'Cannot slice a scalar dataset with {selection}')
            return zarr_array[0]
        return zarr_array[selection]

    def _get_external_hdf5_client(self, url: str) -> h5py.File:
        if url not in _external_hdf5_clients:
            if url.startswith("http://") or url.startswith("https://"):
                ff = remfile.File(url)
            else:
                ff = open(url, "rb")  # this never gets closed
            _external_hdf5_clients[url] = h5py.File(ff, "r")
        return _external_hdf5_clients[url]


def _resolve_references(x: Any):
    if isinstance(x, dict):
        # x should only be a dict when x represents a converted reference
        if '_REFERENCE' in x:
            return LindiH5pyReference(x['_REFERENCE'])
        else:  # pragma: no cover
            raise Exception(f"Unexpected dict in selection: {x}")
    elif isinstance(x, list):
        # Replace any references in the list with the resolved ref in-place
        for i, v in enumerate(x):
            x[i] = _resolve_references(v)
    elif isinstance(x, np.ndarray):
        if x.dtype == object or x.dtype is None:
            # Replace any references in the object array with the resolved ref in-place
            view_1d = x.reshape(-1)
            for i in range(len(view_1d)):
                view_1d[i] = _resolve_references(view_1d[i])
    return x


class LindiH5pyDatasetCompoundFieldSelection:
    """
    This class is returned when a compound dataset is indexed with a field name.
    For example, if the dataset has dtype [('x', 'f4'), ('y', 'f4')], then we
    can do dataset['x'][0] to get the first x value. The dataset['x'] returns an
    object of this class.
    """
    def __init__(self, *, dataset: LindiH5pyDataset, ind: int, dtype: np.dtype):
        self._dataset = dataset  # The parent dataset
        self._ind = ind  # The index of the field in the compound dtype
        self._dtype = dtype  # The dtype of the field
        if self._dataset.ndim != 1:
            # For now we only support 1D datasets
            raise TypeError(
                f"Compound field selection only implemented for 1D datasets, not {self._dataset.ndim}D"
            )
        if not isinstance(self._dataset._dataset_object, zarr.Array):
            raise TypeError(
                f"Compound field selection only implemented for zarr.Array, not {type(self._dataset._dataset_object)}"
            )
        za = self._dataset._dataset_object
        self._zarr_array = za
        # Prepare the data in memory
        d = [za[i][self._ind] for i in range(len(za))]
        if self._dtype == h5py.Reference:
            # Convert to LindiH5pyReference
            # Every element in the selection should be a reference dict
            d = [LindiH5pyReference(x['_REFERENCE']) for x in d]
        self._data = np.array(d, dtype=self._dtype)

    def __len__(self):
        """We conform to h5py, which is the number of elements in the first dimension. TypeError if scalar"""
        if self.ndim == 0:
            raise TypeError("Scalar dataset")
        return self.shape[0]  # type: ignore

    def __iter__(self):
        """We conform to h5py, which is: Iterate over the first axis. TypeError if scalar."""
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        for i in range(shape[0]):
            yield self[i]

    @property
    def ndim(self):
        return self._zarr_array.ndim

    @property
    def shape(self):
        return self._zarr_array.shape

    @property
    def dtype(self):
        self._dtype

    @property
    def size(self):
        return self._data.size

    def __getitem__(self, selection):
        return self._data[selection]
