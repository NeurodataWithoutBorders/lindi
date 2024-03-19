from typing import Dict, Any
import numpy as np
import zarr
import h5py
import remfile
from .LindiZarrWrapperAttributes import LindiZarrWrapperAttributes
from .LindiZarrWrapperReference import LindiZarrWrapperReference


class LindiZarrWrapperDataset:
    def __init__(self, *, _zarr_array: zarr.Array, _client):
        self._zarr_array = _zarr_array
        self._is_scalar = self._zarr_array.attrs.get("_SCALAR", False)
        self._client = _client

        # See if we have the _COMPOUND_DTYPE attribute, which signifies that
        # this is a compound dtype
        compound_dtype_obj = self._zarr_array.attrs.get("_COMPOUND_DTYPE", None)
        if compound_dtype_obj is not None:
            # If we have a compound dtype, then create the numpy dtype
            self._compound_dtype = np.dtype(
                [(compound_dtype_obj[i][0], compound_dtype_obj[i][1]) for i in range(len(compound_dtype_obj))]
            )
        else:
            self._compound_dtype = None

        self._external_hdf5_clients: Dict[str, h5py.File] = {}

    @property
    def file(self):
        return self._client

    @property
    def id(self):
        return None

    @property
    def name(self):
        return self._zarr_array.name

    @property
    def attrs(self):
        """Attributes attached to this object"""
        return LindiZarrWrapperAttributes(_object=self._zarr_array)

    @property
    def ndim(self):
        if self._is_scalar:
            return 0
        return self._zarr_array.ndim

    @property
    def shape(self):
        if self._is_scalar:
            return ()
        return self._zarr_array.shape

    @property
    def dtype(self):
        if self._compound_dtype is not None:
            return self._compound_dtype
        return self._zarr_array.dtype

    @property
    def size(self):
        if self._is_scalar:
            return 1
        return self._zarr_array.size

    @property
    def nbytes(self):
        return self._zarr_array.nbytes

    def __len__(self):
        """We conform to h5py, which is the number of elements in the first dimension, TypeError if scalar"""
        if self.ndim == 0:
            raise TypeError("Scalar dataset")
        return self.shape[0]  # type: ignore

    def __iter__(self):
        """We conform to h5py, which is: Iterate over the first axis.  TypeError if scalar."""
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        for i in range(shape[0]):
            yield self[i]

    def __getitem__(self, selection, new_dtype=None):
        if new_dtype is not None:
            raise TypeError("new_dtype not supported in LindiZarrWrapperDataset.__getitem__")
        # First check whether this is an external array link
        external_array_link = self._zarr_array.attrs.get("_EXTERNAL_ARRAY_LINK", None)
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
                ret: Any = LindiZarrWrapperDatasetCompoundFieldSelection(
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
            return self._zarr_array[0]
        return self._zarr_array[selection]

    def _get_external_hdf5_client(self, url: str) -> h5py.File:
        if url not in self._external_hdf5_clients:
            remf = remfile.File(url)
            self._external_hdf5_clients[url] = h5py.File(remf, "r")
        return self._external_hdf5_clients[url]


class LindiZarrWrapperDatasetCompoundFieldSelection:
    """
    This class is returned when a compound dataset is indexed with a field name.
    For example, if the dataset has dtype [('x', 'f4'), ('y', 'f4')], then we
    can do dataset['x'][0] to get the first x value. The dataset['x'] returns an
    object of this class.
    """
    def __init__(self, *, dataset: LindiZarrWrapperDataset, ind: int, dtype: np.dtype):
        self._dataset = dataset  # The parent dataset
        self._ind = ind  # The index of the field in the compound dtype
        self._dtype = dtype  # The dtype of the field
        if self._dataset.ndim != 1:
            # For now we only support 1D datasets
            raise TypeError(
                f"Compound field selection only implemented for 1D datasets, not {self._dataset.ndim}D"
            )
        # Prepare the data in memory
        za = self._dataset._zarr_array
        d = [za[i][self._ind] for i in range(len(za))]
        if self._dtype == h5py.Reference:
            # Convert to LindiZarrWrapperReference
            d = [LindiZarrWrapperReference(x['_REFERENCE']) for x in d]
        self._data = np.array(d, dtype=self._dtype)

    def __len__(self):
        return self._dataset._zarr_array.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def ndim(self):
        return self._dataset._zarr_array.ndim

    @property
    def shape(self):
        return self._dataset._zarr_array.shape

    @property
    def dtype(self):
        self._dtype

    @property
    def size(self):
        return self._data.size

    def __getitem__(self, selection):
        return self._data[selection]
