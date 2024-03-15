from typing import Dict
import zarr
import h5py
import remfile
from .LindiAttributes import LindiAttributes


class LindiDataset:
    def __init__(self, *, _zarr_array: zarr.Array):
        self._zarr_array = _zarr_array
        self._is_scalar = self._zarr_array.attrs.get("_SCALAR", False)

        self._external_hdf5_clients: Dict[str, h5py.File] = {}

    @property
    def attrs(self):
        """Attributes attached to this object"""
        return LindiAttributes(_object=self._zarr_array)

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

    def __getitem__(self, selection):
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
        # We use zarr's slicing, except in the case of a scalar dataset
        if self.ndim == 0:
            # make sure selection is ()
            if selection != ():
                raise TypeError("Scalar dataset")
            return self._zarr_array[0]
        return self._zarr_array[selection]

    def _get_external_hdf5_client(self, url: str) -> h5py.File:
        if url not in self._external_hdf5_clients:
            remf = remfile.File(url)
            self._external_hdf5_clients[url] = h5py.File(remf, 'r')
        return self._external_hdf5_clients[url]
