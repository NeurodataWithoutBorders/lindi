from typing import TYPE_CHECKING, Union
import h5py
from .LindiH5pyAttributes import LindiH5pyAttributes
from ..LindiZarrWrapper import LindiZarrWrapperDataset


if TYPE_CHECKING:
    from .LindiH5pyFile import LindiH5pyFile


class LindiH5pyDatasetId:
    def __init__(self, _h5py_dataset_id):
        self._h5py_dataset_id = _h5py_dataset_id


class LindiH5pyDataset(h5py.Dataset):
    def __init__(self, _dataset_object: Union[h5py.Dataset, LindiZarrWrapperDataset], _file: "LindiH5pyFile"):
        self._dataset_object = _dataset_object
        self._file = _file

    @property
    def id(self):
        return LindiH5pyDatasetId(self._dataset_object.id)

    @property
    def shape(self):  # type: ignore
        return self._dataset_object.shape

    @property
    def size(self):
        return self._dataset_object.size

    @property
    def dtype(self):
        return self._dataset_object.dtype

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
        return self._dataset_object.ndim

    @property
    def attrs(self):  # type: ignore
        return LindiH5pyAttributes(self._dataset_object.attrs)

    def __getitem__(self, args, new_dtype=None):
        return self._dataset_object.__getitem__(args, new_dtype)
