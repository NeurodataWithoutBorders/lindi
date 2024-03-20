from typing import TYPE_CHECKING, Union, Any
import h5py
import numpy as np

from .LindiH5pyAttributes import LindiH5pyAttributes
from .LindiH5pyReference import LindiH5pyReference
from ..LindiZarrWrapper import LindiZarrWrapperDataset
from ..LindiZarrWrapper import LindiZarrWrapperReference


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
        ret = self._dataset_object.__getitem__(args, new_dtype)
        if isinstance(self._dataset_object, LindiZarrWrapperDataset):
            ret = _resolve_references(ret)
        return ret


def _resolve_references(x: Any):
    if isinstance(x, dict):
        if '_REFERENCE' in x:
            return LindiH5pyReference(LindiZarrWrapperReference(x['_REFERENCE']))
        else:
            for k, v in x.items():
                x[k] = _resolve_references(v)
    elif isinstance(x, LindiZarrWrapperReference):
        return LindiH5pyReference(x)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            x[i] = _resolve_references(v)
    elif isinstance(x, np.ndarray):
        if x.dtype == object or x.dtype is None:
            view_1d = x.reshape(-1)
            for i in range(len(view_1d)):
                view_1d[i] = _resolve_references(view_1d[i])
    return x
