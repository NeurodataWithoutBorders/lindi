from typing import Any, TYPE_CHECKING
import h5py
import zarr
import numpy as np

from ..LindiH5pyReference import LindiH5pyReference
from ...conversion._util import _is_numeric_dtype
from ...conversion.create_zarr_dataset_from_h5_data import h5_object_data_to_zarr_data

if TYPE_CHECKING:
    from ..LindiH5pyDataset import LindiH5pyDataset  # pragma: no cover


class LindiH5pyDatasetWriter:
    def __init__(self, p: 'LindiH5pyDataset'):
        self.p = p

    def __setitem__(self, args, val):
        if isinstance(self.p._zarr_array, h5py.Dataset):
            self.p._zarr_array.__setitem__(args, val)
        elif isinstance(self.p._zarr_array, zarr.Array):
            self._set_item_for_zarr(self.p._zarr_array, args, val)
        else:
            raise Exception(f"Unexpected type: {type(self.p._zarr_array)}")

    def _set_item_for_zarr(self, zarr_array: zarr.Array, selection: Any, val: Any):
        if self.p._compound_dtype is not None:
            raise Exception("Setting compound dataset is not implemented")
        if self.p.ndim == 0:
            if selection != ():
                raise TypeError(f'Cannot slice a scalar dataset with {selection}')
            zarr_array[0] = val
        else:
            dtype = zarr_array.dtype
            if _is_numeric_dtype(dtype) or dtype in [bool, np.bool_]:
                # this is the usual numeric case
                zarr_array[selection] = val
            elif dtype.kind == 'O':
                zarr_array[selection] = h5_object_data_to_zarr_data(val, h5f=None, label='')
            else:
                raise Exception(f'Unsupported dtype for slice setting {dtype} in {self.p.name}')

    @property
    def ref(self):
        return LindiH5pyReference({
            'object_id': self.p.attrs.get('object_id', None),
            'path': self.p.name,
            'source': '.',
            'source_object_id': self.p.file.attrs.get('object_id', None)
        })
