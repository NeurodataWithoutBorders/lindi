from typing import Any, TYPE_CHECKING
import h5py
import zarr
from ..LindiH5pyReference import LindiH5pyReference

if TYPE_CHECKING:
    from ..LindiH5pyDataset import LindiH5pyDataset  # pragma: no cover


class LindiH5pyDatasetWrite:
    def __init__(self, p: 'LindiH5pyDataset'):
        self.p = p

    def __setitem__(self, args, val):
        if isinstance(self.p._dataset_object, h5py.Dataset):
            self.p._dataset_object.__setitem__(args, val)
        elif isinstance(self.p._dataset_object, zarr.Array):
            self._set_item_for_zarr(self.p._dataset_object, args, val)
        else:
            raise Exception(f"Unexpected type: {type(self.p._dataset_object)}")

    def _set_item_for_zarr(self, zarr_array: zarr.Array, selection: Any, val: Any):
        if self.p._compound_dtype is not None:
            raise Exception("Setting compound dataset is not implemented")
        if self.p.ndim == 0:
            if selection != ():
                raise TypeError(f'Cannot slice a scalar dataset with {selection}')
            zarr_array[0] = val
        else:
            dtype = zarr_array.dtype
            if dtype.kind in ['i', 'u', 'f', 'b']:
                # this is the usual numeric case
                zarr_array[selection] = val
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
