from typing import TYPE_CHECKING
import h5py
import numpy as np
import zarr

from ..LindiH5pyDataset import LindiH5pyDataset
from ..LindiH5pyReference import LindiH5pyReference

if TYPE_CHECKING:
    from ..LindiH5pyGroup import LindiH5pyGroup  # pragma: no cover

from ...conversion.create_zarr_dataset_from_h5_data import create_zarr_dataset_from_h5_data


class LindiH5pyGroupWrite:
    def __init__(self, p: 'LindiH5pyGroup'):
        self.p = p

    def create_group(self, name, track_order=None):
        from ..LindiH5pyGroup import LindiH5pyGroup  # avoid circular import
        if track_order is not None:
            raise Exception("track_order is not supported (I don't know what it is)")
        if isinstance(self.p._group_object, h5py.Group):
            return LindiH5pyGroup(
                self.p._group_object.create_group(name), self.p._file
            )
        elif isinstance(self.p._group_object, zarr.Group):
            return LindiH5pyGroup(
                self.p._group_object.create_group(name), self.p._file
            )
        else:
            raise Exception(f'Unexpected group object type: {type(self.p._group_object)}')

    def require_group(self, name):
        if name in self.p:
            ret = self.p[name]
            if not isinstance(ret, LindiH5pyGroup):
                raise Exception(f'Expected a group at {name} but got {type(ret)}')
            return ret
        return self.create_group(name)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        chunks = None
        for k, v in kwds.items():
            if k == 'chunks':
                chunks = v
            else:
                raise Exception(f'Unsupported kwds in create_dataset: {k}')

        if isinstance(self.p._group_object, h5py.Group):
            return LindiH5pyDataset(
                self._group_object.create_dataset(name, shape=shape, dtype=dtype, data=data, chunks=chunks),  # type: ignore
                self.p._file
            )
        elif isinstance(self.p._group_object, zarr.Group):
            if isinstance(data, list):
                data = np.array(data)
            if shape is None:
                if data is None:
                    raise Exception('shape or data must be provided')
                if isinstance(data, np.ndarray):
                    shape = data.shape
                else:
                    shape = ()
            if dtype is None:
                if data is None:
                    raise Exception('dtype or data must be provided')
                if isinstance(data, np.ndarray):
                    dtype = data.dtype
                else:
                    dtype = np.dtype(type(data))
            ds = create_zarr_dataset_from_h5_data(
                zarr_parent_group=self.p._group_object,
                name=name,
                label=(self.p.name or '') + '/' + name,
                h5_chunks=chunks,
                h5_shape=shape,
                h5_dtype=dtype,
                h5_data=data,
                h5f=None
            )
            return LindiH5pyDataset(ds, self.p._file)
        else:
            raise Exception(f'Unexpected group object type: {type(self.p._group_object)}')

    def __setitem__(self, name, obj):
        if isinstance(obj, h5py.SoftLink):
            if isinstance(self.p._group_object, h5py.Group):
                self.p._group_object[name] = obj
            elif isinstance(self.p._group_object, zarr.Group):
                grp = self.p.create_group(name)
                grp._group_object.attrs['_SOFT_LINK'] = {
                    'path': obj.path
                }
            else:
                raise Exception(f'Unexpected group object type: {type(self.p._group_object)}')
        else:
            raise Exception(f'Unexpected type for obj in __setitem__: {type(obj)}')

    def __delitem__(self, name):
        del self.p._group_object[name]

    @property
    def ref(self):
        return LindiH5pyReference({
            'object_id': self.p.attrs.get('object_id', None),
            'path': self.p.name,
            'source': '.',
            'source_object_id': self.p.file.attrs.get('object_id', None)
        })
