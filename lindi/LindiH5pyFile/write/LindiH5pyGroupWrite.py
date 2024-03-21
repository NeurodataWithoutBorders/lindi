from typing import TYPE_CHECKING
import numpy as np
import h5py
import zarr
import numcodecs

from ..LindiH5pyDataset import LindiH5pyDataset
from ..LindiH5pyReference import LindiH5pyReference

if TYPE_CHECKING:
    from ..LindiH5pyGroup import LindiH5pyGroup  # pragma: no cover


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
            if isinstance(data, str):
                data = data.encode('utf-8')
            scalar = False
            object_codec = None
            if data is None:
                if dtype == 'object':
                    object_codec = numcodecs.JSON()
                elif dtype in (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64, np.uint16):
                    if shape is None:
                        raise Exception('shape must be provided if data is None')
                else:
                    raise Exception(f'Unexpected dtype in create_dataset for data of type None: {dtype}')
            else:
                if isinstance(data, bytes):
                    if shape == ():
                        shape = None
                    if shape is not None:
                        raise Exception(f'Unexpected shape in create_dataset for data of type bytes: {shape}')
                    if dtype != 'object':
                        raise Exception(f'Unexpected dtype in create_dataset for data of type bytes: {dtype}')
                    shape = (1,)
                    scalar = True
                    data = [data.decode('utf-8')]
                    object_codec = numcodecs.JSON()
                elif dtype in (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64, np.uint16):
                    if isinstance(data, np.ndarray):
                        if data.dtype != dtype:
                            raise Exception(f'Unexpected dtype in create_dataset for data of type {type(data)}: {data.dtype} != {dtype}')
                        if shape is None:
                            shape = data.shape
                        if shape != data.shape:
                            raise Exception(f'Unexpected shape in create_dataset for data of type {type(data)}: {shape} != {data.shape}')
                    elif isinstance(data, dtype):  # type: ignore
                        # scalar
                        if shape is None:
                            shape = ()
                        if shape != ():
                            raise Exception(f'Unexpected shape in create_dataset for data of type {type(data)}: {shape} != ()')
                        scalar = True
                        data = [data]
                        shape = (1,)
                    else:
                        raise Exception(f'Unexpected data type in create_dataset: {type(data)}')
            ds = self.p._group_object.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                data=data,
                object_codec=object_codec,
                chunks=chunks
            )
            if scalar:
                ds.attrs['_SCALAR'] = True
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

    @property
    def ref(self):
        return LindiH5pyReference({
            'object_id': self.p.attrs.get('object_id', None),
            'path': self.p.name,
            'source': '.',
            'source_object_id': self.p.file.attrs.get('object_id', None)
        })
