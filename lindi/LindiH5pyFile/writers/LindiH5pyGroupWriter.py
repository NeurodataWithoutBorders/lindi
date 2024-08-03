from typing import TYPE_CHECKING
import h5py
import numpy as np
import zarr
import numcodecs
from numcodecs.abc import Codec

from ..LindiH5pyDataset import LindiH5pyDataset
from ..LindiH5pyReference import LindiH5pyReference

if TYPE_CHECKING:
    from ..LindiH5pyGroup import LindiH5pyGroup  # pragma: no cover

from ...conversion.create_zarr_dataset_from_h5_data import create_zarr_dataset_from_h5_data

_compression_not_specified_ = object()


class LindiH5pyGroupWriter:
    def __init__(self, p: 'LindiH5pyGroup'):
        self.p = p

    def create_group(self, name, track_order=None):
        from ..LindiH5pyGroup import LindiH5pyGroup  # avoid circular import
        if track_order is not None:
            raise Exception("track_order is not supported (I don't know what it is)")
        if isinstance(self.p._zarr_group, h5py.Group):
            return LindiH5pyGroup(
                self.p._zarr_group.create_group(name), self.p._file
            )
        elif isinstance(self.p._zarr_group, zarr.Group):
            return LindiH5pyGroup(
                self.p._zarr_group.create_group(name), self.p._file
            )
        else:
            raise Exception(f'Unexpected group object type: {type(self.p._zarr_group)}')

    def require_group(self, name):
        if name in self.p:
            ret = self.p[name]
            if not isinstance(ret, LindiH5pyGroup):
                raise Exception(f'Expected a group at {name} but got {type(ret)}')
            return ret
        return self.create_group(name)

    def create_dataset(
        self,
        name,
        shape=None,
        dtype=None,
        data=None,
        **kwds
    ):
        chunks = None
        compression = _compression_not_specified_
        compression_opts = None
        for k, v in kwds.items():
            if k == 'chunks':
                chunks = v
            elif k == 'compression':
                compression = v
            elif k == 'compression_opts':
                compression_opts = v
            else:
                raise Exception(f'Unsupported kwds in create_dataset: {k}')

        if compression is _compression_not_specified_:
            _zarr_compressor = 'default'
            if compression_opts is not None:
                raise Exception('compression_opts is only supported when compression is provided')
        elif isinstance(compression, Codec):
            _zarr_compressor = compression
            if compression_opts is not None:
                raise Exception('compression_opts is not supported when compression is provided as a Codec')
        elif isinstance(compression, str):
            if compression == 'gzip':
                if compression_opts is None:
                    level = 4  # default for h5py
                elif isinstance(compression_opts, int):
                    level = compression_opts
                else:
                    raise Exception(f'Unexpected type for compression_opts: {type(compression_opts)}')
                _zarr_compressor = numcodecs.GZip(level=level)
            else:
                raise Exception(f'Compression {compression} is not supported')
        elif compression is None:
            _zarr_compressor = None
        else:
            raise Exception(f'Unexpected type for compression: {type(compression)}')

        if isinstance(self.p._zarr_group, h5py.Group):
            if _zarr_compressor != 'default':
                raise Exception('zarr_compressor is not supported when _group_object is h5py.Group')
            return LindiH5pyDataset(
                self._group_object.create_dataset(name, shape=shape, dtype=dtype, data=data, chunks=chunks),  # type: ignore
                self.p._file
            )
        elif isinstance(self.p._zarr_group, zarr.Group):
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
                zarr_parent_group=self.p._zarr_group,
                name=name,
                label=(self.p.name or '') + '/' + name,
                h5_chunks=chunks,
                h5_shape=shape,
                h5_dtype=dtype,
                h5_data=data,
                h5f=None,
                zarr_compressor=_zarr_compressor
            )
            return LindiH5pyDataset(ds, self.p._file)
        else:
            raise Exception(f'Unexpected group object type: {type(self.p._zarr_group)}')

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        if name in self.p:
            ret = self.p[name]
            if not isinstance(ret, LindiH5pyDataset):
                raise Exception(f'Expected a dataset at {name} but got {type(ret)}')
            if ret.shape != shape:
                raise Exception(f'Expected shape {shape} but got {ret.shape}')
            if exact:
                if ret.dtype != dtype:
                    raise Exception(f'Expected dtype {dtype} but got {ret.dtype}')
            else:
                if not np.can_cast(ret.dtype, dtype):
                    raise Exception(f'Cannot cast dtype {ret.dtype} to {dtype}')
            return ret
        return self.create_dataset(name, *(shape, dtype), **kwds)

    def __setitem__(self, name, obj):
        if isinstance(obj, h5py.SoftLink):
            if isinstance(self.p._zarr_group, h5py.Group):
                self.p._zarr_group[name] = obj
            elif isinstance(self.p._zarr_group, zarr.Group):
                grp = self.p.create_group(name)
                grp._zarr_group.attrs['_SOFT_LINK'] = {
                    'path': obj.path
                }
            else:
                raise Exception(f'Unexpected group object type: {type(self.p._zarr_group)}')
        else:
            raise Exception(f'Unexpected type for obj in __setitem__: {type(obj)}')

    def __delitem__(self, name):
        del self.p._zarr_group[name]

    @property
    def ref(self):
        return LindiH5pyReference({
            'object_id': self.p.attrs.get('object_id', None),
            'path': self.p.name,
            'source': '.',
            'source_object_id': self.p.file.attrs.get('object_id', None)
        })
