from typing import TYPE_CHECKING
import h5py
import zarr

from .LindiH5pyDataset import LindiH5pyDataset
from .LindiH5pyLink import LindiH5pyHardLink, LindiH5pySoftLink
from .LindiH5pyAttributes import LindiH5pyAttributes


if TYPE_CHECKING:
    from .LindiH5pyFile import LindiH5pyFile  # pragma: no cover


class LindiH5pyGroup(h5py.Group):
    def __init__(self, _zarr_group: zarr.Group, _file: "LindiH5pyFile"):
        self._zarr_group = _zarr_group
        self._file = _file
        self._readonly = _file.mode not in ['r+']

        # In h5py, the id property is an object that exposes low-level
        # operations specific to the HDF5 library. LINDI aims to override the
        # high-level methods such that the low-level operations on id are not
        # needed. However, sometimes packages (e.g., pynwb) use the id as a
        # unique identifier for purposes of caching. Therefore, we make the id
        # to be a string that is unique for each object. If any of the low-level
        # operations are attempted on this id string, then an exception will be
        # raised, which will usually indicate that one of the high-level methods
        # should be overridden.
        self._id = f'{id(self._file)}/{self._zarr_group.name}'

        # The self._write object handles all the writing operations
        from .writers.LindiH5pyGroupWriter import LindiH5pyGroupWriter  # avoid circular import
        if self._readonly:
            self._writer = None
        else:
            self._writer = LindiH5pyGroupWriter(self)

    def __getitem__(self, name):
        if isinstance(name, (bytes, str)):
            x = self._zarr_group[name]
        else:
            raise TypeError(
                "Accessing a group is done with bytes or str, "
                "not {}".format(type(name))
            )
        if isinstance(x, zarr.Group):
            # follow the link if this is a soft link
            soft_link = x.attrs.get('_SOFT_LINK', None)
            if soft_link is not None:
                link_path = soft_link['path']
                target_item = self._file.get(link_path)
                if not isinstance(target_item, (LindiH5pyGroup, LindiH5pyDataset)):
                    raise Exception(
                        f"Expected a group or dataset at {link_path} but got {type(target_item)}"
                    )
                return target_item
            return LindiH5pyGroup(x, self._file)
        elif isinstance(x, zarr.Array):
            return LindiH5pyDataset(x, self._file)
        else:
            raise Exception(f"Unknown type: {type(x)}")

    def get(self, name, default=None, getclass=False, getlink=False):
        if not (getclass or getlink):
            try:
                return self[name]
            except KeyError:
                return default

        if name not in self:
            return default
        elif getclass and not getlink:
            raise Exception("Getting class is not allowed")
        elif getlink and not getclass:
            x = self._zarr_group.get(name, default=default)
            if x is None:
                return default
            soft_link = x.attrs.get('_SOFT_LINK', None)
            if isinstance(x, zarr.Group) and soft_link is not None:
                return LindiH5pySoftLink(soft_link['path'])
            else:
                return LindiH5pyHardLink()
        else:
            raise Exception("Impossible")

    @property
    def name(self):
        return self._zarr_group.name

    def keys(self):  # type: ignore
        return self._zarr_group.keys()

    def __iter__(self):
        return self._zarr_group.__iter__()

    def __reversed__(self):
        raise Exception("Not implemented: __reversed__")

    def __contains__(self, name):
        return self._zarr_group.__contains__(name)

    def __str__(self):
        return f'<{self.__class__.__name__}: {self.name}>'

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.name}>'

    @property
    def id(self):
        # see comment above
        return self._id

    @property
    def file(self):
        return self._file

    @property
    def attrs(self):  # type: ignore
        return LindiH5pyAttributes(self._zarr_group.attrs, readonly=self._file.mode == 'r')

    @property
    def ref(self):
        if self._readonly:
            raise ValueError("Cannot get ref on read-only object")
        assert self._writer is not None
        return self._writer.ref

    ##############################
    # write
    def create_group(self, name, track_order=None):
        if self._readonly:
            raise Exception('Cannot create group in read-only mode')
        assert self._writer is not None
        return self._writer.create_group(name, track_order=track_order)

    def require_group(self, name):
        if self._readonly:
            raise Exception('Cannot require group in read-only mode')
        assert self._writer is not None
        return self._writer.require_group(name)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if self._readonly:
            raise Exception('Cannot create dataset in read-only mode')
        assert self._writer is not None
        return self._writer.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        if self._readonly:
            raise Exception('Cannot require dataset in read-only mode')
        assert self._writer is not None
        return self._writer.require_dataset(name, shape, dtype, exact=exact, **kwds)

    def __setitem__(self, name, obj):
        if self._readonly:
            raise Exception('Cannot set item in read-only mode')
        assert self._writer is not None
        return self._writer.__setitem__(name, obj)

    def __delitem__(self, name):
        if self._readonly:
            raise Exception('Cannot delete item in read-only mode')
        assert self._writer is not None
        return self._writer.__delitem__(name)
