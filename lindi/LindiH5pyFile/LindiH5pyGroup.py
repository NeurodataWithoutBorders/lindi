from typing import TYPE_CHECKING, Union
import h5py
import zarr

from .LindiH5pyDataset import LindiH5pyDataset
from .LindiH5pyLink import LindiH5pyHardLink, LindiH5pySoftLink
from .LindiH5pyAttributes import LindiH5pyAttributes


if TYPE_CHECKING:
    from .LindiH5pyFile import LindiH5pyFile  # pragma: no cover


class LindiH5pyGroupId:
    def __init__(self, _h5py_group_id):
        self._h5py_group_id = _h5py_group_id


class LindiH5pyGroup(h5py.Group):
    def __init__(self, _group_object: Union[h5py.Group, zarr.Group], _file: "LindiH5pyFile"):
        self._group_object = _group_object
        self._file = _file

        from .write.LindiH5pyGroupWrite import LindiH5pyGroupWrite  # avoid circular import
        self._write = LindiH5pyGroupWrite(self)

    def __getitem__(self, name):
        if isinstance(self._group_object, h5py.Group):
            if isinstance(name, (bytes, str)):
                x = self._group_object[name]
            else:
                raise TypeError(
                    "Accessing a group is done with bytes or str, "
                    "not {}".format(type(name))
                )
            if isinstance(x, h5py.Group):
                return LindiH5pyGroup(x, self._file)
            elif isinstance(x, h5py.Dataset):
                return LindiH5pyDataset(x, self._file)
            else:
                raise Exception(f"Unknown type: {type(x)}")
        elif isinstance(self._group_object, zarr.Group):
            if isinstance(name, (bytes, str)):
                x = self._group_object[name]
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
                    if not isinstance(target_item, LindiH5pyGroup) and not isinstance(target_item, LindiH5pyDataset):
                        raise Exception(
                            f"Expected a group or dataset at {link_path} but got {type(target_item)}"
                        )
                    return target_item
                return LindiH5pyGroup(x, self._file)
            elif isinstance(x, zarr.Array):
                return LindiH5pyDataset(x, self._file)
            else:
                raise Exception(f"Unknown type: {type(x)}")
        else:
            raise Exception(f"Unhandled type: {type(self._group_object)}")

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
            if isinstance(self._group_object, h5py.Group):
                x = self._group_object.get(name, default=default, getlink=True)
                if isinstance(x, h5py.HardLink):
                    return LindiH5pyHardLink()
                elif isinstance(x, h5py.SoftLink):
                    return LindiH5pySoftLink(x.path)
                else:
                    raise Exception(
                        f"Unhandled type for get with getlink at {self.name} {name}: {type(x)}"
                    )
            elif isinstance(self._group_object, zarr.Group):
                x = self._group_object.get(name, default=default)
                if x is None:
                    return default
                soft_link = x.attrs.get('_SOFT_LINK', None)
                if isinstance(x, zarr.Group) and soft_link is not None:
                    return LindiH5pySoftLink(soft_link['path'])
                else:
                    return LindiH5pyHardLink()
            else:
                raise Exception(f"Unhandled type: {type(self._group_object)}")
        else:
            raise Exception("Impossible")

    @property
    def name(self):
        return self._group_object.name

    def keys(self):  # type: ignore
        return self._group_object.keys()

    def __iter__(self):
        return self._group_object.__iter__()

    def __reversed__(self):
        raise Exception("Not implemented: __reversed__")

    def __contains__(self, name):
        return self._group_object.__contains__(name)

    def __str__(self):
        return f'<{self.__class__.__name__}: {self.name}>'

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.name}>'

    @property
    def id(self):
        if isinstance(self._group_object, h5py.Group):
            return LindiH5pyGroupId(self._group_object.id)
        elif isinstance(self._group_object, zarr.Group):
            return LindiH5pyGroupId(None)
        else:
            raise Exception(f'Unexpected group object type: {type(self._group_object)}')

    @property
    def file(self):
        return self._file

    @property
    def attrs(self):  # type: ignore
        if isinstance(self._group_object, h5py.Group):
            attrs_type = 'h5py'
        elif isinstance(self._group_object, zarr.Group):
            attrs_type = 'zarr'
        else:
            raise Exception(f'Unexpected group object type: {type(self._group_object)}')
        return LindiH5pyAttributes(self._group_object.attrs, attrs_type=attrs_type, readonly=self._file.mode == 'r')

    ##############################
    # write
    def create_group(self, name, track_order=None):
        return self._write.create_group(name, track_order=track_order)

    def require_group(self, name):
        return self._write.require_group(name)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        return self._write.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)

    def __setitem__(self, name, obj):
        return self._write.__setitem__(name, obj)

    def __delitem__(self, name):
        return self._write.__delitem__(name)

    @property
    def ref(self):
        return self._write.ref
