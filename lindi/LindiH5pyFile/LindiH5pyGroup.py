from typing import TYPE_CHECKING, Union
import h5py

from lindi.LindiZarrWrapper import LindiZarrWrapperDataset
from .LindiH5pyDataset import LindiH5pyDataset
from .LindiH5pyLink import LindiH5pyHardLink, LindiH5pySoftLink
from ..LindiZarrWrapper import LindiZarrWrapperGroup
from .LindiH5pyAttributes import LindiH5pyAttributes


if TYPE_CHECKING:
    from .LindiH5pyFile import LindiH5pyFile


class LindiH5pyGroupId:
    def __init__(self, _h5py_group_id):
        self._h5py_group_id = _h5py_group_id


class LindiH5pyGroup(h5py.Group):
    def __init__(self, _group_object: Union[h5py.Group, LindiZarrWrapperGroup], _file: "LindiH5pyFile"):
        self._group_object = _group_object
        self._file = _file

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
        elif isinstance(self._group_object, LindiZarrWrapperGroup):
            if isinstance(name, (bytes, str)):
                x = self._group_object[name]
            else:
                raise TypeError(
                    "Accessing a group is done with bytes or str, "
                    "not {}".format(type(name))
                )
            if isinstance(x, LindiZarrWrapperGroup):
                return LindiH5pyGroup(x, self._file)
            elif isinstance(x, LindiZarrWrapperDataset):
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
            elif isinstance(self._group_object, LindiZarrWrapperGroup):
                x = self._group_object.get(name, default=default)
                if isinstance(x, LindiZarrWrapperGroup) and x.soft_link is not None:
                    return LindiH5pySoftLink(x.soft_link['path'])
                else:
                    return LindiH5pyHardLink()
            else:
                raise Exception(f"Unhandled type: {type(self._group_object)}")
        else:
            raise Exception("Impossible")

    @property
    def name(self):
        return self._group_object.name

    def __iter__(self):
        return self._group_object.__iter__()

    def __reversed__(self):
        return self._group_object.__reversed__()

    def __contains__(self, name):
        return self._group_object.__contains__(name)

    @property
    def id(self):
        return LindiH5pyGroupId(self._group_object.id)

    @property
    def file(self):
        return self._file

    @property
    def attrs(self):  # type: ignore
        return LindiH5pyAttributes(self._group_object.attrs)
