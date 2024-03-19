from typing import Union
import h5py
import zarr

from .LindiH5pyGroup import LindiH5pyGroup
from .LindiH5pyDataset import LindiH5pyDataset
from ..LindiZarrWrapper import LindiZarrWrapper, LindiZarrWrapperGroup, LindiZarrWrapperDataset
from .LindiH5pyAttributes import LindiH5pyAttributes
from .LindiH5pyReference import LindiH5pyReference


class LindiH5pyFile(h5py.File):
    def __init__(self, _file_object: Union[h5py.File, LindiZarrWrapper]):
        """
        Do not use this constructor directly. Instead, use
        from_reference_file_system, from_zarr_group, or from_h5py_file.
        """
        self._file_object = _file_object
        self._the_group = LindiH5pyGroup(_file_object, self)

    @staticmethod
    def from_reference_file_system(rfs: dict):
        """
        Create a LindiH5pyFile from a reference file system.
        """
        x = LindiZarrWrapper.from_reference_file_system(rfs)
        return LindiH5pyFile(x)

    @staticmethod
    def from_zarr_group(zarr_group: zarr.Group):
        """
        Create a LindiH5pyFile from a zarr group.
        """
        x = LindiZarrWrapper.from_zarr_group(zarr_group)
        return LindiH5pyFile(x)

    @staticmethod
    def from_h5py_file(h5py_file: h5py.File):
        """
        Create a LindiH5pyFile from an h5py file.
        """
        return LindiH5pyFile(h5py_file)

    @property
    def attrs(self):  # type: ignore
        return LindiH5pyAttributes(self._file_object.attrs)

    @property
    def filename(self):
        # This is not a string, but this is what h5py seems to do
        return self._file_object.filename

    @property
    def driver(self):
        raise Exception("Getting driver is not allowed")

    @property
    def mode(self):
        if isinstance(self._file_object, h5py.File):
            return self._file_object.mode
        elif isinstance(self._file_object, LindiZarrWrapper):
            # hard-coded to read-only
            return "r"
        else:
            raise Exception(f"Unhandled type: {type(self._file_object)}")

    @property
    def libver(self):
        raise Exception("Getting libver is not allowed")

    @property
    def userblock_size(self):
        raise Exception("Getting userblock_size is not allowed")

    @property
    def meta_block_size(self):
        raise Exception("Getting meta_block_size is not allowed")

    def swmr_mode(self, value):  # type: ignore
        raise Exception("Getting swmr_mode is not allowed")

    def close(self):
        # return self._file_object.close()
        pass

    def flush(self):
        if isinstance(self._file_object, h5py.File):
            return self._file_object.flush()

    def __enter__(self):  # type: ignore
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f'<LindiZarrWrapper "{self._file_object}">'

    # Group methods

    def __getitem__(self, name):
        if isinstance(name, LindiH5pyReference):
            assert isinstance(self._file_object, LindiZarrWrapper)
            x = self._file_object[name._reference]
            if isinstance(x, LindiZarrWrapperGroup):
                return LindiH5pyGroup(x, self)
            elif isinstance(x, LindiZarrWrapperDataset):
                return LindiH5pyDataset(x, self)
            else:
                raise Exception(f"Unexpected type for resolved reference at path {name}: {type(x)}")
        elif isinstance(name, h5py.Reference):
            assert isinstance(self._file_object, h5py.File)
            x = self._file_object[name]
            if isinstance(x, h5py.Group):
                return LindiH5pyGroup(x, self)
            elif isinstance(x, h5py.Dataset):
                return LindiH5pyDataset(x, self)
            else:
                raise Exception(f"Unexpected type for resolved reference at path {name}: {type(x)}")
        return self._the_group[name]

    def get(self, name, default=None, getclass=False, getlink=False):
        return self._the_group.get(name, default, getclass, getlink)

    def __iter__(self):
        return self._the_group.__iter__()

    def __reversed__(self):
        return self._the_group.__reversed__()

    def __contains__(self, name):
        return self._the_group.__contains__(name)

    @property
    def id(self):
        return self._the_group.id

    @property
    def file(self):
        return self._the_group.file

    @property
    def name(self):
        return self._the_group.name
