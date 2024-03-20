import h5py


class LindiH5pyHardLink(h5py.HardLink):
    def __init__(self):
        pass


class LindiH5pySoftLink(h5py.SoftLink):
    def __init__(self, path: str):
        self._path = path

    @property
    def path(self):
        return self._path
