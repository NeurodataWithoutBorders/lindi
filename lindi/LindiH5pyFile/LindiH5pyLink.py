class LindiH5pyHardLink:
    def __init__(self):
        pass


class LindiH5pySoftLink:
    def __init__(self, path: str):
        self._path = path

    @property
    def path(self):
        return self._path
