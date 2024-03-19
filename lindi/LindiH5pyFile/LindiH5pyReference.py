import h5py
from ..LindiZarrWrapper import LindiZarrWrapperReference


class LindiH5pyReference(h5py.h5r.Reference):
    def __init__(self, reference: LindiZarrWrapperReference):
        self._reference = reference
