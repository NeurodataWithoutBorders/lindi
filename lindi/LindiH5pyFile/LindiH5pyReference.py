import h5py
from ..LindiZarrWrapper import LindiZarrWrapperReference


class LindiH5pyReference(h5py.h5r.Reference):
    def __init__(self, reference: LindiZarrWrapperReference):
        self._reference = reference

    def __repr__(self):
        return f"LindiH5pyReference({self._reference})"

    def __str__(self):
        return f"LindiH5pyReference({self._reference})"
