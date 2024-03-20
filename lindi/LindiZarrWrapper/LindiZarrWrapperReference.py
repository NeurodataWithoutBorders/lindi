import h5py


# We need h5py.Reference as a base class so that type checking will be okay for
# arrays of compound types that contain references
class LindiZarrWrapperReference(h5py.Reference):
    def __init__(self, obj: dict):
        self._object_id = obj["object_id"]
        self._path = obj["path"]
        self._source = obj["source"]
        self._source_object_id = obj["source_object_id"]

    @property
    def name(self):
        return self._path

    def __repr__(self):
        return f"LindiZarrWrapperReference({self._source}, {self._path})"

    def __str__(self):
        return f"LindiZarrWrapperReference({self._source}, {self._path})"
