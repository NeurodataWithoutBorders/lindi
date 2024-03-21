import h5py


class LindiH5pyReference(h5py.h5r.Reference):
    def __init__(self, obj: dict):
        self._obj = obj
        self._object_id = obj["object_id"]
        self._path = obj["path"]
        self._source = obj["source"]
        self._source_object_id = obj["source_object_id"]

    def __repr__(self):
        return f"LindiH5pyReference({self._object_id}, {self._path})"

    def __str__(self):
        return f"LindiH5pyReference({self._object_id}, {self._path})"
