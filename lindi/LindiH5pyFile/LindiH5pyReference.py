import h5py


class LindiH5pyReference(h5py.h5r.Reference):
    def __init__(self, obj: dict):
        self._object_id = obj["object_id"]
        self._path = obj["path"]
        self._source = obj["source"]
        self._source_object_id = obj["source_object_id"]

    def __repr__(self):
        return f"LindiH5pyReference({self._object_id}, {self._path})"

    def __str__(self):
        return f"LindiH5pyReference({self._object_id}, {self._path})"


def test_coverage():
    obj = {
        "object_id": "object_id",
        "path": "path",
        "source": "source",
        "source_object_id": "source_object_id",
    }
    ref = LindiH5pyReference(obj)
    assert repr(ref) == "LindiH5pyReference(object_id, path)"
    assert str(ref) == "LindiH5pyReference(object_id, path)"
    assert ref._object_id == "object_id"
    assert ref._path == "path"
    assert ref._source == "source"
    assert ref._source_object_id == "source_object_id"
    assert ref.__class__.__name__ == "LindiH5pyReference"
    assert isinstance(ref, h5py.h5r.Reference)
    assert isinstance(ref, LindiH5pyReference)
