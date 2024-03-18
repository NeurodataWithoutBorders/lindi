class LindiReference:
    def __init__(self, obj: dict):
        self._object_id = obj["object_id"]
        self._path = obj["path"]
        self._source = obj["source"]
        self._source_object_id = obj["source_object_id"]

    @property
    def name(self):
        return self._path

    def __repr__(self):
        return f"LindiReference({self._source}, {self._path})"

    def __str__(self):
        return f"LindiReference({self._source}, {self._path})"
