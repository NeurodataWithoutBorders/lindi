from typing import Union, Any
from .LindiH5pyReference import LindiH5pyReference
from ..LindiZarrWrapper import LindiZarrWrapperAttributes, LindiZarrWrapperReference


class LindiH5pyAttributes:
    def __init__(self, attrs: Union[Any, LindiZarrWrapperAttributes]):
        self._attrs = attrs

    def __getitem__(self, key):
        val = self._attrs[key]
        if isinstance(val, LindiZarrWrapperReference):
            return LindiH5pyReference(val)
        return val

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        for k in self:
            yield k, self[k]

    def __iter__(self):
        for k in self._attrs:
            yield k
