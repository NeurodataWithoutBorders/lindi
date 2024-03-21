from typing import List, Union
import json
from .FloatJsonEncoder import FloatJSONEncoder


def _join(a: str, b: str) -> str:
    if a == "":
        return b
    else:
        return f"{a}/{b}"


def _get_chunk_names_for_dataset(chunk_coords_shape: List[int]) -> List[str]:
    """Get the chunk names for a dataset with the given chunk coords shape.

    For example: _get_chunk_names_for_dataset([1, 2, 3]) returns
    ['0.0.0', '0.0.1', '0.0.2', '0.1.0', '0.1.1', '0.1.2']
    """
    ndim = len(chunk_coords_shape)
    if ndim == 0:
        return ["0"]
    elif ndim == 1:
        return [str(i) for i in range(chunk_coords_shape[0])]
    else:
        names0 = _get_chunk_names_for_dataset(chunk_coords_shape[1:])
        names = []
        for i in range(chunk_coords_shape[0]):
            for name0 in names0:
                names.append(f"{i}.{name0}")
        return names


def _reformat_json(x: Union[bytes, None]) -> Union[bytes, None]:
    """Reformat to not include whitespace and to encode NaN, Inf, and -Inf as strings."""
    if x is None:
        return None
    a = json.loads(x.decode("utf-8"))
    return json.dumps(a, cls=FloatJSONEncoder, separators=(",", ":")).encode("utf-8")
