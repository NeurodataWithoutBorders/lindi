from typing import Union
import json


def reformat_json(x: Union[bytes, None]) -> Union[bytes, None]:
    """Reformat to not include whitespace and to not allow nan, inf, and ninf.

    It is assumed that float attributes nan, inf, and ninf float values have
    been encoded as strings. See encode_nan_inf_ninf() and h5_to_zarr_attr().
    """
    if x is None:
        return None
    a = json.loads(x.decode("utf-8"))
    return json.dumps(a, separators=(",", ":"), allow_nan=False).encode("utf-8")
