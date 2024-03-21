import json
import numpy as np


# From https://github.com/rly/h5tojson/blob/b162ff7f61160a48f1dc0026acb09adafdb422fa/h5tojson/h5tojson.py#L121-L156
class FloatJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN, Inf, and -Inf to strings."""

    def encode(self, obj, *args, **kwargs):  # type: ignore
        """Convert NaN, Inf, and -Inf to strings."""
        obj = FloatJSONEncoder._convert_nan(obj)
        return super().encode(obj, *args, **kwargs)

    def iterencode(self, obj, *args, **kwargs):  # type: ignore
        """Convert NaN, Inf, and -Inf to strings."""
        obj = FloatJSONEncoder._convert_nan(obj)
        return super().iterencode(obj, *args, **kwargs)

    @staticmethod
    def _convert_nan(obj):
        """Convert NaN, Inf, and -Inf from a JSON object to strings."""
        if isinstance(obj, dict):
            return {k: FloatJSONEncoder._convert_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FloatJSONEncoder._convert_nan(v) for v in obj]
        elif isinstance(obj, float):
            return FloatJSONEncoder._nan_to_string(obj)
        return obj

    @staticmethod
    def _nan_to_string(obj: float):
        """Convert NaN, Inf, and -Inf from a float to a string."""
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            if obj > 0:
                return "Infinity"
            else:
                return "-Infinity"
        else:
            return float(obj)
