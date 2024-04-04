import numpy as np


def _is_numeric_dtype(dtype: np.dtype) -> bool:
    """Return True if the dtype is a numeric dtype."""
    return np.issubdtype(dtype, np.number)
