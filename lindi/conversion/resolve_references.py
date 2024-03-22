from typing import Any
import numpy as np


def resolve_references(x: Any):
    from ..LindiH5pyFile.LindiH5pyReference import LindiH5pyReference  # Avoid circular import
    if isinstance(x, dict):
        # x should only be a dict when x represents a converted reference
        if '_REFERENCE' in x:
            return LindiH5pyReference(x['_REFERENCE'])
        else:  # pragma: no cover
            raise Exception(f"Unexpected dict in selection: {x}")
    elif isinstance(x, list):
        # Replace any references in the list with the resolved ref in-place
        for i, v in enumerate(x):
            x[i] = resolve_references(v)
    elif isinstance(x, np.ndarray):
        if x.dtype == object or x.dtype is None:
            # Replace any references in the object array with the resolved ref in-place
            view_1d = x.reshape(-1)
            for i in range(len(view_1d)):
                view_1d[i] = resolve_references(view_1d[i])
    return x
