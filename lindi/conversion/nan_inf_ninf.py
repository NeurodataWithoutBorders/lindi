import numpy as np


def decode_nan_inf_ninf(val):
    if isinstance(val, list):
        return [decode_nan_inf_ninf(v) for v in val]
    elif isinstance(val, dict):
        return {k: decode_nan_inf_ninf(v) for k, v in val.items()}
    elif val == 'NaN':
        return float('nan')
    elif val == 'Infinity':
        return float('inf')
    elif val == '-Infinity':
        return float('-inf')
    else:
        return val


def encode_nan_inf_ninf(val):
    if isinstance(val, list):
        return [encode_nan_inf_ninf(v) for v in val]
    elif isinstance(val, dict):
        return {k: encode_nan_inf_ninf(v) for k, v in val.items()}
    elif type(val) in [float, np.float16, np.float32, np.float64, np.float128]:
        if np.isnan(val):
            return 'NaN'
        elif val == float('inf'):
            return 'Infinity'
        elif val == float('-inf'):
            return '-Infinity'
        else:
            return val
    else:
        return val
