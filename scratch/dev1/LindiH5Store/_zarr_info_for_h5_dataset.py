import json
import struct
from typing import Union, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import numcodecs
import h5py
from numcodecs.abc import Codec
from ._h5_filters_to_codecs import _h5_filters_to_codecs


@dataclass
class ZarrInfoForH5Dataset:
    shape: Tuple[int]
    chunks: Union[None, Tuple[int]]
    dtype: Any
    filters: Union[List[Codec], None]
    fill_value: Any
    object_codec: Union[None, Codec]
    inline_data: Union[bytes, None]


def _zarr_info_for_h5_dataset(h5_dataset: h5py.Dataset) -> ZarrInfoForH5Dataset:
    """Get the information needed to create a zarr dataset from an h5py dataset.

    This is the main workhorse function for LindiH5Store. It takes an h5py
    dataset and returns a ZarrInfoForH5Dataset object.

    It handles the following cases:

    For non-scalars, if it is a numeric array, then the data can stay where it
    is in the hdf5 file. The hdf5 filters are translated into zarr filters using
    the _h5_filters_to_codecs function.

    If it is a non-scalar object array, then the inline_data will be a JSON string and the
    JSON codec will be used.

    When the shape is (), we have a scalar dataset. Since zarr doesn't support
    scalar datasets, we make an array of shape (1,). The _ARRAY_DIMENSIONS
    attribute will be set to [] elsewhere to indicate that it is actually a
    scalar. The inline_data attribute will be set. In the case of a numeric
    scalar, it will be a bytes object with the binary representation of the
    value. In the case of an object, the inline_data will be a JSON string and
    the JSON codec will be used.
    """
    shape = h5_dataset.shape
    dtype = h5_dataset.dtype

    if len(shape) == 0:
        # scalar dataset
        value = h5_dataset[()]
        # zarr doesn't support scalar datasets, so we make an array of shape (1,)
        # and the _ARRAY_DIMENSIONS attribute will be set to [] to indicate that
        # it is a scalar dataset

        # Let's handle all the possible types explicitly
        numeric_format_str = _get_numeric_format_str(dtype)
        if numeric_format_str is not None:
            # Handle the simple numeric types
            inline_data = struct.pack(numeric_format_str, value)
            return ZarrInfoForH5Dataset(
                shape=(1,),
                chunks=None,
                dtype=dtype,
                filters=None,
                fill_value=0,
                object_codec=None,
                inline_data=inline_data
            )
        elif dtype == object:
            # For type object, we are going to use the JSON codec
            # which requires inline data of the form [[val], '|O', [1]]
            if isinstance(value, (bytes, str)):
                if isinstance(value, bytes):
                    value = value.decode()
                return ZarrInfoForH5Dataset(
                    shape=(1,),
                    chunks=None,
                    dtype=dtype,
                    filters=None,
                    fill_value=' ',
                    object_codec=numcodecs.JSON(),
                    inline_data=json.dumps([value, '|O', [1]]).encode('utf-8')
                )
            else:
                raise Exception(f'Not yet implemented (1): object scalar dataset with value {value} and dtype {dtype}')
        else:
            raise Exception(f'Cannot handle scalar dataset {h5_dataset.name} with dtype {dtype}')
    else:
        # not a scalar dataset
        if dtype.kind in ['i', 'u', 'f']:  # integer, unsigned integer, float
            # This is the normal case of a chunked dataset with a numeric dtype
            filters = _h5_filters_to_codecs(h5_dataset)
            return ZarrInfoForH5Dataset(
                shape=shape,
                chunks=h5_dataset.chunks,
                dtype=dtype,
                filters=filters,
                fill_value=h5_dataset.fillvalue,
                object_codec=None,
                inline_data=None
            )
        elif dtype.kind == 'O':
            # For type object, we are going to use the JSON codec
            # which requires inline data of the form [list, of, items, ..., '|O', [n1, n2, ...]]
            object_codec = numcodecs.JSON()
            data = h5_dataset[:]
            data_vec_view = data.ravel()
            for i, val in enumerate(data_vec_view):
                if isinstance(val, bytes):
                    data_vec_view[i] = val.decode()
                elif isinstance(val, str):
                    data_vec_view[i] = val
                elif isinstance(val, h5py.h5r.Reference):
                    print(f'Warning: reference in dataset {h5_dataset.name} not handled')
                    data_vec_view[i] = None
                else:
                    raise Exception(f'Cannot handle dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')
            inline_data = json.dumps(data.tolist() + ['|O', list(shape)]).encode('utf-8')
            return ZarrInfoForH5Dataset(
                shape=shape,
                chunks=None,
                dtype=dtype,
                filters=None,
                fill_value=' ',  # not sure what to put here
                object_codec=object_codec,
                inline_data=inline_data
            )
        elif dtype.kind in 'SU':  # byte string or unicode string
            raise Exception(f'Not yet implemented (2): dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')
        else:
            raise Exception(f'Not yet implemented (3): dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')


def _get_numeric_format_str(dtype: Any) -> Union[str, None]:
    """Get the format string for a numeric dtype.

    This is used to convert a scalar dataset to inline data using struct.pack.
    """
    if dtype == np.int8:
        return '<b'
    elif dtype == np.uint8:
        return '<B'
    elif dtype == np.int16:
        return '<h'
    elif dtype == np.uint16:
        return '<H'
    elif dtype == np.int32:
        return '<i'
    elif dtype == np.uint32:
        return '<I'
    elif dtype == np.int64:
        return '<q'
    elif dtype == np.uint64:
        return '<Q'
    elif dtype == np.float32:
        return '<f'
    elif dtype == np.float64:
        return '<d'
    else:
        return None
