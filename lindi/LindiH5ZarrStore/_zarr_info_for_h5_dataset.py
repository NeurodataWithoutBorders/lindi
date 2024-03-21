import json
import struct
from typing import Union, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import numcodecs
import h5py
from numcodecs.abc import Codec
from ._h5_attr_to_zarr_attr import _h5_ref_to_zarr_attr
from ._h5_filters_to_codecs import _h5_filters_to_codecs


@dataclass
class ZarrInfoForH5Dataset:
    shape: Tuple[int]
    chunks: Tuple[int]
    dtype: Any
    filters: Union[List[Codec], None]
    fill_value: Any
    object_codec: Union[None, Codec]
    inline_data: Union[bytes, None]


def _zarr_info_for_h5_dataset(h5_dataset: h5py.Dataset) -> ZarrInfoForH5Dataset:
    """Get the information needed to create a zarr dataset from an h5py dataset.

    This is the main workhorse function for LindiH5ZarrStore. It takes an h5py
    dataset and returns a ZarrInfoForH5Dataset object.

    It handles the following cases:

    For non-scalars, if it is a numeric array, then the data can stay where it
    is in the hdf5 file. The hdf5 filters are translated into zarr filters using
    the _h5_filters_to_codecs function.

    If it is a non-scalar object array, then the inline_data will be a JSON
    string and the JSON codec will be used.

    When the shape is (), we have a scalar dataset. Since zarr doesn't support
    scalar datasets, we make an array of shape (1,). The _SCALAR attribute will
    be set to True elsewhere to indicate that it is actually a scalar. The
    inline_data attribute will be set. In the case of a numeric scalar, it will
    be a bytes object with the binary representation of the value. In the case
    of an object, the inline_data will be a JSON string and the JSON codec will
    be used.
    """
    shape = h5_dataset.shape
    dtype = h5_dataset.dtype

    if len(shape) == 0:
        # scalar dataset
        value = h5_dataset[()]
        # zarr doesn't support scalar datasets, so we make an array of shape (1,)
        # and the _SCALAR attribute will be set to True elsewhere to indicate that
        # it is a scalar dataset

        # Let's handle all the possible types explicitly
        numeric_format_str = _get_numeric_format_str(dtype)
        if numeric_format_str is not None:
            # Handle the simple numeric types
            inline_data = struct.pack(numeric_format_str, value)
            return ZarrInfoForH5Dataset(
                shape=(1,),
                chunks=(1,),  # be explicit about chunks
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
                    chunks=(1,),  # be explicit about chunks
                    dtype=dtype,
                    filters=None,
                    fill_value=' ',
                    object_codec=numcodecs.JSON(),
                    inline_data=json.dumps([value, '|O', [1]], separators=(',', ':')).encode('utf-8')
                )
            else:
                raise Exception(f'Not yet implemented (1): object scalar dataset with value {value} and dtype {dtype}')
        else:
            raise Exception(f'Cannot handle scalar dataset {h5_dataset.name} with dtype {dtype}')
    else:
        # not a scalar dataset
        if dtype.kind in ['i', 'u', 'f', 'b']:  # integer, unsigned integer, float, boolean
            # This is the normal case of a chunked dataset with a numeric (or boolean) dtype
            filters = _h5_filters_to_codecs(h5_dataset)
            chunks = h5_dataset.chunks
            if chunks is None:
                # If the dataset is not chunked, we use the entire dataset as a single chunk
                # It's important to be explicit about the chunks, because I think None means that zarr could decide otherwise
                chunks = shape
            return ZarrInfoForH5Dataset(
                shape=shape,
                chunks=chunks,
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
                elif isinstance(val, h5py.Reference):
                    data_vec_view[i] = _h5_ref_to_zarr_attr(val, label=f'{h5_dataset.name}[{i}]', h5f=h5_dataset.file)
                else:
                    raise Exception(f'Cannot handle dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')
            inline_data = json.dumps(data.tolist() + ['|O', list(shape)], separators=(',', ':')).encode('utf-8')
            return ZarrInfoForH5Dataset(
                shape=shape,
                chunks=shape,  # be explicit about chunks
                dtype=dtype,
                filters=None,
                fill_value=' ',  # not sure what to put here
                object_codec=object_codec,
                inline_data=inline_data
            )
        elif dtype.kind in 'SU':  # byte string or unicode string
            raise Exception(f'Not yet implemented (2): dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')
        elif dtype.kind == 'V':  # void (i.e. compound)
            if h5_dataset.ndim == 1:
                # for now we only handle the case of a 1D compound dataset
                data = h5_dataset[:]
                # Create an array that would be for example like this
                # dtype = np.dtype([('x', np.float64), ('y', np.int32), ('weight', np.float64)])
                # array_list = [[3, 4, 5.3], [2, 1, 7.1], ...]
                # where the first entry corresponds to x in the example above, the second to y, and the third to weight
                # This is a more compact representation than [{'x': ...}]
                # The _COMPOUND_DTYPE attribute will be set on the dataset in the zarr store
                # which will be used to interpret the data
                array_list = [
                    [
                        _json_serialize(data[name][i], dtype[name], h5_dataset)
                        for name in dtype.names
                    ]
                    for i in range(h5_dataset.shape[0])
                ]
                object_codec = numcodecs.JSON()
                inline_data = array_list + ['|O', list(shape)]
                return ZarrInfoForH5Dataset(
                    shape=shape,
                    chunks=shape,  # be explicit about chunks
                    dtype='object',
                    filters=None,
                    fill_value=' ',  # not sure what to put here
                    object_codec=object_codec,
                    inline_data=json.dumps(inline_data, separators=(',', ':')).encode('utf-8')
                )
            else:
                raise Exception(f'More than one dimension not supported for compound dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')
        else:
            print(dtype.kind)
            raise Exception(f'Not yet implemented (3): dataset {h5_dataset.name} with dtype {dtype} and shape {shape}')


def _json_serialize(val: Any, dtype: np.dtype, h5_dataset: h5py.Dataset) -> Any:
    if dtype.kind in ['i', 'u']:  # integer, unsigned integer
        return int(val)
    elif dtype.kind == 'f':  # float
        return float(val)
    elif dtype.kind == 'b':  # boolean
        return bool(val)
    elif dtype.kind == 'S':  # byte string
        return val.decode()
    elif dtype.kind == 'U':  # unicode string
        return val
    elif dtype == h5py.Reference:
        return _h5_ref_to_zarr_attr(val, label=f'{h5_dataset.name}', h5f=h5_dataset.file)
    else:
        raise Exception(f'Cannot serialize item {val} with dtype {dtype} when serializing dataset {h5_dataset.name} with compound dtype.')


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
