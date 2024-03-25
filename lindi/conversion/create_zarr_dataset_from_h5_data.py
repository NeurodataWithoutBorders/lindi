from typing import Union, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import numcodecs
import h5py
import zarr
from .attr_conversion import h5_to_zarr_attr


def create_zarr_dataset_from_h5_data(
    zarr_parent_group: zarr.Group,
    shape: Tuple,
    dtype: Any,
    data: Any,
    h5f: Union[h5py.File, None],
    name: str,
    label: str,
    chunks: Union[Tuple, None]
):
    if dtype is None:
        raise Exception(f'No dtype in h5_to_zarr_dataset_prep for dataset {label}')
    if len(shape) == 0:
        # scalar dataset
        # zarr doesn't support scalar datasets, so we make an array of shape (1,)
        # and the _SCALAR attribute will be set to True elsewhere to indicate that
        # it is a scalar dataset

        numeric_format_str = _get_numeric_format_str(dtype)
        if numeric_format_str is not None:
            # Handle the simple numeric types
            return zarr_parent_group.create_dataset(
                name,
                shape=(1,),
                chunks=(1,),
                data=[data[()]] if isinstance(data, h5py.Dataset) or isinstance(data, np.ndarray) else [data],
            )
        elif dtype.kind == 'O':
            # For type object, we are going to use the JSON codec
            # for encoding [scalar_value]
            scalar_value = data[()]
            if isinstance(scalar_value, (bytes, str)):
                if isinstance(scalar_value, bytes):
                    scalar_value = scalar_value.decode()
                return zarr_parent_group.create_dataset(
                    name,
                    shape=(1,),
                    chunks=(1,),
                    data=[scalar_value]
                )
            else:
                raise Exception(f'Unsupported scalar value type: {type(scalar_value)}')
        else:
            raise Exception(f'Cannot handle scalar dataset {label} with dtype {dtype}')
    else:
        # not a scalar dataset
        if dtype.kind in ['i', 'u', 'f', 'b']:  # integer, unsigned integer, float, boolean
            # This is the normal case of a chunked dataset with a numeric (or boolean) dtype
            if chunks is None:
                # We require that chunks be specified when writing a dataset with more
                # than 1 million elements. This is because zarr may default to
                # suboptimal chunking. Note that the default for h5py is to use the
                # entire dataset as a single chunk.
                total_size = np.prod(shape) if len(shape) > 0 else 1
                if total_size > 1000 * 1000:
                    raise Exception(f'Chunks must be specified when writing dataset of shape {shape}')
            if isinstance(data, list):
                data = np.array(data)
            # Note that we are not using the same filters as in the h5py dataset
            return zarr_parent_group.create_dataset(
                name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                data=data
            )
        elif dtype.kind == 'O':
            # For type object, we are going to use the JSON codec
            if isinstance(data, h5py.Dataset):
                data = data[:]
            data_vec_view = data.ravel()
            for i, val in enumerate(data_vec_view):
                if isinstance(val, bytes):
                    data_vec_view[i] = val.decode()
                elif isinstance(val, str):
                    data_vec_view[i] = val
                elif isinstance(val, h5py.Reference):
                    # encode the reference as a dictionary with a special key
                    data_vec_view[i] = h5_to_zarr_attr(val, label=f'{label}[{i}]', h5f=h5f)
                else:
                    raise Exception(f'Cannot handle dataset {label} with dtype {dtype} and shape {shape}')
            object_codec = numcodecs.JSON()
            return zarr_parent_group.create_dataset(
                name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                data=data,
                object_codec=object_codec
            )
        elif dtype.kind in 'SU':  # byte string or unicode string
            raise Exception(f'Not yet implemented (2): dataset {label} with dtype {dtype} and shape {shape}')
        elif dtype.kind == 'V' and dtype.fields is not None:  # compound dtype
            if isinstance(data, list):
                data = np.array(data)
            data_1d_view = data.ravel()
            data2 = np.empty(shape, dtype='object')
            data2_1d_view = data2.ravel()
            for i in range(len(data_1d_view)):
                elmt = tuple([
                    _make_json_serializable(
                        data_1d_view[i][field_name],
                        dtype[field_name],
                        label=f'{label}[{i}].{field_name}',
                        h5f=h5f
                    )
                    for field_name in dtype.names
                ])
                data2_1d_view[i] = elmt
            return zarr_parent_group.create_dataset(
                name,
                shape=shape,
                chunks=chunks,
                dtype='object',
                data=data2,
                object_codec=numcodecs.JSON()
            )
        else:
            print(dtype.kind)
            raise Exception(f'Not yet implemented (3): dataset {label} with dtype {dtype} and shape {shape}')


@dataclass
class CreateZarrDatasetInfo:
    shape: Tuple
    dtype: Any
    fill_value: Any
    scalar: bool
    compound_dtype: Union[List[Tuple[str, str]], None]


def _make_json_serializable(val: Any, dtype: np.dtype, label: str, h5f: Union[h5py.File, None]) -> Any:
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
        return h5_to_zarr_attr(val, label=label, h5f=h5f)
    else:
        raise Exception(f'Cannot serialize item {val} with dtype {dtype} when serializing dataset {label} with compound dtype.')


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
