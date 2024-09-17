from typing import Union, List, Any, Tuple, Literal
from dataclasses import dataclass
import numpy as np
import numcodecs
from numcodecs.abc import Codec
import h5py
import zarr
from .h5_ref_to_zarr_attr import h5_ref_to_zarr_attr
from .attr_conversion import h5_to_zarr_attr
from ._util import _is_numeric_dtype


def create_zarr_dataset_from_h5_data(
    zarr_parent_group: zarr.Group,
    h5_shape: Tuple,
    h5_dtype: Any,
    h5_data: Union[Any, None],
    h5f: Union[h5py.File, None],
    name: str,
    label: str,
    h5_chunks: Union[Tuple, None],
    zarr_compressor: Union[Codec, Literal['default'], None] = 'default'
):
    """Create a zarr dataset from an h5py dataset.

    Parameters
    ----------
    zarr_parent_group : zarr.Group
        The parent group in the zarr hierarchy. The new dataset will be created
        in this group.
    h5_shape : tuple
        The shape of the h5py dataset.
    h5_dtype : numpy.dtype
        The dtype of the h5py dataset.
    h5_data : any
        The data of the h5py dataset. If None, the dataset will be created
        without data.
    h5f : h5py.File
        The file that the h5py dataset is in.
    name : str
        The name of the new dataset in the zarr hierarchy.
    label : str
        The name of the h5py dataset for error messages.
    h5_chunks : tuple
        The chunk shape of the h5py dataset.
    zarr_compressor : numcodecs.abc.Codec, 'default', or None
        The codec compressor to use when writing the dataset. If default, the
        default compressor will be used. When None, no compressor will be used.
    """
    if h5_dtype is None:
        raise Exception(f'No dtype in h5_to_zarr_dataset_prep for dataset {label}')
    if len(h5_shape) == 0:
        # scalar dataset
        # zarr doesn't support scalar datasets, so we make an array of shape (1,)
        # and the _SCALAR attribute will be set to True elsewhere to indicate that
        # it is a scalar dataset

        if h5_data is None:
            raise Exception(f'Data must be provided for scalar dataset {label}')

        if zarr_compressor != 'default' and zarr_compressor is not None:
            raise Exception('zarr_compressor is not supported for scalar datasets')

        if np.issubdtype(h5_dtype, np.complexfloating):
            raise Exception(f'Complex scalar datasets are not supported: dataset {label} with dtype {h5_dtype}')

        if _is_numeric_dtype(h5_dtype) or h5_dtype in [bool, np.bool_]:
            # Handle the simple numeric types
            ds = zarr_parent_group.create_dataset(
                name,
                shape=(1,),
                chunks=(1,),
                data=[h5_data[()]] if isinstance(h5_data, h5py.Dataset) or isinstance(h5_data, np.ndarray) else [h5_data],
            )
            ds.attrs['_SCALAR'] = True
            return ds
        elif h5_dtype.kind == 'O':
            # For type object, we are going to use the JSON codec
            # for encoding [scalar_value]
            scalar_value = h5_data[()] if isinstance(h5_data, h5py.Dataset) or isinstance(h5_data, np.ndarray) else h5_data
            if isinstance(scalar_value, (bytes, str)):
                if isinstance(scalar_value, bytes):
                    scalar_value = scalar_value.decode()
                ds = zarr_parent_group.create_dataset(
                    name,
                    shape=(1,),
                    chunks=(1,),
                    data=[scalar_value]
                )
                ds.attrs['_SCALAR'] = True
                return ds
            else:
                raise Exception(f'Unsupported scalar value type: {type(scalar_value)}')
        elif h5_dtype.kind == 'S' or h5_dtype.kind == 'U':
            # byte string
            if h5_data is None:
                raise Exception(f'Data must be provided for scalar dataset {label}')
            scalar_value = h5_data[()] if isinstance(h5_data, h5py.Dataset) or isinstance(h5_data, np.ndarray) else h5_data
            ds = zarr_parent_group.create_dataset(
                name,
                shape=(1,),
                chunks=(1,),
                data=[scalar_value]
            )
            ds.attrs['_SCALAR'] = True
            return ds
        else:
            raise Exception(f'Cannot handle scalar dataset {label} with dtype {h5_dtype}')
    else:
        # not a scalar dataset

        if isinstance(h5_data, list):
            # If we have a list, then we need to convert it to an array
            h5_data = np.array(h5_data)

        if np.issubdtype(h5_dtype, np.complexfloating):
            raise Exception(f'Complex datasets are not supported: dataset {label} with dtype {h5_dtype}')

        if _is_numeric_dtype(h5_dtype) or h5_dtype in [bool, np.bool_]:  # integer, unsigned integer, float, bool
            # This is the normal case of a chunked dataset with a numeric (or boolean) dtype
            if h5_chunks is None:
                # # We require that chunks be specified when writing a dataset with more
                # # than 1 million elements. This is because zarr may default to
                # # suboptimal chunking. Note that the default for h5py is to use the
                # # entire dataset as a single chunk.
                # total_size = int(np.prod(h5_shape)) if len(h5_shape) > 0 else 1
                # if total_size > 1000 * 1000:
                #     raise Exception(f'Chunks must be specified explicitly when writing dataset of shape {h5_shape}')
                h5_chunks = _get_default_chunks(h5_shape, h5_dtype)
            # Note that we are not using the same filters as in the h5py dataset
            return zarr_parent_group.create_dataset(
                name,
                shape=h5_shape,
                chunks=h5_chunks,
                dtype=h5_dtype,
                data=h5_data,
                compressor=zarr_compressor
            )
        elif h5_dtype.kind == 'O':
            # For type object, we are going to use the JSON codec
            if zarr_compressor != 'default' and zarr_compressor is not None:
                raise Exception('zarr_compressor is not supported for object datasets')
            if h5_data is not None:
                if isinstance(h5_data, h5py.Dataset):
                    h5_data = h5_data[:]
                zarr_data = h5_object_data_to_zarr_data(h5_data, h5f=h5f, label=label)
            else:
                zarr_data = None
            object_codec = numcodecs.JSON()
            return zarr_parent_group.create_dataset(
                name,
                shape=h5_shape,
                chunks=h5_chunks,
                dtype=h5_dtype,
                data=zarr_data,
                object_codec=object_codec
            )
        elif h5_dtype.kind == 'S':  # byte string
            if zarr_compressor != 'default' and zarr_compressor is not None:
                raise Exception('zarr_compressor is not supported for byte string datasets')
            if h5_data is None:
                raise Exception(f'Data must be provided when converting dataset {label} with dtype {h5_dtype}')
            return zarr_parent_group.create_dataset(
                name,
                shape=h5_shape,
                chunks=h5_chunks,
                dtype=h5_dtype,
                data=h5_data
            )
        elif h5_dtype.kind == 'U':  # unicode string
            if zarr_compressor != 'default' and zarr_compressor is not None:
                raise Exception('zarr_compressor is not supported for unicode string datasets')
            raise Exception(f'Array of unicode strings not supported: dataset {label} with dtype {h5_dtype} and shape {h5_shape}')
        elif h5_dtype.kind == 'V' and h5_dtype.fields is not None:  # compound dtype
            if zarr_compressor != 'default' and zarr_compressor is not None:
                raise Exception('zarr_compressor is not supported for compound datasets')
            if h5_data is None:
                raise Exception(f'Data must be provided when converting compound dataset {label}')
            h5_data_1d_view = h5_data.ravel()
            zarr_data = np.empty(h5_shape, dtype='object')
            zarr_data_1d_view = zarr_data.ravel()
            for i in range(len(h5_data_1d_view)):
                elmt = tuple([
                    _make_json_serializable(
                        h5_data_1d_view[i][field_name],
                        h5_dtype[field_name],
                        label=f'{label}[{i}].{field_name}',
                        h5f=h5f
                    )
                    for field_name in h5_dtype.names
                ])
                zarr_data_1d_view[i] = elmt
            ds = zarr_parent_group.create_dataset(
                name,
                shape=h5_shape,
                chunks=h5_chunks,
                dtype='object',
                data=zarr_data,
                object_codec=numcodecs.JSON()
            )
            compound_dtype = []
            for name in h5_dtype.names:
                tt = h5_dtype[name]
                if tt == h5py.special_dtype(ref=h5py.Reference):
                    tt = "<REFERENCE>"
                compound_dtype.append((name, str(tt)))
            ds.attrs['_COMPOUND_DTYPE'] = compound_dtype
            return ds
        else:
            raise Exception(f'Not yet implemented (3): dataset {label} with dtype {h5_dtype} and shape {h5_shape}')


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


def h5_object_data_to_zarr_data(h5_data: Union[np.ndarray, list], *, h5f: Union[h5py.File, None], label: str) -> np.ndarray:
    from ..LindiH5pyFile.LindiH5pyReference import LindiH5pyReference  # Avoid circular import
    if isinstance(h5_data, list):
        h5_data = np.array(h5_data)
    zarr_data = np.empty(h5_data.shape, dtype='object')
    h5_data_1d_view = h5_data.ravel()
    zarr_data_1d_view = zarr_data.ravel()
    for i, val in enumerate(h5_data_1d_view):
        if isinstance(val, bytes):
            zarr_data_1d_view[i] = val.decode()
        elif isinstance(val, str):
            zarr_data_1d_view[i] = val
        elif isinstance(val, LindiH5pyReference):
            zarr_data_1d_view[i] = {
                '_REFERENCE': val._obj
            }
        elif isinstance(val, h5py.Reference):
            if h5f is None:
                raise Exception(f'h5f cannot be None when converting h5py.Reference to zarr attribute at {label}')
            zarr_data_1d_view[i] = h5_ref_to_zarr_attr(val, h5f=h5f)
        else:
            raise Exception(f'Cannot handle value of type {type(val)} in dataset {label} with dtype {h5_data.dtype} and shape {h5_data.shape}')
    return zarr_data


def _get_default_chunks(shape: Tuple, dtype: Any) -> Tuple:
    dtype_size = np.dtype(dtype).itemsize
    shape_prod_0 = np.prod(shape[1:])
    optimal_chunk_size_bytes = 1024 * 1024 * 20  # 20 MB
    optimal_chunk_size = optimal_chunk_size_bytes // (dtype_size * shape_prod_0)
    if optimal_chunk_size <= shape[0]:
        return shape
    if optimal_chunk_size < 1:
        return (1,) + shape[1:]
    return (optimal_chunk_size,) + shape[1:]
