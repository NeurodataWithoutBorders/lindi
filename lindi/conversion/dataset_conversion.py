import json
from typing import Union, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import numcodecs
import h5py
import struct
from .attr_conversion import h5_to_zarr_attr
from ._h5_filters_to_codecs import _h5_filters_to_codecs
from .resolve_references import resolve_references


@dataclass
class CreateZarrDatasetInfo:
    shape: Tuple
    chunks: Tuple
    dtype: Any
    fill_value: Any
    scalar: bool
    compound_dtype: Union[List[Tuple[str, str]], None]


def h5_to_zarr_dataset(
    h5_dataset: h5py.Dataset
):
    filters = _h5_filters_to_codecs(h5_dataset)
    label = h5_dataset.name or '<>'
    info = h5_to_zarr_dataset_prep(
        shape=h5_dataset.shape,
        dtype=h5_dataset.dtype,
        scalar_value=h5_dataset[()] if len(h5_dataset.shape) == 0 else None,
        label=label,
        chunks=h5_dataset.chunks
    )
    shape = h5_dataset.shape
    inline_data_bytes = None
    object_code = None
    if info.scalar:
        scalar_value = h5_dataset[()]
        if isinstance(scalar_value, bytes):
            scalar_value = scalar_value.decode()
        numeric_format_str = _get_numeric_format_str(h5_dataset.dtype)
        if numeric_format_str is not None:
            inline_data_bytes = struct.pack(numeric_format_str, scalar_value)
        elif h5_dataset.dtype.kind == 'O':
            object_code = numcodecs.JSON()
            inline_data_bytes = json.dumps([scalar_value, '|O', [1]], separators=(',', ':')).encode('utf-8')
    else:
        if h5_dataset.dtype.kind == 'O':
            # For type object, we are going to use the JSON codec
            # which requires inline data of the form [list, of, items, ..., '|O', [n1, n2, ...]]
            data = h5_dataset[:]
            data_vec_view = data.ravel()
            for i, val in enumerate(data_vec_view):
                if isinstance(val, bytes):
                    data_vec_view[i] = val.decode()
                elif isinstance(val, str):
                    data_vec_view[i] = val
                elif isinstance(val, h5py.Reference):
                    data_vec_view[i] = h5_to_zarr_attr(val, label=f'{label}[{i}]', h5f=h5_dataset.file)
                else:
                    raise Exception(f'Cannot handle dataset {label} with dtype {h5_dataset.dtype} and shape {shape}')
            object_code = numcodecs.JSON()
            inline_data_bytes = json.dumps(data.tolist() + ['|O', list(shape)], separators=(',', ':')).encode('utf-8')
        elif info.compound_dtype is not None:
            assert len(shape) == 1  # only handle 1d (see below)
            # We will load it into memory and create inline data
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
                    _make_json_serializable(
                        data[i][name],
                        h5_dataset.dtype[name],
                        h5f=h5_dataset.file,
                        label=f'{label}[{i}].{name}'
                    )
                    for name in h5_dataset.dtype.names
                ]
                for i in range(shape[0])
            ]
            object_code = numcodecs.JSON()
            inline_data_list = array_list + ['|O', list(shape)]
            inline_data_bytes = json.dumps(inline_data_list, separators=(',', ':')).encode('utf-8')
    return info, filters, inline_data_bytes, object_code


def h5_to_zarr_dataset_prep(
    shape: Tuple,
    dtype: Any,
    scalar_value: Union[Any, None],
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
            return CreateZarrDatasetInfo(
                shape=(1,),
                chunks=(1,),  # be explicit about chunks
                dtype=dtype,
                fill_value=0,
                scalar=True,
                compound_dtype=None
            )
        elif dtype.kind == 'O':
            # For type object, we are going to use the JSON codec
            # which requires inline data of the form [[val], '|O', [1]]
            if isinstance(scalar_value, (bytes, str)):
                if isinstance(scalar_value, bytes):
                    scalar_value = scalar_value.decode()
                return CreateZarrDatasetInfo(
                    shape=(1,),
                    chunks=(1,),  # be explicit about chunks
                    dtype=dtype,
                    fill_value=' ',
                    scalar=True,
                    compound_dtype=None
                )
            else:
                raise Exception(f'Unsupported scalar object type: {type(scalar_value)}')
        else:
            raise Exception(f'Cannot handle scalar dataset {label} with dtype {dtype}')
    else:
        # not a scalar dataset
        if dtype.kind in ['i', 'u', 'f', 'b']:  # integer, unsigned integer, float, boolean
            # This is the normal case of a chunked dataset with a numeric (or boolean) dtype
            if chunks is None:
                # If the dataset is not chunked, we use the entire dataset as a single chunk
                # It's important to be explicit about the chunks, because I think None means that zarr could decide otherwise
                chunks = shape
            return CreateZarrDatasetInfo(
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                fill_value=0,  # is this right?
                scalar=False,
                compound_dtype=None
            )
        elif dtype.kind == 'O':
            return CreateZarrDatasetInfo(
                shape=shape,
                chunks=shape,  # be explicit about chunks
                dtype=dtype,
                fill_value=' ',  # not sure what to put here
                scalar=False,
                compound_dtype=None
            )
        elif dtype.kind in 'SU':  # byte string or unicode string
            raise Exception(f'Not yet implemented (2): dataset {label} with dtype {dtype} and shape {shape}')
        elif dtype.kind == 'V' and dtype.fields is not None:  # void (i.e. compound)
            if len(shape) == 1:
                # for now we only handle the case of a 1D compound dataset
                compound_dtype = []
                for i in range(len(dtype.names)):
                    name = dtype.names[i]
                    tt = dtype.fields[name][0]
                    if tt == h5py.special_dtype(ref=h5py.Reference):
                        tt = '<REFERENCE>'
                    compound_dtype.append((name, str(tt)))
                return CreateZarrDatasetInfo(
                    shape=shape,
                    chunks=shape,  # be explicit about chunks
                    dtype='object',
                    fill_value=' ',  # not sure what to put here
                    scalar=False,
                    compound_dtype=compound_dtype
                )
            else:
                raise Exception(f'More than one dimension not supported for compound dataset {label} with dtype {dtype} and shape {shape}')
        else:
            print(dtype.kind)
            raise Exception(f'Not yet implemented (3): dataset {label} with dtype {dtype} and shape {shape}')


def h5_object_data_to_zarr_object_data(data: Any) -> Any:
    from ..LindiH5pyFile.LindiH5pyReference import LindiH5pyReference  # Avoid circular import
    if isinstance(data, list):
        return [h5_object_data_to_zarr_object_data(x) for x in data]
    elif isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, str):
        return data
    elif isinstance(data, LindiH5pyReference):
        return {
            '_REFERENCE': data._obj
        }
    else:
        raise Exception(f"Unexpected type for object value: {type(data)}")


@dataclass
class H5InfoForZarrDataset:
    shape: Tuple
    dtype: Any
    data: Any
    chunks: Union[Tuple, None]


def zarr_to_h5_dataset(
    *,
    shape: Union[Tuple, None],
    dtype: Any,
    data=Union[Any, None],
    chunks: Union[Tuple, None],
    is_scalar: bool,
    compound_dtype: Union[List[Tuple[str, str]], None]
):
    if isinstance(data, list):
        data = np.array(data)

    if shape is None:
        if data is not None:
            if isinstance(data, np.ndarray):
                shape = data.shape
            else:
                shape = (1,)
        else:
            raise Exception('shape must be provided if data is None')

    if dtype is None:
        if isinstance(data, np.ndarray):
            dtype = data.dtype
        else:
            dtype = np.dtype(type(data))

    # We require that chunks be specified when writing a dataset with more
    # than 1 million elements. This is because zarr may default to
    # suboptimal chunking. Note that the default for h5py is to use the
    # entire dataset as a single chunk.
    total_size = np.prod(shape) if len(shape) > 0 else 1
    if total_size > 1000 * 1000:
        if chunks is None:
            raise Exception(f'Chunks must be specified when writing dataset of shape {shape}')

    # in this case we are creating a dataset with no data
    if is_scalar:
        if compound_dtype is not None:
            raise Exception(f'Unexpected compound_dtype in zarr_to_h5_dataset for scalar dataset: {compound_dtype}')
        h5_shape = ()
        h5_chunks = None
        if dtype.kind in ['i', 'u', 'f', 'b']:
            h5_dtype = dtype
        elif dtype.kind == 'O':
            h5_dtype = dtype
        else:
            raise Exception(f'Unexpected dtype in zarr_to_h5_dataset for scalar dataset: {dtype}')
        return H5InfoForZarrDataset(
            shape=h5_shape,
            dtype=h5_dtype,
            data=data,
            chunks=h5_chunks
        )
    else:
        # not scalar
        if dtype.kind in ['i', 'u', 'f', 'b']:
            # This is the normal case of a chunked dataset with a numeric (or boolean) dtype
            if compound_dtype is not None:
                raise Exception(f'Unexpected compound_dtype in zarr_to_h5_dataset for dataset with numeric dtype: {compound_dtype}')
            return H5InfoForZarrDataset(
                shape=shape,
                dtype=dtype,
                data=data,
                chunks=chunks
            )
        elif dtype.kind == 'O':
            if compound_dtype is not None:
                if len(shape) != 1:
                    raise Exception(f'Unexpected shape in zarr_to_h5_dataset for compound dataset: {shape}')
                h5_dtype = []
                for i in range(len(compound_dtype)):
                    name, dtype_str = compound_dtype[i]
                    if dtype_str == '<REFERENCE>':
                        h5_dtype.append((name, h5py.special_dtype(ref=h5py.Reference)))
                    else:
                        h5_dtype.append((name, dtype_str))
                if data is not None:
                    h5_data = resolve_references(data)
                else:
                    h5_data = None
                return H5InfoForZarrDataset(
                    shape=shape,
                    dtype=h5_dtype,
                    data=h5_data,
                    chunks=chunks
                )
            else:
                # not a compound dataset
                if data is not None:
                    h5_data = resolve_references(data)
                else:
                    h5_data = None
                return H5InfoForZarrDataset(
                    shape=shape,
                    dtype=dtype,
                    data=h5_data,
                    chunks=chunks
                )
        else:
            raise Exception(f'Not yet implemented (4): dataset with dtype {dtype}')


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
