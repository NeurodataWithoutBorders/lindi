import pytest
import json
import math
import numpy as np
from lindi.conversion.reformat_json import reformat_json
from lindi.conversion.nan_inf_ninf import decode_nan_inf_ninf, encode_nan_inf_ninf
from lindi.conversion.h5_ref_to_zarr_attr import _decode_if_needed
from lindi.conversion.decode_references import decode_references
from lindi.conversion.attr_conversion import h5_to_zarr_attr, zarr_to_h5_attr
from lindi.LindiH5pyFile.LindiH5pyReference import LindiH5pyReference


def test_reformat_json():
    assert reformat_json(None) is None
    assert reformat_json(json.dumps({"a": 1}).encode("utf-8")) == b'{"a":1}'
    assert reformat_json(json.dumps({"a": 1.0}).encode("utf-8")) == b'{"a":1.0}'
    with pytest.raises(ValueError):
        reformat_json(json.dumps({"a": float("nan")}).encode("utf-8"))
    with pytest.raises(ValueError):
        reformat_json(json.dumps({"a": float("inf")}).encode("utf-8"))
    with pytest.raises(ValueError):
        reformat_json(json.dumps({"a": float("-inf")}).encode("utf-8"))


def test_decode_nan_inf_ninf():
    assert math.isnan(decode_nan_inf_ninf('NaN'))  # type: ignore
    assert decode_nan_inf_ninf('Infinity') == float('inf')
    assert decode_nan_inf_ninf('-Infinity') == float('-inf')
    assert decode_nan_inf_ninf('a') == 'a'
    assert decode_nan_inf_ninf([1, 'Infinity', '-Infinity']) == [1, float('inf'), float('-inf')]
    assert decode_nan_inf_ninf({'b': 'Infinity', 'c': '-Infinity'}) == {'b': float('inf'), 'c': float('-inf')}


def test_encode_nan_inf_ninf():
    assert encode_nan_inf_ninf(float('nan')) == 'NaN'
    assert encode_nan_inf_ninf(float('inf')) == 'Infinity'
    assert encode_nan_inf_ninf(float('-inf')) == '-Infinity'
    assert encode_nan_inf_ninf('a') == 'a'
    assert encode_nan_inf_ninf([1, float('nan'), float('inf'), float('-inf')]) == [1, 'NaN', 'Infinity', '-Infinity']
    assert encode_nan_inf_ninf({'a': float('nan'), 'b': float('inf'), 'c': float('-inf')}) == {'a': 'NaN', 'b': 'Infinity', 'c': '-Infinity'}


def test_decode_if_needed():
    assert _decode_if_needed(b'abc') == 'abc'
    assert _decode_if_needed('abc') == 'abc'


def test_decode_references():
    x = decode_references({
        '_REFERENCE': {
            'object_id': 'a',
            'path': 'b',
            'source': 'c',
            'source_object_id': 'd'
        }
    })
    assert isinstance(x, LindiH5pyReference)
    assert x._object_id == 'a'
    assert x._path == 'b'
    assert x._source == 'c'
    assert x._source_object_id == 'd'
    x = decode_references([{
        '_REFERENCE': {
            'object_id': 'a',
            'path': 'b',
            'source': 'c',
            'source_object_id': 'd'
        }
    }])
    assert isinstance(x, list)
    assert isinstance(x[0], LindiH5pyReference)
    x = decode_references(np.array([{
        '_REFERENCE': {
            'object_id': 'a',
            'path': 'b',
            'source': 'c',
            'source_object_id': 'd'
        }
    }], dtype=object))
    assert isinstance(x, np.ndarray)
    assert isinstance(x[0], LindiH5pyReference)
    with pytest.raises(Exception):
        x = decode_references({
            'a': {
                'b': {
                    '_REFERENCE': {
                        'object_id': 'a',
                        'path': 'b',
                        'source': 'c',
                        'source_object_id': 'd'
                    }
                }
            }
        })


def test_h5_to_zarr_attr():
    # lists
    assert h5_to_zarr_attr([]) == []
    assert h5_to_zarr_attr([1, 2, 3]) == [1, 2, 3]
    assert h5_to_zarr_attr([1.0, float('nan'), float('inf'), float('-inf')]) == [1, 'NaN', 'Infinity', '-Infinity']
    assert h5_to_zarr_attr([True, False]) == [True, False]
    assert h5_to_zarr_attr([np.bool_(True), np.bool_(False)]) == [True, False]
    with pytest.raises(Exception):
        h5_to_zarr_attr([1, 2, 'a'])

    # special str
    with pytest.raises(ValueError):
        h5_to_zarr_attr('NaN')
    with pytest.raises(ValueError):
        h5_to_zarr_attr('Infinity')
    with pytest.raises(ValueError):
        h5_to_zarr_attr('-Infinity')

    # None
    with pytest.raises(Exception):
        h5_to_zarr_attr(None)

    # int
    assert h5_to_zarr_attr(1) == 1
    assert h5_to_zarr_attr(np.int8(1)) == 1
    assert h5_to_zarr_attr(np.int16(1)) == 1
    assert h5_to_zarr_attr(np.int32(1)) == 1
    assert h5_to_zarr_attr(np.int64(1)) == 1
    assert h5_to_zarr_attr(np.uint8(1)) == 1
    assert h5_to_zarr_attr(np.uint16(1)) == 1
    assert h5_to_zarr_attr(np.uint32(1)) == 1
    assert h5_to_zarr_attr(np.uint64(1)) == 1

    # float
    assert h5_to_zarr_attr(1.0) == 1.0
    assert h5_to_zarr_attr(np.float16(1.0)) == 1.0
    assert h5_to_zarr_attr(np.float32(1.0)) == 1.0
    assert h5_to_zarr_attr(np.float64(1.0)) == 1.0

    # complex
    with pytest.raises(Exception):
        h5_to_zarr_attr(complex(1, 2))
    with pytest.raises(Exception):
        h5_to_zarr_attr(np.complex64(1))

    # bool
    assert h5_to_zarr_attr(True) is True
    assert h5_to_zarr_attr(False) is False
    assert h5_to_zarr_attr(np.bool_(True)) is True
    assert h5_to_zarr_attr(np.bool_(False)) is False

    # tuple
    with pytest.raises(Exception):
        h5_to_zarr_attr((1, 2))

    # dict
    with pytest.raises(Exception):
        h5_to_zarr_attr({'a': 1})

    # set
    with pytest.raises(Exception):
        h5_to_zarr_attr({1, 2})

    # str and bytes
    assert h5_to_zarr_attr('abc') == 'abc'
    assert h5_to_zarr_attr(b'abc') == 'abc'

    # numpy array
    assert h5_to_zarr_attr(np.array([1, 2, 3])) == [1, 2, 3]
    assert h5_to_zarr_attr(np.array([1.0, float('nan'), float('inf'), float('-inf')])) == [1, 'NaN', 'Infinity', '-Infinity']
    assert h5_to_zarr_attr(np.array(['a', 'b', 'c'])) == ['a', 'b', 'c']
    assert h5_to_zarr_attr(np.array([b'a', b'b', b'c'])) == ['a', 'b', 'c']
    assert h5_to_zarr_attr(np.array([True, False])) == [True, False]
    assert h5_to_zarr_attr(np.array([1, 2, 3], dtype='int64')) == [1, 2, 3]
    assert h5_to_zarr_attr(np.array([1.0, 2.0, 3.0], dtype='float64')) == [1, 2, 3]
    with pytest.raises(Exception):
        h5_to_zarr_attr(np.array([1, 2], dtype=np.complex128))


def test_zarr_to_h5_attr():
    # str
    assert zarr_to_h5_attr('abc') == 'abc'

    # int
    assert zarr_to_h5_attr(1) == 1

    # float
    assert zarr_to_h5_attr(1.0) == 1.0

    # bool
    assert zarr_to_h5_attr(True) is True
    assert zarr_to_h5_attr(False) is False

    # list
    assert zarr_to_h5_attr([]).tolist() == []  # type: ignore
    assert zarr_to_h5_attr([1, 2, 3]).tolist() == [1, 2, 3]  # type: ignore
    assert zarr_to_h5_attr([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]  # type: ignore
    assert zarr_to_h5_attr([True, False]).tolist() == [True, False]  # type: ignore
    assert zarr_to_h5_attr(['a', 'b', 'c']).tolist() == ['a', 'b', 'c']  # type: ignore
    assert zarr_to_h5_attr([b'a', b'b', b'c']).tolist() == [b'a', b'b', b'c']  # type: ignore
    with pytest.raises(Exception):
        zarr_to_h5_attr([1, 'a', 2.0, True])
    with pytest.raises(Exception):
        zarr_to_h5_attr([1 + 2j])
