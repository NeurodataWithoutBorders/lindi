import pytest
import json
import math
import numpy as np
from lindi.conversion.reformat_json import reformat_json
from lindi.conversion.nan_inf_ninf import decode_nan_inf_ninf, encode_nan_inf_ninf
from lindi.conversion.h5_ref_to_zarr_attr import _decode_if_needed
from lindi.conversion.decode_references import decode_references
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


if __name__ == "__main__":
    test_reformat_json()
