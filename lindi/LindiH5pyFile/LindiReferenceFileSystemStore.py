from typing import Literal, Dict, Union
import os
import json
import time
import base64
import numpy as np
import requests
from zarr.storage import Store as ZarrStore

from ..LocalCache.LocalCache import ChunkTooLargeError, LocalCache
from ..tar.lindi_tar import LindiTarFile


class LindiReferenceFileSystemStore(ZarrStore):
    """
    A Zarr store that reads data from a reference file system.

    The reference file system is based on the ReferenceFileSystem of fsspec, but
    this is a custom implementation that serves some LINDI-specific needs. In
    particular, it handles reading data from DANDI URLs, even when the file is
    part of an embargoed dataset. This requires some special handling as the
    DANDI API URL must be exchanged for a pre-signed S3 bucket URL by
    authenticating with a DANDI API token. This presigned URL expires after a
    period of time, so this Zarr store handles the renewal of the presigned URL.
    It also does the exchange once the first time and caches the redirected URL
    for a period so that the redirect doesn't need to be done every time a
    segment of a file is read.

    To read from a file in an embargoed DANDI dataset, you will need to set the
    DANDI_API_KEY environment variable to your DANDI API token. Or, if this is a
    Dandiset on the staging server, you will need to set the
    DANDI_STAGING_API_KEY.

    Following the fsspec convention
    (https://fsspec.github.io/kerchunk/spec.html), the reference file system is
    specified as a dictionary with a "refs" key. The value of "refs" is a
    dictionary where the keys are the names of the files and each value is
    either a string, a list, or a dict. If the value is a string, it is assumed
    to be the data of the file, which may be base64 encoded (see below). If the
    value is a list, it is assumed to have three elements: the URL of the file
    (or path of a local file), the byte offset of the data within the file, and
    the byte length of the data. Note that we do not permit the case of a list
    of a single (url) element supported by fsspec, because it is good to be able
    to know the size of the chunks without making a request to the file. If the
    value is a dict, it represents a json file, and the content of the file is
    the json representation of the dict.

    If the value for a file is a string, it may be prefixed with "base64:". If
    it is, the string is assumed to be base64 encoded and is decoded before
    being returned. Otherwise, the string is utf-8 encoded and returned as is.
    Note that a file that actually begins with "base64:" should be represented
    by a base64 encoded string, to avoid ambiguity.

    We also support the use of templates as in fsspec, but do not support the
    full jinja2 templating. There may be an optional "templates" key in the
    dictionary, which is a dictionary of template strings. For example, {
        "templates": {"u1": "https://some/url", "u2": "https://some/other/url"},
        "refs": {
            ... "/some/key/0": [
                "{{u1}}" 0, 100
            ],
            ...
        }
    } In this case, the "{{u1}}" will be replaced with the value of the "u1"
    template string.

    Optionally, the reference file system may contain a "generationMetadata"
    key, which is a dictionary of metadata about the generation of the reference
    file system. This metadata is not used by this class, but could be by other
    software. See LindiH5pyFile.write_lindi_file(...)

    It is okay for rfs to be modified outside of this class, and the changes
    will be reflected immediately in the store.
    """
    def __init__(
            self,
            rfs: dict,
            *,
            mode: Literal["r", "r+"] = "r+",
            local_cache: Union[LocalCache, None] = None,
            _source_url_or_path: Union[str, None] = None,
            _source_tar_file: Union[LindiTarFile, None] = None
    ):
        """
        Create a LindiReferenceFileSystemStore.

        Parameters
        ----------
        rfs : dict
            The reference file system (see class docstring for details).
        mode : str
            The mode to open the store in. Only "r" is supported at this time.
        local_cache : LocalCache, optional
            The local cache to use for caching data chunks read from the
            remote URLs. If None, no caching is done.
        """
        if "refs" not in rfs:
            raise Exception("rfs must contain a 'refs' key")

        # validate rfs['refs']
        for k, v in rfs["refs"].items():
            if isinstance(v, str):
                pass
            elif isinstance(v, dict):
                # the content of the file is the json representation of the dict
                pass
            elif isinstance(v, list):
                if len(v) != 3:
                    raise Exception(f"Problem with {k}: list must have 3 elements")
                if not isinstance(v[0], str):
                    raise Exception(f"Problem with {k}: first element must be a string")
                if not isinstance(v[1], int):
                    raise Exception(f"Problem with {k}: second element must be an int")
                if not isinstance(v[2], int):
                    raise Exception(f"Problem with {k}: third element must be an int")
            else:
                raise Exception(f"Problem with {k}: value must be a string or a list")

        # validate templates
        if "templates" in rfs:
            for k, v in rfs["templates"].items():
                if not isinstance(v, str):
                    raise Exception(f"Problem with templates: value for {k} must be a string")

        self.rfs = rfs
        self.mode = mode
        self.local_cache = local_cache
        self._source_url_or_path = _source_url_or_path
        self._source_tar_file = _source_tar_file

    # These methods are overridden from MutableMapping
    def __contains__(self, key: object):
        if not isinstance(key, str):
            return False
        return key in self.rfs["refs"]

    def __getitem__(self, key: str):
        val = self._get_helper(key)

        if val is not None:
            padded_size = _get_padded_size(self, key, val)
            if padded_size is not None:
                val = _pad_chunk(val, padded_size)

        return val

    def _get_helper(self, key: str):
        if key not in self.rfs["refs"]:
            raise KeyError(key)
        x = self.rfs["refs"][key]
        if isinstance(x, str):
            if x.startswith("base64:"):
                return base64.b64decode(x[len("base64:"):])
            else:
                return x.encode("utf-8")
        elif isinstance(x, dict):
            return json.dumps(x).encode("utf-8")
        elif isinstance(x, list):
            if len(x) != 3:
                raise Exception("list must have 3 elements")  # pragma: no cover
            url_or_path = x[0]
            offset = x[1]
            length = x[2]
            if '{{' in url_or_path and '}}' in url_or_path and 'templates' in self.rfs:
                for k, v in self.rfs["templates"].items():
                    url_or_path = url_or_path.replace("{{" + k + "}}", v)
            is_url = url_or_path.startswith('http://') or url_or_path.startswith('https://')
            if url_or_path.startswith('./'):
                if self._source_url_or_path is None:
                    raise Exception(f"Cannot resolve relative path {url_or_path} without source file path")
                if self._source_tar_file is None:
                    raise Exception(f"Cannot resolve relative path {url_or_path} without source file type")
                if self._source_tar_file and (not self._source_tar_file._dir_representation):
                    start_byte, end_byte = self._source_tar_file.get_file_byte_range(file_name=url_or_path[2:])
                    if start_byte + offset + length > end_byte:
                        raise Exception(f"Chunk {key} is out of bounds in tar file {url_or_path}")
                    url_or_path = self._source_url_or_path
                    offset = offset + start_byte
                elif self._source_tar_file and self._source_tar_file._dir_representation:
                    fname = self._source_tar_file._tar_path_or_url + '/' + url_or_path[2:]
                    if not os.path.exists(fname):
                        raise Exception(f"File does not exist: {fname}")
                    file_size = os.path.getsize(fname)
                    if offset + length > file_size:
                        raise Exception(f"Chunk {key} is out of bounds in tar file {url_or_path}: {fname}")
                    url_or_path = fname
                else:
                    if is_url:
                        raise Exception(f"Cannot resolve relative path {url_or_path} for URL that is not a tar")
                    else:
                        source_file_parent_dir = '/'.join(self._source_url_or_path.split('/')[:-1])
                        abs_path = source_file_parent_dir + '/' + url_or_path[2:]
                        url_or_path = abs_path
            if is_url:
                if self.local_cache is not None:
                    x = self.local_cache.get_remote_chunk(url=url_or_path, offset=offset, size=length)
                    if x is not None:
                        return x
                val = _read_bytes_from_url_or_path(url_or_path, offset, length)
                if self.local_cache is not None:
                    try:
                        self.local_cache.put_remote_chunk(url=url_or_path, offset=offset, size=length, data=val)
                    except ChunkTooLargeError:
                        print(f'Warning: unable to cache chunk of size {length} on LocalCache (key: {key})')
            else:
                val = _read_bytes_from_url_or_path(url_or_path, offset, length)
            return val
        else:
            # should not happen given checks in __init__, but self.rfs is mutable
            # and contains mutable lists
            raise Exception(f"Problem with {key}: value {x} must be a string or a list")

    def __setitem__(self, key: str, value: bytes):
        # We intentionally do not allow value to be a dict here! When the rfs is
        # written to a .json file elsewhere in the codebase of lindi, the value
        # will automatically be converted to a json object if it is json
        # serializable.
        if not isinstance(value, bytes):
            raise ValueError("value must be bytes")
        try:
            # try to ascii encode the value
            value2 = value.decode("ascii")
        except UnicodeDecodeError:
            # if that fails, base64 encode it
            value2 = "base64:" + base64.b64encode(value).decode("ascii")
        self.rfs["refs"][key] = value2

    def __delitem__(self, key: str):
        del self.rfs["refs"][key]

    def __iter__(self):
        return iter(self.rfs["refs"])

    def __len__(self):
        return len(self.rfs["refs"])

    # These methods are overridden from BaseStore
    def is_readable(self):
        return self.mode in ["r", "r+"]

    def is_writeable(self):
        return self.mode in ["r+"]

    def is_listable(self):
        return True

    def is_erasable(self):
        return False

    @staticmethod
    def replace_meta_file_contents_with_dicts_in_rfs(rfs: dict) -> None:
        """
        Utility function for replacing the contents of the .zattrs, .zgroup, and
        .zarray files in an rfs with the json representation of the contents.
        """
        # important to use the LindiReferenceFileSystemStore here because then we
        # can resolve any base64 encoded values, etc when converting them to dicts
        store = LindiReferenceFileSystemStore(rfs)
        for k, v in rfs['refs'].items():
            if k.endswith('.zattrs') or k.endswith('.zgroup') or k.endswith('.zarray') or k.endswith('zarr.json'):  # note: zarr.json is for zarr v3
                rfs['refs'][k] = json.loads(store[k].decode('utf-8'))

    @staticmethod
    def use_templates_in_rfs(rfs: dict) -> None:
        """
        Utility for replacing URLs in an rfs with template strings. Only URLs
        that occur 5 or more times are replaced with template strings. The
        templates are added to the "templates" key of the rfs. The template
        strings are of the form "{{u1}}", "{{u2}}", etc.
        """
        url_counts: Dict[str, int] = {}
        for k, v in rfs['refs'].items():
            if isinstance(v, list):
                url = v[0]
                if '{{' not in url:
                    url_counts[url] = url_counts.get(url, 0) + 1
        urls_with_many_occurrences = sorted([url for url, count in url_counts.items() if count >= 5])
        new_templates = rfs.get('templates', {})
        template_names_for_urls: Dict[str, str] = {}
        for template_name, url in new_templates.items():
            template_names_for_urls[url] = template_name
        for url in urls_with_many_occurrences:
            if url in template_names_for_urls:
                continue
            i = 1
            while f'u{i}' in new_templates:
                i += 1
            new_templates[f'u{i}'] = url
            template_names_for_urls[url] = f'u{i}'
        if new_templates:
            rfs['templates'] = new_templates
        for k, v in rfs['refs'].items():
            if isinstance(v, list):
                url = v[0]
                if url in template_names_for_urls:
                    v[0] = '{{' + template_names_for_urls[url] + '}}'

    @staticmethod
    def remove_templates_in_rfs(rfs: dict) -> None:
        """
        Utility for removing templates from an rfs. This is the opposite of
        use_templates_in_rfs.
        """
        templates0 = rfs.get('templates', {})
        for k, v in rfs['refs'].items():
            if isinstance(v, list):
                url = v[0]
                if '{{' in url and '}}' in url:
                    template_name = url[2:-2].strip()
                    if template_name in templates0:
                        v[0] = templates0[template_name]
        rfs['templates'] = {}


def _read_bytes_from_url_or_path(url_or_path: str, offset: int, length: int):
    """
    Read a range of bytes from a URL.
    """
    from ..LindiRemfile.LindiRemfile import _resolve_url
    if url_or_path.startswith('http://') or url_or_path.startswith('https://'):
        num_retries = 8
        for try_num in range(num_retries):
            try:
                url_resolved = _resolve_url(url_or_path)  # handle DANDI auth
                range_start = offset
                range_end = offset + length - 1
                range_header = f"bytes={range_start}-{range_end}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
                    "Range": range_header
                }
                response = requests.get(url_resolved, headers=headers)
                response.raise_for_status()
                return response.content
            except Exception as e:
                if try_num == num_retries - 1:
                    raise e
                else:
                    delay = 0.1 * 2 ** try_num
                    print(f'Retry load data from {url_or_path} in {delay} seconds')
                    time.sleep(delay)
        raise Exception(f"Failed to load data from {url_or_path}")
    else:
        with open(url_or_path, 'rb') as f:
            f.seek(offset)
            return f.read(length)


def _is_chunk_base_key(base_key: str) -> bool:
    a = base_key.split('.')
    if len(a) == 0:
        return False
    for x in a:
        # check if integer
        try:
            int(x)
        except ValueError:
            return False
    return True


def _get_itemsize(dtype: str) -> int:
    d = np.dtype(dtype)
    return d.itemsize


def _pad_chunk(data: bytes, expected_chunk_size: int) -> bytes:
    return data + b'\0' * (expected_chunk_size - len(data))


def _get_padded_size(store, key: str, val: bytes):
    # If the key is a chunk and it's smaller than the expected size, then we
    # need to pad it with zeros. This can happen if this is the final chunk
    # in a contiguous hdf5 dataset. See
    # https://github.com/NeurodataWithoutBorders/lindi/pull/84
    base_key = key.split('/')[-1]
    if val and _is_chunk_base_key(base_key):
        parent_key = key.split('/')[:-1]
        zarray_key = '/'.join(parent_key) + '/.zarray'
        if zarray_key in store:
            zarray_json = store.__getitem__(zarray_key)
            assert isinstance(zarray_json, bytes)
            zarray = json.loads(zarray_json)
            chunk_shape = zarray['chunks']
            dtype = zarray['dtype']
            if np.dtype(dtype).kind in ['i', 'u', 'f']:
                expected_chunk_size = int(np.prod(chunk_shape)) * _get_itemsize(dtype)
                if len(val) < expected_chunk_size:
                    return expected_chunk_size

    return None
