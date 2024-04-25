from typing import Literal, Dict, Union
import json
import base64
import requests
from zarr.storage import Store as ZarrStore

from ..LocalCache.LocalCache import LocalCache


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
    dictionary, which is a dictionary of template strings. For example,
    {
        "templates": {"u1": "https://some/url", "u2": "https://some/other/url"},
        "refs": {
            ... "/some/key/0": [
                "{{u1}}" 0, 100
            ],
            ...
        }
    }
    In this case, the "{{u1}}" will be replaced with the value of the "u1"
    template string.

    It is okay for rfs to be modified outside of this class, and the changes
    will be reflected immediately in the store. This can be used by experimental
    tools such as lindi-cloud.
    """
    def __init__(self, rfs: dict, *, mode: Literal["r", "r+"] = "r+", local_cache: Union[LocalCache, None] = None):
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

    # These methods are overridden from MutableMapping
    def __getitem__(self, key: str):
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
            url = x[0]
            offset = x[1]
            length = x[2]
            if '{{' in url and '}}' in url and 'templates' in self.rfs:
                for k, v in self.rfs["templates"].items():
                    url = url.replace("{{" + k + "}}", v)
            if self.local_cache is not None:
                x = self.local_cache.get_remote_chunk(url=url, offset=offset, size=length)
                if x is not None:
                    return x
            val = _read_bytes_from_url_or_path(url, offset, length)
            if self.local_cache is not None:
                self.local_cache.put_remote_chunk(url=url, offset=offset, size=length, data=val)
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


def _read_bytes_from_url_or_path(url_or_path: str, offset: int, length: int):
    """
    Read a range of bytes from a URL.
    """
    from ..LindiRemfile.LindiRemfile import _resolve_url
    if url_or_path.startswith('http://') or url_or_path.startswith('https://'):
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
    else:
        with open(url_or_path, 'rb') as f:
            f.seek(offset)
            return f.read(length)
