from typing import Literal, Dict
import base64
from zarr.storage import Store as ZarrStore
from .FileSegmentReader.FileSegmentReader import FileSegmentReader
from .FileSegmentReader.DandiFileSegmentReader import DandiFileSegmentReader


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
    DANDI_API_KEY environment variable to your DANDI API token. Or, if this is
    a Dandiset on the staging server, you will need to set the
    DANDI_STAGING_API_KEY.

    Following the fsspec convention (https://fsspec.github.io/kerchunk/spec.html),
    the reference file system is specified as a
    dictionary with a "refs" key. The value of "refs" is a dictionary where the
    keys are the names of the files and the values are either strings or lists.
    If the value is a string, it is assumed to be the data of the file, which
    may be base64 encoded (see below). If the value is a list, it is assumed to
    have three elements: the URL of the file (or path of a local file), the byte
    offset of the data within the file, and the byte length of the data.

    If the value for a file is a string, it may be prefixed with "base64:". If
    it is, the string is assumed to be base64 encoded and is decoded before
    being returned. Otherwise, the string is utf-8 encoded and returned as is.
    Note that a file that actually begins with "base64:" should be represented
    by a base64 encoded string, to avoid ambiguity.
    """
    def __init__(self, rfs: dict, mode: Literal["r"] = "r"):
        """
        Create a LindiReferenceFileSystemStore.

        Parameters
        ----------
        rfs : dict
            The reference file system (see class docstring for details).
        mode : str
            The mode to open the store in. Only "r" is supported at this time.
        """
        if "refs" not in rfs:
            raise Exception("rfs must contain a 'refs' key")

        # validate rfs['refs']
        for k, v in rfs["refs"].items():
            if isinstance(v, str):
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

        self.rfs = rfs
        self.mode = mode

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
        elif isinstance(x, list):
            if len(x) != 3:
                raise Exception("list must have 3 elements")  # pragma: no cover
            url = x[0]
            offset = x[1]
            length = x[2]
            val = _read_bytes_from_url(url, offset, length)
            return val
        else:
            # should not happen given checks in __init__, but self.rfs is mutable
            # and contains mutable lists
            raise Exception(f"Problem with {key}: value must be a string or a list")

    def __setitem__(self, key: str, value):
        raise Exception("Setting items is not allowed")

    def __delitem__(self, key: str):
        raise Exception("Deleting items is not allowed")

    def __iter__(self):
        return iter(self.rfs["refs"])

    def __len__(self):
        return len(self.rfs["refs"])

    # These methods are overridden from BaseStore
    def is_readable(self):
        return self.mode in ["r"]

    def is_writeable(self):
        return False

    def is_listable(self):
        return True

    def is_erasable(self):
        return False


# Keep a global cache of file segment readers that apply to all instances of
# LindiReferenceFileSystemStore. The key is the URL of the file.
_file_segment_readers: Dict[str, FileSegmentReader] = {}


def _read_bytes_from_url(url: str, offset: int, length: int):
    """
    Read a range of bytes from a URL.
    """
    if url not in _file_segment_readers:
        if DandiFileSegmentReader.is_dandi_url(url):
            # This is a DANDI URL, so it needs to be handled specially
            # see the docstring for DandiFileSegmentReader for details
            file_segment_reader = DandiFileSegmentReader(url)
        else:
            # This is a non-DANDI URL or local file path
            file_segment_reader = FileSegmentReader(url)
        _file_segment_readers[url] = file_segment_reader
    return _file_segment_readers[url].read(offset, length)
