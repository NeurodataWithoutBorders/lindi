from typing import Union
import os
import time
import requests
import remfile
from .FileSegmentReader import FileSegmentReader


class DandiFileSegmentReader(FileSegmentReader):
    """
    A class that reads a segment of a file from a DANDI URL.

    See the documentation for LindiReferenceFileSystemStore for more information
    on why this class is needed.
    """
    def __init__(self, url: str):
        # we intentionally do not call the super constructor
        if not DandiFileSegmentReader.is_dandi_url(url):
            raise Exception(f"{url} is not a dandi url")
        self.url = url

        # The DandiFile has a get_url() method which is in charge of resolving
        # the URL, possibly using the DANDI API token, caching the result, and
        # renewing periodically. remfile.File will accept such an object, and
        # will call .get_url() as needed.
        dandi_file = DandiFile(url)
        self.remfile = remfile.File(dandi_file, verbose=False)

    def read(self, offset: int, length: int):
        self.remfile.seek(offset)
        return self.remfile.read(length)

    @staticmethod
    def is_dandi_url(url: str):
        if url.startswith('https://api.dandiarchive.org/api/'):
            return True
        if url.startswith('https://api-staging.dandiarchive.org/'):
            return True
        return False


class DandiFile:
    def __init__(self, url: str):
        """
        Create a new DandiFile which is in charge of resolving a DANDI URL
        possibly using the DANDI API token, caching the result, and periodically
        renewing the result.
        """
        self._url = url
        self._resolved_url: Union[str, None] = None
        self._resolved_url_timestamp: Union[float, None] = None

    def get_url(self) -> str:
        if self._resolved_url is not None and self._resolved_url_timestamp is not None:
            if time.time() - self._resolved_url_timestamp < 120:
                return self._resolved_url
        resolve_with_dandi_api_key = None
        if self._url.startswith('https://api.dandiarchive.org/api/'):
            dandi_api_key = os.environ.get('DANDI_API_KEY', None)
            if dandi_api_key is not None:
                resolve_with_dandi_api_key = dandi_api_key
        elif self._url.startswith('https://api-staging.dandiarchive.org/'):
            dandi_api_key = os.environ.get('DANDI_STAGING_API_KEY', None)
            if dandi_api_key is not None:
                resolve_with_dandi_api_key = dandi_api_key
        url0 = _resolve_dandi_url(url=self._url, dandi_api_key=resolve_with_dandi_api_key)
        self._resolved_url = url0
        self._resolved_url_timestamp = time.time()
        return self._resolved_url

# Example of URL resolution with DANDI API token:
# https://api.dandiarchive.org/api/dandisets/000939/versions/0.240318.1555/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/
# resolves to pre-signed S3 URL
# https://dandiarchive.s3.amazonaws.com/blobs/a2b/94f/a2b94f91-6a75-43d8-b5db-21d89449f481?response-content-disposition=attachment%3B%20filename%3D%22sub-A3701_ses-191119_behavior%2Becephys.nwb%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAUBRWC5GAEKH3223E%2F20240321%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20240321T122953Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=ed8cfd7a9d17a226cc18e20b5f32cfe9c467025c226f1abba236793e645dfcb0


def _resolve_dandi_url(url: str, dandi_api_key: Union[str, None]) -> str:
    headers = {}
    if dandi_api_key is not None:
        headers['Authorization'] = f'token {dandi_api_key}'
    # do it synchronously here
    resp = requests.head(url, allow_redirects=True, headers=headers)
    return str(resp.url)
