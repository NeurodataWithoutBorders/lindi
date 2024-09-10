from typing import Union
import time
import os
import requests
from .additional_url_resolvers import get_additional_url_resolvers
from ..LocalCache.LocalCache import LocalCache

default_min_chunk_size = 128 * 1024  # This is different from Remfile - this is an important decision because it determines the chunk size in the LocalCache
default_max_cache_size = 1024 * 1024 * 1024
default_chunk_increment_factor = 1.7
default_max_chunk_size = 100 * 1024 * 1024


class LindiRemfile:
    def __init__(
        self,
        url: str,
        *,
        verbose: bool = False,
        local_cache: Union[LocalCache, None],
        _min_chunk_size: int = default_min_chunk_size,  # recommended not to change because it will make the accumulated cache in LocalCache useless
        _max_cache_size: int = default_max_cache_size,
        _chunk_increment_factor: float = default_chunk_increment_factor,
        _max_chunk_size: int = default_max_cache_size,
        _impose_request_failures_for_testing: bool = False,
    ):
        """Create a file-like object for reading a remote file. Optimized for reading hdf5 files. The arguments starting with an underscore are for testing and debugging purposes - they may experience breaking changes in the future.

        Args:
            url (str): The url of the remote file, or an object with a .get_url() method. The latter is useful if the url is a presigned AWS URL that expires after a certain amount of time.
            verbose (bool, optional): Whether to print info for debugging. Defaults to False.
            local_cache (LocalCache, optional): The local cache for storing the chunks of the file. Defaults to None.
            _min_chunk_size (int, optional): The minimum chunk size. When reading, the chunks will be loaded in multiples of this size.
            _max_cache_size (int, optional): The maximum number of bytes to keep in the cache.
            _chunk_increment_factor (int, optional): The factor by which to increase the number of chunks to load when the system detects that the chunks are being loaded in order.
            _max_chunk_size (int, optional): The maximum chunk size. When reading, the chunks will be loaded in multiples of the minimum chunk size up to this size.
            _impose_request_failures_for_testing (bool, optional): Whether to impose request failures for testing purposes. Defaults to False.

        Difference between this and regular RemFile?
            Uses Lindi's LocalCache instead of Remfile's DiskCache
            Requires that url is a string (does not accept object with .get_url() function)
            Does not support using multiple threads
            Does not use memory cache if LocalCache is specified
            Handles DANDI authentication
            Handles additional_url_resolver

        A note:
            In the context of LINDI, this LindiRemfile is going to be used for loading
            metadata of the hdf5 file. The large chunks are going to be loaded using
            a different (Zarr) mechanism. That's one reason for the above differences.
        """
        if not isinstance(url, str):
            # Do this check, because regular Remfile allows for objects
            raise Exception('Only string urls are supported for LindiRemfile')
        self._url = url
        self._verbose = verbose
        self._local_cache = local_cache
        self._min_chunk_size = _min_chunk_size
        self._max_chunks_in_cache = int(_max_cache_size / _min_chunk_size)
        self._chunk_increment_factor = _chunk_increment_factor
        self._max_chunk_size = _max_chunk_size
        self._impose_request_failures_for_testing = _impose_request_failures_for_testing
        self._memory_chunks = {}
        self._memory_chunk_indices: list[
            int
        ] = (
            []
        )  # list of chunk indices in order of loading for purposes of cleaning up the cache
        self._position = 0
        self._smart_loader_last_chunk_index_accessed = -99
        self._smart_loader_chunk_sequence_length = 1

        # use aborted GET request rather than HEAD request to get the length
        # this is needed for presigned AWS URLs because HEAD requests are not supported
        response = requests.get(_resolve_url(self._url), stream=True)
        if response.status_code == 200:
            self.length = int(response.headers["Content-Length"])
        else:
            raise Exception(
                f"Error getting file length: {response.status_code} {response.reason}"
            )
        # Close the connection without reading the content to avoid downloading the whole file
        response.close()

        # response = requests.head(_get_url_str(self._url))
        # self.length = int(response.headers['Content-Length'])
        self.session = requests.Session()

    def read(self, size=None):
        """Read bytes from the file.

        Args:
            size (_type_): The number of bytes to read.

        Raises:
            Exception: If the size argument is not provided.

        Returns:
            bytes: The bytes read.
        """
        if size is None:
            raise Exception(
                "The size argument must be provided in remfile"
            )  # pragma: no cover

        chunk_start_index = self._position // self._min_chunk_size
        chunk_end_index = (self._position + size - 1) // self._min_chunk_size
        loaded_chunks = {}
        for chunk_index in range(chunk_start_index, chunk_end_index + 1):
            loaded_chunks[chunk_index] = self._load_chunk(chunk_index)
        if chunk_end_index == chunk_start_index:
            chunk = loaded_chunks[chunk_start_index]
            chunk_offset = self._position % self._min_chunk_size
            chunk_length = size
            self._position += size
            return chunk[chunk_offset: chunk_offset + chunk_length]
        else:
            pieces_to_concat = []
            for chunk_index in range(chunk_start_index, chunk_end_index + 1):
                chunk = loaded_chunks[chunk_index]
                if chunk_index == chunk_start_index:
                    chunk_offset = self._position % self._min_chunk_size
                    chunk_length = self._min_chunk_size - chunk_offset
                elif chunk_index == chunk_end_index:
                    chunk_offset = 0
                    chunk_length = size - sum([len(p) for p in pieces_to_concat])
                else:
                    chunk_offset = 0
                    chunk_length = self._min_chunk_size
                pieces_to_concat.append(
                    chunk[chunk_offset: chunk_offset + chunk_length]
                )
        ret = b"".join(pieces_to_concat)
        self._position += size

        # clean up the cache
        if len(self._memory_chunk_indices) > self._max_chunks_in_cache:
            if self._verbose:
                print("Cleaning up cache")
            for chunk_index in self._memory_chunk_indices[
                : int(self._max_chunks_in_cache * 0.5)
            ]:
                if chunk_index in self._memory_chunks:
                    del self._memory_chunks[chunk_index]
                else:
                    # it is possible that the chunk was already deleted (repeated chunk index in the list)
                    pass
            self._memory_chunk_indices = self._memory_chunk_indices[
                int(self._max_chunks_in_cache * 0.5):
            ]

        return ret

    def _load_chunk(self, chunk_index: int) -> bytes:
        """Load a chunk of the file.

        Args:
            chunk_index (int): The index of the chunk to load.
        """
        if chunk_index in self._memory_chunks:
            self._smart_loader_last_chunk_index_accessed = chunk_index
            return self._memory_chunks[chunk_index]

        if self._local_cache is not None:
            cached_value = self._local_cache.get_remote_chunk(
                url=self._url,
                offset=chunk_index * self._min_chunk_size,
                size=min(self._min_chunk_size, self.length - chunk_index * self._min_chunk_size),
            )
            if cached_value is not None:
                if self._local_cache is None:
                    self._memory_chunks[chunk_index] = cached_value
                    self._memory_chunk_indices.append(chunk_index)
                self._smart_loader_last_chunk_index_accessed = chunk_index
                return cached_value

        if chunk_index == self._smart_loader_last_chunk_index_accessed + 1:
            # round up to the chunk sequence length times 1.7
            self._smart_loader_chunk_sequence_length = round(
                self._smart_loader_chunk_sequence_length * 1.7 + 0.5
            )
            if (
                self._smart_loader_chunk_sequence_length > self._max_chunk_size / self._min_chunk_size
            ):
                self._smart_loader_chunk_sequence_length = int(
                    self._max_chunk_size / self._min_chunk_size
                )
            # make sure the chunk sequence length is valid
            for j in range(1, self._smart_loader_chunk_sequence_length):
                if chunk_index + j in self._memory_chunks:
                    # already loaded this chunk
                    self._smart_loader_chunk_sequence_length = j
                    break
        else:
            self._smart_loader_chunk_sequence_length = round(
                self._smart_loader_chunk_sequence_length / 1.7 + 0.5
            )
        data_start = chunk_index * self._min_chunk_size
        data_end = (
            data_start + self._min_chunk_size * self._smart_loader_chunk_sequence_length - 1
        )
        if self._verbose:
            print(
                f"Loading {self._smart_loader_chunk_sequence_length} chunks starting at {chunk_index} ({(data_end - data_start + 1)/1e6} million bytes)"
            )
        if data_end >= self.length:
            data_end = self.length - 1
        x = _get_bytes(
            self.session,
            _resolve_url(self._url),
            data_start,
            data_end,
            verbose=self._verbose,
            _impose_request_failures_for_testing=self._impose_request_failures_for_testing,
        )
        if not x:
            raise Exception(f'Error loading chunk {chunk_index} from {self._url}')
        if self._smart_loader_chunk_sequence_length == 1:
            if self._local_cache is None:
                self._memory_chunks[chunk_index] = x
            if self._local_cache is not None:
                size = min(self._min_chunk_size, self.length - chunk_index * self._min_chunk_size)
                self._local_cache.put_remote_chunk(
                    url=self._url,
                    offset=chunk_index * self._min_chunk_size,
                    size=size,
                    data=x
                )
            self._memory_chunk_indices.append(chunk_index)
        else:
            for i in range(self._smart_loader_chunk_sequence_length):
                if i * self._min_chunk_size >= len(x):
                    break
                if self._local_cache is None:
                    self._memory_chunks[chunk_index + i] = x[
                        i * self._min_chunk_size: (i + 1) * self._min_chunk_size
                    ]
                    self._memory_chunk_indices.append(chunk_index + i)
                if self._local_cache is not None:
                    size = min(self._min_chunk_size, self.length - (chunk_index + i) * self._min_chunk_size)
                    data = x[i * self._min_chunk_size: (i + 1) * self._min_chunk_size]
                    if len(data) != size:
                        raise ValueError(f'Unexpected: len(data) != size: {len(data)} != {size}')
                    self._local_cache.put_remote_chunk(
                        url=self._url,
                        offset=(chunk_index + i) * self._min_chunk_size,
                        size=size,
                        data=data
                    )
        self._smart_loader_last_chunk_index_accessed = (
            chunk_index + self._smart_loader_chunk_sequence_length - 1
        )
        return x[: self._min_chunk_size]

    def seek(self, offset: int, whence: int = 0):
        """Seek to a position in the file.

        Args:
            offset (int): The offset to seek to.
            whence (int, optional): The code for the reference point for the offset. Defaults to 0.

        Raises:
            ValueError: If the whence argument is not 0, 1, or 2.
        """
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset  # pragma: no cover
        elif whence == 2:
            self._position = self.length + offset
        else:
            raise ValueError(
                "Invalid argument: 'whence' must be 0, 1, or 2."
            )  # pragma: no cover

    def tell(self):
        return self._position

    def close(self):
        pass


_num_request_retries = 8


def _get_bytes(
    session: requests.Session,
    url: str,
    start_byte: int,
    end_byte: int,
    *,
    verbose=False,
    _impose_request_failures_for_testing=False,
):
    """Get bytes from a remote file.

    Args:
        url (str): The url of the remote file.
        start_byte (int): The first byte to get.
        end_byte (int): The last byte to get.
        verbose (bool, optional): Whether to print info for debugging. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Function to be used in threads for fetching the byte ranges
    def fetch_bytes(range_start: int, range_end: int, num_retries: int, verbose: bool):
        """Fetch a range of bytes from a remote file using the range header

        Args:
            range_start (int): The first byte to get.
            range_end (int): The last byte to get.
            num_retries (int): The number of retries.

        Returns:
            bytes: The bytes fetched.
        """
        for try_num in range(num_retries + 1):
            try:
                actual_url = url
                if _impose_request_failures_for_testing:
                    if try_num == 0:
                        actual_url = "_error_" + url
                range_header = f"bytes={range_start}-{range_end}"
                # response = requests.get(actual_url, headers={'Range': range_header})
                # use session to avoid creating a new connection each time
                response = session.get(actual_url, headers={"Range": range_header})
                return response.content
            except Exception as e:
                if try_num == num_retries:
                    raise e  # pragma: no cover
                else:
                    delay = 0.1 * 2**try_num
                    if verbose:
                        print(f"Retrying after exception: {e}")
                        print(f"Waiting {delay} seconds")
                    time.sleep(delay)

    return fetch_bytes(start_byte, end_byte, _num_request_retries, verbose)


_global_resolved_urls = {}  # url -> {timestamp, url}


def _is_dandi_url(url: str):
    if url.startswith('https://api.dandiarchive.org/api/'):
        return True
    if url.startswith('https://api-staging.dandiarchive.org/'):
        return True
    return False


def _resolve_dandi_url(url: str):
    resolve_with_dandi_api_key = None
    if url.startswith('https://api.dandiarchive.org/api/'):
        dandi_api_key = os.environ.get('DANDI_API_KEY', None)
        if dandi_api_key is not None:
            resolve_with_dandi_api_key = dandi_api_key
    elif url.startswith('https://api-staging.dandiarchive.org/'):
        dandi_api_key = os.environ.get('DANDI_STAGING_API_KEY', None)
        if dandi_api_key is not None:
            resolve_with_dandi_api_key = dandi_api_key
    headers = {}
    if resolve_with_dandi_api_key is not None:
        headers['Authorization'] = f'token {resolve_with_dandi_api_key}'
    # do it synchronously here
    resp = requests.head(url, allow_redirects=True, headers=headers)
    return str(resp.url)


def _resolve_url(url: str):
    for aur in get_additional_url_resolvers():
        url = aur(url)
    if url in _global_resolved_urls:
        elapsed = time.time() - _global_resolved_urls[url]["timestamp"]
        if elapsed < 60 * 10:
            return _global_resolved_urls[url]["url"]
    if _is_dandi_url(url):
        resolved_url = _resolve_dandi_url(url)
    else:
        resolved_url = url
    _global_resolved_urls[url] = {"timestamp": time.time(), "url": resolved_url}
    return resolved_url
