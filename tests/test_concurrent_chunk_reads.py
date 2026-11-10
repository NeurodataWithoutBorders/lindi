import os
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import h5py
import numpy as np
import pytest

import lindi


class _RangeRequestHandler(SimpleHTTPRequestHandler):
    """Serve byte ranges of files and record how many requests are in flight."""

    stats = None

    def do_GET(self):
        stats = type(self).stats
        with stats["lock"]:
            stats["num_requests"] += 1
            stats["in_flight"] += 1
            stats["max_in_flight"] = max(stats["max_in_flight"], stats["in_flight"])
        try:
            # A small delay makes overlapping requests observable.
            time.sleep(0.02)
            path = self.translate_path(self.path)
            with open(path, "rb") as f:
                data = f.read()
            range_header = self.headers.get("Range")
            if range_header is None:
                self.send_response(200)
                body = data
            else:
                start, end = range_header.removeprefix("bytes=").split("-")
                body = data[int(start):int(end) + 1]
                self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with stats["lock"]:
                stats["in_flight"] -= 1

    def log_message(self, format, *args):
        pass


@pytest.fixture
def served_hdf5_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test.h5")
        data = np.arange(100 * 64, dtype="float32")
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=data, chunks=(64,), compression="gzip")

        stats = {"lock": threading.Lock(), "num_requests": 0, "in_flight": 0, "max_in_flight": 0}
        handler = type("Handler", (_RangeRequestHandler,), {"stats": stats})

        def make_handler(*args, **kwargs):
            return handler(*args, directory=tmpdir, **kwargs)

        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/test.h5"
            with lindi.LindiH5ZarrStore.from_file(filename, url=url) as store:
                rfs = store.to_reference_file_system()
            yield rfs, data, stats, tmpdir
        finally:
            server.shutdown()
            server.server_close()


def test_chunks_are_fetched_concurrently(served_hdf5_file):
    rfs, data, stats, _ = served_hdf5_file
    client = lindi.LindiH5pyFile.from_reference_file_system(rfs)
    stats["num_requests"] = 0

    result = client["data"][:]  # type: ignore

    np.testing.assert_array_equal(result, data)
    assert stats["num_requests"] == 100
    assert stats["max_in_flight"] > 1


def test_concurrent_reads_use_local_cache(served_hdf5_file):
    rfs, data, stats, tmpdir = served_hdf5_file
    local_cache = lindi.LocalCache(cache_dir=os.path.join(tmpdir, "local_cache"))
    client = lindi.LindiH5pyFile.from_reference_file_system(rfs, local_cache=local_cache)
    np.testing.assert_array_equal(client["data"][:], data)  # type: ignore
    stats["num_requests"] = 0

    result = client["data"][:]  # type: ignore

    np.testing.assert_array_equal(result, data)
    assert stats["num_requests"] == 0
