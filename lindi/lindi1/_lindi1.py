import os
import json
import tempfile
import urllib.request


def _load_rfs_dict_from_local_lindi_file(filename: str):
    header_buf = _read_bytes_from_file(filename, 0, 1024)
    lindi1_header = _parse_lindi1_header(header_buf)
    if lindi1_header is not None:
        # This is lindi1 format
        assert lindi1_header['format'] == 'lindi1'
        rfs_start = lindi1_header['rfs_start']
        rfs_size = lindi1_header['rfs_size']
        rfs_buf = _read_bytes_from_file(filename, rfs_start, rfs_start + rfs_size)
        return json.loads(rfs_buf), True
    else:
        # In this case, it must be a regular json file
        with open(filename, "r") as f:
            return json.load(f), False


def _load_rfs_dict_from_remote_lindi_file(url: str):
    file_size = _get_file_size_of_remote_file(url)
    if file_size < 1024 * 1024 * 2:
        # if it's a small file, we'll just download the whole thing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_fname = f"{tmpdir}/temp.lindi.json"
            _download_file(url, tmp_fname)
            return _load_rfs_dict_from_local_lindi_file(tmp_fname)
    else:
        # if it's a large file, we start by downloading the first 1024 bytes
        header_buf = _download_file_byte_range(url, 0, 1024)
        lindi1_header = _parse_lindi1_header(header_buf)
        if lindi1_header is not None:
            # This is lindi1 format
            assert lindi1_header['format'] == 'lindi1'
            rfs_start = lindi1_header['rfs_start']
            rfs_size = lindi1_header['rfs_size']
            rfs_buf = _download_file_byte_range(url, rfs_start, rfs_start + rfs_size)
            return json.loads(rfs_buf), True
        else:
            # In this case, it must be a regular json file
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_fname = f"{tmpdir}/temp.lindi.json"
                _download_file(url, tmp_fname)
                with open(tmp_fname, "r") as f:
                    return json.load(f), False


def _write_rfs_dict_to_lindi1_file(*, rfs: dict, output_file_name: str):
    rfs_buf = json.dumps(rfs).encode("utf-8")
    header_buf = _read_bytes_from_file(output_file_name, 0, 1024)
    lindi1_header = _parse_lindi1_header(header_buf)
    assert lindi1_header is not None
    assert lindi1_header['format'] == 'lindi1'
    old_rfs_start = lindi1_header['rfs_start']
    old_rfs_size = lindi1_header['rfs_size']
    old_rfs_padding = lindi1_header['rfs_padding']
    if len(rfs_buf) < old_rfs_size + old_rfs_padding - 1:
        # we don't need to allocate a new space in the file for the rfs
        rfs_buf_padded = rfs_buf + b"\0" * (old_rfs_size + old_rfs_padding - len(rfs_buf))
        _write_bytes_within_file(output_file_name, old_rfs_start, rfs_buf_padded)
        new_rfs_start = old_rfs_start
        new_rfs_size = len(rfs_buf)
        new_rfs_padding = old_rfs_size + old_rfs_padding - len(rfs_buf)
    else:
        # we need to allocate a new space. First zero out the old space
        zeros = b"\0" * (old_rfs_size + old_rfs_padding)
        _write_bytes_within_file(output_file_name, old_rfs_start, zeros)
        file_size = os.path.getsize(output_file_name)
        new_rfs_start = file_size
        # determine size of new space, to be double the needed size
        new_rfs_size = len(rfs_buf) * 2
        new_rfs_padding = new_rfs_size - len(rfs_buf)
        new_rfs_buf_padded = rfs_buf + b"\0" * new_rfs_padding
        # write the new rfs
        _append_bytes_to_file(output_file_name, new_rfs_buf_padded)
    new_lindi1_header = {
        **lindi1_header,
        "rfs_start": new_rfs_start,
        "rfs_size": new_rfs_size,
        "rfs_padding": new_rfs_padding
    }
    new_lindi1_header_buf = json.dumps(new_lindi1_header).encode("utf-8")
    if len(new_lindi1_header_buf) > 1024:
        raise Exception("New header is too long")
    new_lindi1_header_buf_padded = new_lindi1_header_buf + b"\0" * (1024 - len(new_lindi1_header_buf))
    _write_bytes_within_file(output_file_name, 0, new_lindi1_header_buf_padded)


def _download_file(url: str, filename: str) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(filename, "wb") as f:
            f.write(response.read())


def _download_file_byte_range(url: str, start: int, end: int) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Range": f"bytes={start}-{end - 1}"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return response.read()


def _read_bytes_from_file(filename: str, start: int, end: int) -> bytes:
    with open(filename, "rb") as f:
        f.seek(start)
        return f.read(end - start)


def _write_bytes_within_file(filename: str, start: int, buf: bytes) -> None:
    with open(filename, "r+b") as f:
        f.seek(start)
        f.write(buf)


def _append_bytes_to_file(filename: str, buf: bytes) -> None:
    with open(filename, "ab") as f:
        f.write(buf)


def _parse_lindi1_header(buf: bytes):
    first_zero_index = buf.find(0)
    if first_zero_index == -1:
        return None
    header_json = buf[:first_zero_index].decode("utf-8")
    header: dict = json.loads(header_json)
    if header.get('format') != 'lindi1':
        raise Exception(f"Not lindi1 format: {header.get('format')}")
    return header


def _get_file_size_of_remote_file(url: str) -> int:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return int(response.headers['Content-Length'])


def _create_empty_lindi_file(filename: str, create_binary: bool):
    empty_rfs = {
        "refs": {
            '.zgroup': {
                'zarr_format': 2
            }
        },
    }
    if create_binary:
        rfs_buf = json.dumps(empty_rfs).encode("utf-8")
        # start with reasonable padding
        total_rfs_allocated_space = 1024 * 1024
        while len(rfs_buf) * 2 > total_rfs_allocated_space:
            total_rfs_allocated_space *= 2
        lindi1_header = {
            "format": "lindi1",
            "rfs_start": 1024,
            "rfs_size": len(rfs_buf),
            "rfs_padding": total_rfs_allocated_space - len(rfs_buf)
        }
        lindi1_header_buf = json.dumps(lindi1_header).encode("utf-8")
        lindi1_header_buf_padded = lindi1_header_buf + b"\0" * (1024 - len(lindi1_header_buf))
        with open(filename, "wb") as f:
            f.write(lindi1_header_buf_padded)
            f.write(rfs_buf)
            f.write(b"\0" * lindi1_header['rfs_padding'])
    else:
        with open(filename, "w") as f:
            json.dump(empty_rfs, f, indent=2)