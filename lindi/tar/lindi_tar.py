import json
import random
import urllib.request
from .create_tar_header import create_tar_header


TAR_ENTRY_JSON_SIZE = 1024
INITIAL_TAR_INDEX_JSON_SIZE = 1024 * 8
INITIAL_LINDI_JSON_SIZE = 1024 * 8


class LindiTarFile:
    def __init__(self, tar_path_or_url: str):
        self._tar_path_or_url = tar_path_or_url
        self._is_remote = tar_path_or_url.startswith("http://") or tar_path_or_url.startswith("https://")

        # Load the entry json
        entry_json = _load_bytes_from_local_or_remote_file(self._tar_path_or_url, 512, 512 + TAR_ENTRY_JSON_SIZE)
        entry = json.loads(entry_json)
        index_info = entry['index']

        # Load the index json
        index_json = _load_bytes_from_local_or_remote_file(self._tar_path_or_url, index_info['d'], index_info['d'] + index_info['s'])
        self._index = json.loads(index_json)
        self._index_has_changed = False

        self._index_lookup = {}
        for file in self._index['files']:
            self._index_lookup[file['n']] = file

        self._file = open(self._tar_path_or_url, "r+b") if not self._is_remote else None

    def close(self):
        self._update_index()
        if self._file is not None:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_file_info(self, file_name: str):
        return self._index_lookup.get(file_name, None)

    def overwrite_file_content(self, file_name: str, data: bytes):
        if self._is_remote:
            raise ValueError("Cannot overwrite file content in a remote tar file")
        if self._file is None:
            raise ValueError("File is not open")
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found")
        if info['s'] != len(data):
            raise ValueError("Unexpected problem in overwrite_file_content(): data size must match the size of the existing file")
        self._file.seek(info['d'])
        self._file.write(data)

    def trash_file(self, file_name: str, do_write_index=True):
        if self._is_remote:
            raise ValueError("Cannot trash a file in a remote tar file")
        if self._file is None:
            raise ValueError("File is not open")
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found")
        zeros = b"-" * info['s']
        self._file.seek(info['d'])
        self._file.write(zeros)
        self._change_name_of_file(file_name, f'.trash/{file_name}.{_create_random_string()}', do_write_index=do_write_index)

    def write_rfs(self, rfs: dict):
        rfs_json = json.dumps(rfs, indent=2, sort_keys=True)

        existing_lindi_json_info = self.get_file_info("lindi.json")
        if existing_lindi_json_info is not None:
            file_size = existing_lindi_json_info['s']
            if file_size >= len(rfs_json):
                # We are going to overwrite the existing lindi.json with the new
                # one. But first we pad it with spaces to the same size as the
                # existing one.
                padding = b" " * (file_size - len(rfs_json))
                rfs_json = rfs_json.encode() + padding
                self.overwrite_file_content("lindi.json", rfs_json)
            else:
                # In this case we need to trash the existing file and write a new one
                # at the end of the tar file.
                self.trash_file("lindi.json")
                rfs_json = _pad_bytes_to_leave_room_for_growth(rfs_json, INITIAL_LINDI_JSON_SIZE)
                self.write_file("lindi.json", rfs_json)
        else:
            # We are writing a new lindi.json.
            rfs_json = _pad_bytes_to_leave_room_for_growth(rfs_json, INITIAL_LINDI_JSON_SIZE)
            self.write_file("lindi.json", rfs_json)

    def get_file_byte_range(self, file_name: str) -> tuple:
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found in tar file")
        return info['d'], info['d'] + info['s']

    def has_file_with_name(self, file_name: str) -> bool:
        return self.get_file_info(file_name) is not None

    def _change_name_of_file(self, file_name: str, new_file_name: str, do_write_index=True):
        if self._is_remote:
            raise ValueError("Cannot change the name of a file in a remote tar file")
        if self._file is None:
            raise ValueError("File is not open")
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found")
        header_start_byte = info['o']
        file_name_byte_range = (header_start_byte + 0, header_start_byte + 100)
        file_name_prefix_byte_range = (header_start_byte + 345, header_start_byte + 345 + 155)
        self._file.seek(file_name_byte_range[0])
        self._file.write(new_file_name.encode())
        # set the rest of the field to zeros
        self._file.write(b"\x00" * (file_name_byte_range[1] - file_name_byte_range[0] - len(new_file_name)))

        self._file.seek(file_name_prefix_byte_range[0])
        self._file.write(b"\x00" * (file_name_prefix_byte_range[1] - file_name_prefix_byte_range[0]))

        _fix_checksum_in_header(self._file, header_start_byte)
        file_in_index = self._index_lookup.get(file_name, None)
        if file_in_index is None:
            raise ValueError(f"File {file_name} not found in index")
        file_in_index['n'] = new_file_name
        self._index_has_changed = True

    def write_file(self, file_name: str, data: bytes):
        if self._is_remote:
            raise ValueError("Cannot write a file in a remote tar file")
        if self._file is None:
            raise ValueError("File is not open")
        self._file.seek(-1024, 2)
        hh = self._file.read(1024)
        if hh != b"\x00" * 1024:
            raise ValueError("The tar file does not end with 1024 bytes of zeros")
        self._file.seek(-1024, 2)

        file_pos = self._file.tell()
        x = {
            'n': file_name,
            'o': file_pos,
            'd': file_pos + 512,  # we assume the header is 512 bytes
            's': len(data)
        }

        # write the tar header
        tar_header = create_tar_header(file_name, len(data))
        self._file.write(tar_header)
        # write the data
        self._file.write(data)

        # pad up to blocks of 512
        if len(data) % 512 != 0:
            padding = b"\x00" * (512 - len(data) % 512)
            self._file.write(padding)

        # write the 1024 bytes marking the end of the file
        self._file.write(b"\x00" * 1024)

        self._index['files'].append(x)
        self._index_lookup[file_name] = x
        self._index_has_changed = True

    def read_file(self, file_name: str) -> bytes:
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found")
        start_byte = info['d']
        size = info['s']
        return _load_bytes_from_local_or_remote_file(self._tar_path_or_url, start_byte, start_byte + size)

    @staticmethod
    def create(fname: str, *, rfs: dict):
        with open(fname, "wb") as f:
            # Define the sizes and names of the entry and index files
            tar_entry_json_name = ".tar_entry.json"
            tar_entry_json_size = TAR_ENTRY_JSON_SIZE
            tar_index_json_size = INITIAL_TAR_INDEX_JSON_SIZE
            tar_index_json_name = ".tar_index.json"
            tar_index_json_offset = 512 + TAR_ENTRY_JSON_SIZE
            tar_index_json_offset_data = tar_index_json_offset + 512

            # Define the content of .tar_entry.json
            initial_entry_json = json.dumps({
                'index': {
                    'n': tar_index_json_name,
                    'o': tar_index_json_offset,
                    'd': tar_index_json_offset_data,
                    's': tar_index_json_size
                }
            }, indent=2, sort_keys=True)
            initial_entry_json = initial_entry_json.encode() + b" " * (tar_entry_json_size - len(initial_entry_json))

            # Define the content of .tar_index.json
            initial_index_json = json.dumps({
                'files': [
                    {
                        'n': tar_entry_json_name,
                        'o': 0,
                        'd': 512,
                        's': tar_entry_json_size
                    },
                    {
                        'n': tar_index_json_name,
                        'o': tar_index_json_offset,
                        'd': tar_index_json_offset_data,
                        's': tar_index_json_size
                    }
                ]
            }, indent=2, sort_keys=True)
            initial_index_json = initial_index_json.encode() + b" " * (tar_index_json_size - len(initial_index_json))

            # Write the initial entry file (.tar_entry.json). This will always
            # be the first file in the tar file, and has a fixed size.
            header = create_tar_header(tar_entry_json_name, tar_entry_json_size)
            f.write(header)
            f.write(initial_entry_json)

            # Write the initial index file (.tar_index.json) this will start as
            # the second file in the tar file but as it grows outside the
            # initial bounds, a new index file will be appended to the end of
            # the tar, and then entry file will be updated accordingly to point
            # to the new index file.
            header = create_tar_header(tar_index_json_name, tar_index_json_size)
            f.write(header)
            f.write(initial_index_json)

            with open(fname, "ab") as f:
                f.write(b"\x00" * 1024)

        # write the rfs file
        tf = LindiTarFile(fname)
        tf.write_rfs(rfs)
        tf.close()

    def _update_index(self):
        if not self._index_has_changed:
            return
        if self._is_remote:
            raise ValueError("Cannot update the index in a remote tar file")
        if self._file is None:
            raise ValueError("File is not open")
        existing_index_json = self.read_file(".tar_index.json")
        new_index_json = json.dumps(self._index, indent=2, sort_keys=True)
        if len(new_index_json) <= len(existing_index_json):
            # we can overwrite the existing index file
            new_index_json = new_index_json.encode() + b" " * (len(existing_index_json) - len(new_index_json))
            self.overwrite_file_content(".tar_index.json", new_index_json)
        else:
            # we must create a new index file
            self.trash_file(".tar_index.json", do_write_index=False)

            # after we trash the file, the index has changed once again
            new_index_json = json.dumps(self._index, indent=2, sort_keys=True)

            new_index_json = _pad_bytes_to_leave_room_for_growth(new_index_json, INITIAL_TAR_INDEX_JSON_SIZE)
            self.write_file(".tar_index.json", new_index_json)

            # now we need to update the entry file
            # tar_index_info = self.get_file_info(".tar_index.json")
            # if tar_index_info is None:
            #     raise ValueError("tar_index_info is None")
            # new_entry_json = json.dumps({
            #     'index': {
            #         'n': tar_index_info.name,
            #         'o': tar_index_info.offset,
            #         'd': tar_index_info.offset_data,
            #         's': tar_index_info.size
            #     }
            # }, indent=2, sort_keys=True)
            tar_index_info = next(file for file in self._index['files'] if file['n'] == ".tar_index.json")
            new_entry_json = json.dumps({
                'index': {
                    'n': tar_index_info['n'],
                    'o': tar_index_info['o'],
                    'd': tar_index_info['d'],
                    's': tar_index_info['s']
                }
            }, indent=2, sort_keys=True)
            new_entry_json = new_entry_json.encode() + b" " * (TAR_ENTRY_JSON_SIZE - len(new_entry_json))
            with open(self._tar_path_or_url, "r+b") as f:
                # we assume the first file is the entry file, and we assume the header is 512 bytes
                # this is to avoid calling the potentially expensive getmember() method
                f.seek(512)
                f.write(new_entry_json)
        self._file.flush()
        self._index_has_changed = False


def _download_file_byte_range(url: str, start: int, end: int) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Range": f"bytes={start}-{end - 1}"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return response.read()


def _load_bytes_from_local_or_remote_file(path_or_url: str, start: int, end: int) -> bytes:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return _download_file_byte_range(path_or_url, start, end)
    else:
        with open(path_or_url, "rb") as f:
            f.seek(start)
            return f.read(end - start)


def _pad_bytes_to_leave_room_for_growth(x: str, initial_size: int) -> bytes:
    total_size = initial_size
    while total_size < len(x) * 4:
        total_size *= 2
    padding = b" " * (total_size - len(x))
    return x.encode() + padding


def _fix_checksum_in_header(f, header_start_byte):
    f.seek(header_start_byte)
    header = f.read(512)

    # From https://en.wikipedia.org/wiki/Tar_(computing)
    # The checksum is calculated by taking the sum of the unsigned byte values
    # of the header record with the eight checksum bytes taken to be ASCII
    # spaces (decimal value 32). It is stored as a six digit octal number with
    # leading zeroes followed by a NUL and then a space. Various implementations
    # do not adhere to this format. In addition, some historic tar
    # implementations treated bytes as signed. Implementations typically
    # calculate the checksum both ways, and treat it as good if either the
    # signed or unsigned sum matches the included checksum.

    header_byte_list = []
    for byte in header:
        header_byte_list.append(byte)
    for i in range(148, 156):
        header_byte_list[i] = 32
    sum = 0
    for byte in header_byte_list:
        sum += byte
    checksum = oct(sum).encode()[2:]
    while len(checksum) < 6:
        checksum = b"0" + checksum
    checksum += b"\x00 "
    f.seek(header_start_byte + 148)
    f.write(checksum)


def _create_random_string():
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10))