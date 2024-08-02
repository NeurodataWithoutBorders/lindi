import json
import tarfile
import random
import io
import urllib.request


TAR_ENTRY_JSON_SIZE = 1024
INITIAL_TAR_INDEX_JSON_SIZE = 1024 * 256
INITIAL_LINDI_JSON_SIZE = 1024 * 256


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

    def get_file_info(self, file_name: str):
        for file in self._index['files']:
            if file['n'] == file_name:
                return file
        return None

    def overwrite_file_content(self, file_name: str, data: bytes):
        if self._is_remote:
            raise ValueError("Cannot overwrite file content in a remote tar file")
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found")
        if info['s'] != len(data):
            raise ValueError("Unexpected problem in overwrite_file_content(): data size must match the size of the existing file")
        with open(self._tar_path_or_url, "r+b") as f:
            f.seek(info['d'])
            f.write(data)

    def trash_file(self, file_name: str, do_write_index=True):
        if self._is_remote:
            raise ValueError("Cannot trash a file in a remote tar file")
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found")
        zeros = b"-" * info['s']
        with open(self._tar_path_or_url, "r+b") as f:
            f.seek(info['d'])
            f.write(zeros)
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

    def _change_name_of_file(self, file_name: str, new_file_name: str, do_write_index=True):
        if self._is_remote:
            raise ValueError("Cannot change the name of a file in a remote tar file")
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found")
        header_start_byte = info['o']
        file_name_byte_range = (header_start_byte + 0, header_start_byte + 100)
        file_name_prefix_byte_range = (header_start_byte + 345, header_start_byte + 345 + 155)
        with open(self._tar_path_or_url, "r+b") as f:
            f.seek(file_name_byte_range[0])
            f.write(new_file_name.encode())
            # set the rest of the field to zeros
            f.write(b"\0" * (file_name_byte_range[1] - file_name_byte_range[0] - len(new_file_name)))

            f.seek(file_name_prefix_byte_range[0])
            f.write(b"\0" * (file_name_prefix_byte_range[1] - file_name_prefix_byte_range[0]))

            _fix_checksum_in_header(f, header_start_byte)
        try:
            file_in_index = next(file for file in self._index['files'] if file['n'] == file_name)
        except StopIteration:
            raise ValueError(f"File {file_name} not found in index")
        file_in_index['n'] = new_file_name
        if do_write_index:
            self._update_index()

    def write_file(self, file_name: str, data: bytes):
        if self._is_remote:
            raise ValueError("Cannot write a file in a remote tar file")
        with tarfile.open(self._tar_path_or_url, "a") as tar:
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(data)
            fileobj = io.BytesIO(data)
            tar.addfile(tarinfo, fileobj)
        with tarfile.open(self._tar_path_or_url, "r") as tar:
            # TODO: do not call getmember here, because it may be slow instead
            # parse the header of the new file directly and get the offset from
            # there
            info = tar.getmember(file_name)
            self._index['files'].append({
                'n': file_name,
                'o': info.offset,
                'd': info.offset_data,
                's': info.size
            })
        self._update_index()

    def read_file(self, file_name: str) -> bytes:
        info = self.get_file_info(file_name)
        if info is None:
            raise FileNotFoundError(f"File {file_name} not found")
        start_byte = info['d']
        size = info['s']
        return _load_bytes_from_local_or_remote_file(self._tar_path_or_url, start_byte, start_byte + size)

    @staticmethod
    def create(fname: str):
        with tarfile.open(fname, "w") as tar:
            # write the initial entry file this MUST be the first file in the
            # tar file
            tar_entry_json_name = ".tar_entry.json"
            tarinfo = tarfile.TarInfo(name=tar_entry_json_name)
            tarinfo.size = TAR_ENTRY_JSON_SIZE
            tar.addfile(tarinfo, io.BytesIO(b" " * TAR_ENTRY_JSON_SIZE))

            # write the initial index file this will start as the second file in
            # the tar file but as it grows it will be replaced. Importantly, the
            # entry will always be the first file.
            tar_index_json_size = INITIAL_TAR_INDEX_JSON_SIZE
            tar_index_json_name = ".tar_index.json"
            tarinfo = tarfile.TarInfo(name=tar_index_json_name)
            tarinfo.size = tar_index_json_size
            tar.addfile(tarinfo, io.BytesIO(b" " * tar_index_json_size))

        # It seems that we need to close and then open it again in order
        # to get the correct data offsets for the files.
        with tarfile.open(fname, "r") as tar:
            tar_entry_json_info = tar.getmember(tar_entry_json_name)
            tar_index_json_info = tar.getmember(tar_index_json_name)

            # fill the entry file
            initial_entry_json = json.dumps({
                'index': {
                    'n': tar_index_json_name,
                    'o': tar_index_json_info.offset,
                    'd': tar_index_json_info.offset_data,
                    's': tar_index_json_info.size
                }
            }, indent=2, sort_keys=True)
            initial_entry_json = initial_entry_json.encode() + b" " * (tar_entry_json_info.size - len(initial_entry_json))
            with open(fname, "r+b") as f:
                f.seek(tar_entry_json_info.offset_data)
                f.write(initial_entry_json)

            # fill the index file
            initial_index_json = json.dumps({
                'files': [
                    {
                        'n': info.name,
                        'o': info.offset,
                        'd': info.offset_data,
                        's': info.size
                    }
                    for info in [tar_entry_json_info, tar_index_json_info]
                ]
            }, indent=2, sort_keys=True)
            initial_index_json = initial_index_json.encode() + b" " * (tar_index_json_size - len(initial_index_json))
            with open(fname, "r+b") as f:
                f.seek(tar_index_json_info.offset_data)
                f.write(initial_index_json)

    def _update_index(self):
        if self._is_remote:
            raise ValueError("Cannot update the index in a remote tar file")
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
            tar_index_info = self.get_file_info(".tar_index.json")
            if tar_index_info is None:
                raise ValueError("tar_index_info is None")
            new_entry_json = json.dumps({
                'index': {
                    'n': tar_index_info.name,
                    'o': tar_index_info.offset,
                    'd': tar_index_info.offset_data,
                    's': tar_index_info.size
                }
            }, indent=2, sort_keys=True)
            new_entry_json = new_entry_json.encode() + b" " * (TAR_ENTRY_JSON_SIZE - len(new_entry_json))
            with open(self._tar_path_or_url, "r+b") as f:
                # we assume the first file is the entry file, and we assume the header is 512 bytes
                # this is to avoid calling the potentially expensive getmember() method
                f.seek(512)
                f.write(new_entry_json)


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
    checksum += b"\0 "
    f.seek(header_start_byte + 148)
    f.write(checksum)


def _create_random_string():
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10))
