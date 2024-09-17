import tempfile
import tarfile
import io


test_file_name = "text/file.txt"
test_file_size = 8100


def create_tar_header_using_tarfile(file_name: str, file_size: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_fname = f"{tmpdirname}/test.tar"
        with tarfile.open(tmp_fname, "w") as tar:
            tarinfo = tarfile.TarInfo(file_name)
            tarinfo.size = file_size
            fobj = io.BytesIO(b"0" * file_size)
            tar.addfile(tarinfo, fileobj=fobj)
            tar.close()
        with open(tmp_fname, "rb") as f:
            header = f.read(512)
            return header


def practice_form_header(file_name: str, file_size: int, header: bytes):
    h = b''
    # We use USTAR format only

    # file name
    a = header[0:100]
    b = file_name.encode() + b"\x00" * (100 - len(file_name))
    assert a == b
    h += b

    # file mode
    a = header[100:108]
    b = b"0000644\x00"  # 644 is the default permission - you can read and write, but others can only read
    assert a == b
    h += b

    # uid
    a = header[108:116]
    b = b"0000000\x00"  # 0 is the default user id
    assert a == b
    h += b

    # gid
    a = header[116:124]
    b = b"0000000\x00"  # 0 is the default group id
    assert a == b
    h += b

    # size
    a = header[124:136]
    # we need an octal representation of the size
    b = f"{file_size:011o}".encode() + b"\x00"  # 11 octal digits
    assert a == b
    h += b

    # mtime
    a = header[136:148]
    b = b"00000000000\x00"  # 0 is the default modification time
    assert a == b
    h += b

    # chksum
    # We'll determine the checksum after creating the full header
    h += b" " * 8  # 8 spaces for now

    # typeflag
    a = header[156:157]
    b = b"0"  # default typeflag is 0 representing a regular file
    assert a == b
    h += b

    # linkname
    a = header[157:257]
    b = b"\x00" * 100  # no link name
    assert a == b
    h += b

    # magic
    a = header[257:263]
    b = b"ustar\x00"  # specifies the ustar format
    assert a == b
    h += b

    # version
    a = header[263:265]
    b = b"00"  # ustar version
    assert a == b
    h += b

    # uname
    a = header[265:297]
    b = b"\x00" * 32  # no user name
    assert a == b
    h += b

    # gname
    a = header[297:329]
    b = b"\x00" * 32  # no group name
    assert a == b
    h += b

    # devmajor
    a = header[329:337]
    b = b"\x00" * 8  # no device major number
    assert a == b
    h += b

    # devminor
    a = header[337:345]
    b = b"\x00" * 8  # no device minor number
    assert a == b
    h += b

    # prefix
    a = header[345:500]
    b = b"\x00" * 155  # no prefix
    assert a == b
    h += b

    # padding
    a = header[500:]
    b = b"\x00" * 12  # padding
    assert a == b
    h += b

    # Now we calculate the checksum
    chksum = _compute_checksum_for_header(h)
    h = h[:148] + chksum + h[156:]

    assert h == header


def create_tar_header(file_name: str, file_size: int) -> bytes:
    # We use USTAR format only
    h = b''

    # file name
    a = file_name.encode() + b"\x00" * (100 - len(file_name))
    h += a

    # file mode
    a = b"0000644\x00"  # 644 is the default permission - you can read and write, but others can only read
    h += a

    # uid
    a = b"0000000\x00"  # 0 is the default user id
    h += a

    # gid
    a = b"0000000\x00"  # 0 is the default group id
    h += a

    # size
    # we need an octal representation of the size
    a = f"{file_size:011o}".encode() + b"\x00"  # 11 octal digits
    h += a

    # mtime
    a = b"00000000000\x00"  # 0 is the default modification time
    h += a

    # chksum
    # We'll determine the checksum after creating the full header
    a = b" " * 8  # 8 spaces for now
    h += a

    # typeflag
    a = b"0"  # default typeflag is 0 representing a regular file
    h += a

    # linkname
    a = b"\x00" * 100  # no link name
    h += a

    # magic
    a = b"ustar\x00"  # specifies the ustar format
    h += a

    # version
    a = b"00"  # ustar version
    h += a

    # uname
    a = b"\x00" * 32  # no user name
    h += a

    # gname
    a = b"\x00" * 32  # no group name
    h += a

    # devmajor
    a = b"\x00" * 8  # no device major number
    h += a

    # devminor
    a = b"\x00" * 8  # no device minor number
    h += a

    # prefix
    a = b"\x00" * 155  # no prefix
    h += a

    # padding
    a = b"\x00" * 12  # padding
    h += a

    # Now we calculate the checksum
    chksum = _compute_checksum_for_header(h)
    h = h[:148] + chksum + h[156:]

    assert len(h) == 512

    return h


def _compute_checksum_for_header(header: bytes) -> bytes:
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
    return checksum


def main():
    header1 = create_tar_header_using_tarfile(test_file_name, test_file_size)
    practice_form_header(test_file_name, test_file_size, header1)
    header2 = create_tar_header(test_file_name, test_file_size)

    assert header1 == header2

    print("Success!")


if __name__ == "__main__":
    main()
