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
