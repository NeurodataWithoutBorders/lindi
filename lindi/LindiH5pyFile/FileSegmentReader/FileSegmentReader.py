import remfile


class FileSegmentReader:
    """
    A class that reads a segment of a file from a URL or a local path.
    """
    def __init__(self, url: str):
        """
        Create a new FileSegmentReader.

        Parameters
        ----------
        url : str
            The URL of the file to read, or a local path.
        """
        self.url = url
        self.remfile = None
        self.local_path = None
        if url.startswith("http://") or url.startswith("https://"):
            # remfile does not need to be closed
            self.remfile = remfile.File(url)
        else:
            self.local_path = url

    def read(self, offset: int, length: int):
        if self.remfile is not None:
            self.remfile.seek(offset)
            return self.remfile.read(length)
        elif self.local_path is not None:
            with open(self.local_path, "rb") as f:
                f.seek(offset)
                return f.read(length)
        else:
            raise Exception("Unexpected: no remfile or local_path")
