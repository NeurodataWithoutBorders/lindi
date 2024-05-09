from typing import Union
import os
import random
import string
import datetime
import shutil


class StagingArea:
    """
    A staging area where files can be stored temporarily before being
    consolidated and uploaded to a storage system.

    This class is a context manager, so it can be used in a `with` statement to
    ensure that the staging area is cleaned up when it is no longer needed.
    """
    def __init__(self, *, _directory: str) -> None:
        """
        Do not call this constructor directly. Instead, use the `create` method
        to create a new staging area.
        """
        self._directory = os.path.abspath(_directory)

    @staticmethod
    def create(*, base_dir: Union[str, None] = None, dir: Union[str, None] = None) -> 'StagingArea':
        """
        Create a new staging area. Provide either `base_dir` or `dir`, but not
        both.

        Parameters
        ----------
        base_dir : str or None
            If provided, the base directory where the staging area will be
            created. The staging directory will be a subdirectory of this
            directory.
        dir : str or None
            If provided, the exact directory where the staging area will be
            created. It is okay if this directory already exists.
        """
        if base_dir is not None and dir is not None:
            raise ValueError("Provide either base_dir or dir, but not both")
        if base_dir is not None:
            dir = os.path.join(base_dir, _create_random_id())
        if dir is None:
            raise ValueError("Provide either base_dir or dir")
        return StagingArea(_directory=dir)

    def cleanup(self) -> None:
        """
        Clean up the staging area, deleting all files in it. This method is
        called automatically when the staging area is used as a context manager
        in a `with` statement.
        """
        if os.path.exists(self._directory):
            shutil.rmtree(self._directory)

    def __enter__(self) -> 'StagingArea':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()

    @property
    def directory(self) -> str:
        """
        The directory where the files are stored.
        """
        return self._directory

    def store_file(self, relpath: str, value: bytes) -> str:
        """
        Store a file in the staging area.

        Parameters
        ----------
        relpath : str
            The relative path to the file, relative to the staging area root.
        value : bytes
            The contents of the file.
        """
        path = os.path.join(self._directory, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(value)
        return path

    def get_full_path(self, relpath: str) -> str:
        """
        Get the full path to a file in the staging area.

        Parameters
        ----------
        relpath : str
            The relative path to the file, relative to the staging area root.
        """
        return os.path.join(self._directory, relpath)


def _create_random_id():
    # This is going to be a timestamp suitable for alphabetical chronological order plus a random string
    return f"{_timestamp_str()}-{_random_str(8)}"


def _timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def _random_str(n):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))
