import os
from zarr.storage import Store as ZarrStore
from ..LindiH5pyFile.LindiReferenceFileSystemStore import LindiReferenceFileSystemStore


class Lindi1Store(ZarrStore):
    """
    A Zarr store that allows supplementing a base LindiReferenceFileSystemStore
    where the large data blobs are appended to a lindi file of lindi1 format.
    """
    def __init__(self, *, base_store: LindiReferenceFileSystemStore, lindi1_file_name: str):
        """
        Create a LindiStagingStore.

        Parameters
        ----------
        base_store : LindiReferenceFileSystemStore
            The base store that this store supplements.
        lindi1_file_name : str
            The name of the lindi1 file that will be created or appended to.
        """
        self._base_store = base_store
        self._lindi1_file_name = lindi1_file_name

    def __getitem__(self, key: str):
        return self._base_store.__getitem__(key)

    def __setitem__(self, key: str, value: bytes):
        key_parts = key.split("/")
        key_base_name = key_parts[-1]
        if key_base_name.startswith('.') or key_base_name.endswith('.json'):  # always inline .zattrs, .zgroup, .zarray, zarr.json
            inline = True
        else:
            # presumably it is a chunk of an array
            if not isinstance(value, bytes):
                raise ValueError("Value must be bytes")
            size = len(value)
            inline = size < 1000  # this should be a configurable threshold
        if inline:
            # If inline, save in memory
            return self._base_store.__setitem__(key, value)
        else:
            # If not inline, append it to the lindi1 file
            key_without_initial_slash = key if not key.startswith("/") else key[1:]
            lindi1_file_size = os.path.getsize(self._lindi1_file_name)
            with open(self._lindi1_file_name, "ab") as f:
                f.write(value)
            self._set_ref_reference(key_without_initial_slash, '.', lindi1_file_size, len(value))

    def __delitem__(self, key: str):
        # We can't delete the file from the lindi1 file, but we do need to remove the reference
        return self._base_store.__delitem__(key)

    def __iter__(self):
        return self._base_store.__iter__()

    def __len__(self):
        return self._base_store.__len__()

    # These methods are overridden from BaseStore
    def is_readable(self):
        return True

    def is_writeable(self):
        return True

    def is_listable(self):
        return True

    def is_erasable(self):
        return False

    def _set_ref_reference(self, key: str, filename: str, offset: int, size: int):
        rfs = self._base_store.rfs
        if 'refs' not in rfs:
            # this shouldn't happen, but we'll be defensive
            rfs['refs'] = {}
        rfs['refs'][key] = [
            filename,
            offset,
            size
        ]
