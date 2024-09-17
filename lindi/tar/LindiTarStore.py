import numpy as np
from zarr.storage import Store as ZarrStore
from ..LindiH5pyFile.LindiReferenceFileSystemStore import LindiReferenceFileSystemStore
from .lindi_tar import LindiTarFile


class LindiTarStore(ZarrStore):
    def __init__(self, *, base_store: LindiReferenceFileSystemStore, tar_file: LindiTarFile):
        self._base_store = base_store
        self._tar_file = tar_file

    def __getitem__(self, key: str):
        return self._base_store.__getitem__(key)

    def __setitem__(self, key: str, value: bytes):
        self.setitems({key: value})

    def setitems(self, items_dict: dict):
        for key, value in items_dict.items():
            key_parts = key.split("/")
            key_base_name = key_parts[-1]

            files_to_write_to_tar = {}

            if key_base_name.startswith('.') or key_base_name.endswith('.json'):  # always inline .zattrs, .zgroup, .zarray, zarr.json
                inline = True
            else:
                # presumably it is a chunk of an array
                if isinstance(value, np.ndarray):
                    value = value.tobytes()
                if not isinstance(value, bytes):
                    print(f"key: {key}, value type: {type(value)}")
                    raise ValueError("Value must be bytes")
                size = len(value)
                inline = size < 1000  # this should be a configurable threshold
            if inline:
                # If inline, save in memory
                return self._base_store.__setitem__(key, value)
            else:
                # If not inline, save it as a new file in the tar file
                key_without_initial_slash = key if not key.startswith("/") else key[1:]
                fname_in_tar = f'blobs/{key_without_initial_slash}'
                if self._tar_file.has_file_with_name(fname_in_tar):
                    v = 2
                    while self._tar_file.has_file_with_name(f'{fname_in_tar}.v{v}'):
                        v += 1
                    fname_in_tar = f'{fname_in_tar}.v{v}'
                files_to_write_to_tar[fname_in_tar] = value

                self._set_ref_reference(key_without_initial_slash, f'./{fname_in_tar}', 0, len(value))

            self._tar_file.write_files(files_to_write_to_tar)

    def __delitem__(self, key: str):
        # We don't actually delete the file from the tar, but maybe it would be
        # smart to put it in .trash in the future
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
