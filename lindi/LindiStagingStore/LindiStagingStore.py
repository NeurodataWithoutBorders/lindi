import os
from zarr.storage import Store as ZarrStore
from ..LindiH5pyFile.LindiReferenceFileSystemStore import LindiReferenceFileSystemStore
from .StagingArea import StagingArea, _random_str


class LindiStagingStore(ZarrStore):
    """
    A Zarr store that allows supplementing a base LindiReferenceFileSystemStore
    where the large data blobs are stored in a staging area. After writing new
    data to the store, the data blobs can be consolidated into larger files and
    then uploaded to a custom storage system, for example DANDI or a cloud
    bucket.
    """
    def __init__(self, *, base_store: LindiReferenceFileSystemStore, staging_area: StagingArea):
        """
        Create a LindiStagingStore.

        Parameters
        ----------
        base_store : LindiReferenceFileSystemStore
            The base store that this store supplements.
        staging_area : StagingArea
            The staging area where large data blobs are stored.
        """
        self._base_store = base_store
        self._staging_area = staging_area

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
            # If not inline, save it as a file in the staging directory
            key_without_initial_slash = key if not key.startswith("/") else key[1:]
            stored_file_path = self._staging_area.store_file(key_without_initial_slash, value)

            self._set_ref_reference(key_without_initial_slash, stored_file_path, 0, len(value))

    def __delitem__(self, key: str):
        # We don't delete the file from the staging directory, because that
        # would be dangerous if the file was part of a consolidated file.
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

    def consolidate_chunks(self):
        """
        Consolidate the chunks in the staging area.
        """
        rfs = self._base_store.rfs
        refs_keys_by_reference_parent_path = {}
        for k, v in rfs['refs'].items():
            if isinstance(v, list) and len(v) == 3:
                url = v[0]
                if not url.startswith(self._staging_area.directory + '/'):
                    continue
                parent_path = os.path.dirname(url)
                if parent_path not in refs_keys_by_reference_parent_path:
                    refs_keys_by_reference_parent_path[parent_path] = []
                refs_keys_by_reference_parent_path[parent_path].append(k)
        for root, dirs, files1 in os.walk(self._staging_area._directory):
            files = [
                f for f in files1
                if not f.startswith('.') and not f.endswith('.json') and not f.startswith('consolidated.')
            ]
            if len(files) <= 1:
                continue
            refs_keys_for_this_dir = refs_keys_by_reference_parent_path.get(root, [])
            if len(refs_keys_for_this_dir) <= 1:
                continue

            # sort so that the files are in order 0.0.0, 0.0.1, 0.0.2, ...
            files = _sort_by_chunk_key(files)

            print(f'Consolidating {len(files)} files in {root}')

            offset = 0
            offset_maps = {}
            consolidated_id = _random_str(8)
            consolidated_index = 0
            max_size_of_consolidated_file = 1024 * 1024 * 1024  # 1 GB, a good size for cloud bucket files
            consolidated_fname = f"{root}/consolidated.{consolidated_id}.{consolidated_index}"
            consolidated_f = open(consolidated_fname, "wb")
            try:
                for fname in files:
                    full_fname = f"{root}/{fname}"
                    with open(full_fname, "rb") as f2:
                        consolidated_f.write(f2.read())
                    offset_maps[full_fname] = (consolidated_fname, offset)
                    offset += os.path.getsize(full_fname)
                    if offset > max_size_of_consolidated_file:
                        consolidated_f.close()
                        consolidated_index += 1
                        consolidated_fname = f"{root}/consolidated.{consolidated_id}.{consolidated_index}"
                        consolidated_f = open(consolidated_fname, "wb")
                        offset = 0
            finally:
                consolidated_f.close()
            for key in refs_keys_for_this_dir:
                filename, old_offset, old_size = rfs['refs'][key]
                if filename not in offset_maps:
                    continue
                consolidated_fname, new_offset = offset_maps[filename]
                rfs['refs'][key] = [consolidated_fname, new_offset + old_offset, old_size]
            # remove the old files
            for fname in files:
                os.remove(f"{root}/{fname}")

    def copy_chunks_to_staging_area(self, *, download_remote: bool):
        """
        Copy the chunks in the base store to the staging area. This is done
        in preparation for uploading to a storage system.

        Parameters
        ----------
        download_remote : bool
            If True, download the remote chunks to the staging area. If False,
            just copy the local chunks.
        """
        if download_remote:
            raise NotImplementedError("Downloading remote chunks not yet implemented")
        rfs = self._base_store.rfs
        templates = rfs.get('templates', {})
        for k, v in rfs['refs'].items():
            if isinstance(v, list) and len(v) == 3:
                url = _apply_templates(v[0], templates)
                if url.startswith('http://') or url.startswith('https://'):
                    if download_remote:
                        raise NotImplementedError("Downloading remote chunks not yet implemented")
                    continue
                elif url.startswith(self._staging_area.directory + '/'):
                    # already in the staging area
                    continue
                else:
                    # copy the local file to the staging area
                    path0 = url
                    chunk_data = _read_chunk_data(path0, v[1], v[2])
                    stored_file_path = self._staging_area.store_file(k, chunk_data)
                    self._set_ref_reference(k, stored_file_path, 0, v[2])


def _apply_templates(x: str, templates: dict) -> str:
    if '{{' in x and '}}' in x:
        for key, val in templates.items():
            x = x.replace('{{' + key + '}}', val)
    return x


def _sort_by_chunk_key(files: list) -> list:
    # first verify that all the files have the same number of parts
    num_parts = None
    for fname in files:
        parts = fname.split('.')
        if num_parts is None:
            num_parts = len(parts)
        elif len(parts) != num_parts:
            raise ValueError(f"Files have different numbers of parts: {files}")
    # Verify that all the parts are integers
    for fname in files:
        parts = fname.split('.')
        for p in parts:
            try:
                int(p)
            except ValueError:
                raise ValueError(f"File part is not an integer: {fname}")

    def _chunk_key(fname: str) -> tuple:
        parts = fname.split('.')
        return tuple(int(p) for p in parts)
    return sorted(files, key=_chunk_key)


def _read_chunk_data(filename: str, offset: int, size: int) -> bytes:
    with open(filename, "rb") as f:
        f.seek(offset)
        return f.read(size)
