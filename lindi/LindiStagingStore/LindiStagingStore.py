from typing import Callable
import json
import tempfile
import os
from zarr.storage import Store as ZarrStore
from ..LindiH5pyFile.LindiReferenceFileSystemStore import LindiReferenceFileSystemStore
from .StagingArea import StagingArea, _random_str
from ..LindiH5ZarrStore._util import _write_rfs_to_file


# Accepts a string path to a file, uploads (or copies) it somewhere, and returns a string URL
# (or local path)
UploadFileFunc = Callable[[str], str]


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

    def upload(
        self,
        *,
        on_upload_blob: UploadFileFunc,
        on_upload_main: UploadFileFunc,
        consolidate_chunks: bool = True
    ):
        """
        Consolidate the chunks in the staging area, upload them to a storage
        system, updating the references in the base store, and then upload the
        updated reference file system .json file.

        Parameters
        ----------
        on_upload_blob : StoreFileFunc
            A function that takes a string path to a blob file, uploads or copies it
            somewhere, and returns a string URL (or local path).
        on_upload_main : StoreFileFunc
            A function that takes a string path to the main .json file, stores
            it somewhere, and returns a string URL (or local path).
        consolidate_chunks : bool
            If True (the default), consolidate the chunks in the staging area
            before uploading.

        Returns
        -------
        str
            The URL (or local path) of the uploaded reference file system .json
            file.
        """
        if consolidate_chunks:
            self.consolidate_chunks()
        rfs = self._base_store.rfs
        rfs = json.loads(json.dumps(rfs))  # deep copy
        LindiReferenceFileSystemStore.replace_meta_file_contents_with_dicts_in_rfs(rfs)
        blob_mapping = _upload_directory_of_blobs(self._staging_area.directory, on_upload_blob=on_upload_blob)
        for k, v in rfs['refs'].items():
            if isinstance(v, list) and len(v) == 3:
                url1 = v[0]
                if url1.startswith(self._staging_area.directory + '/'):
                    url2 = blob_mapping.get(url1, None)
                    if url2 is None:
                        raise ValueError(f"Could not find url in blob mapping: {url1}")
                    rfs['refs'][k][0] = url2
        with tempfile.TemporaryDirectory() as tmpdir:
            rfs_fname = f"{tmpdir}/rfs.lindi.json"
            LindiReferenceFileSystemStore.use_templates_in_rfs(rfs)
            _write_rfs_to_file(rfs=rfs, output_file_name=rfs_fname)
            return on_upload_main(rfs_fname)

    def consolidate_chunks(self):
        """
        Consolidate the chunks in the staging area.

        This method is called by `upload` if `consolidate_chunks` is True.
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


def _upload_directory_of_blobs(
    staging_dir: str,
    on_upload_blob: UploadFileFunc
) -> dict:
    """
    Upload all the files in a directory to a storage system and return a mapping
    from the original file paths to the URLs of the uploaded files.
    """
    all_files = []
    for root, dirs, files in os.walk(staging_dir):
        for fname in files:
            full_fname = f"{root}/{fname}"
            all_files.append(full_fname)
    blob_mapping = {}
    for i, full_fname in enumerate(all_files):
        relative_fname = full_fname[len(staging_dir):]
        size_bytes = os.path.getsize(full_fname)
        print(f'Uploading blob {i + 1} of {len(all_files)} {relative_fname} ({_format_size_bytes(size_bytes)})')
        blob_url = on_upload_blob(full_fname)
        blob_mapping[full_fname] = blob_url
    return blob_mapping


def _format_size_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.1f} GB"
