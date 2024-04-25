from typing import Any
import tempfile
from ..LindiH5pyFile.LindiH5pyFile import LindiH5pyFile
from ..LindiStagingStore.StagingArea import StagingArea


def LindiNWBHDF5IO(
    path: str,
    mode: str = "r",
    load_namespaces: bool = False,
    manager: Any = None,
    on_upload_blob: Any = None
):
    import pynwb  # don't make this a required dependency for the package
    if path.endswith('.nwb') or path.endswith('.h5') or path.endswith('.hdf5'):
        kwargs = dict(
            path=path,
            mode=mode,
            load_namespaces=load_namespaces
        )
        if manager is not None:
            kwargs['manager'] = manager
        return pynwb.NWBHDF5IO(**kwargs)
    elif path.endswith('.lindi.json'):
        with tempfile.TemporaryDirectory() as tmpdir:
            if mode == 'r':
                staging_area = None
                h5f = LindiH5pyFile.from_lindi_file(path)
            elif mode == 'w':
                staging_area = StagingArea.create(tmpdir)
                h5f = LindiH5pyFile.from_reference_file_system(None, mode='r+', staging_area=staging_area)
            elif mode == 'a':
                staging_area = StagingArea.create(tmpdir)
                h5f = LindiH5pyFile.from_reference_file_system(path, mode='r+', staging_area=staging_area)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            kwargs = dict(
                file=h5f,
                mode=mode
            )
            if manager is not None:
                kwargs['manager'] = manager

            class NWBHDF5IOWithCleanupOnExit(pynwb.NWBHDF5IO):
                def __init__(self, *args, on_exit, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.on_exit = on_exit

                def __exit__(self, *args):
                    super().__exit__(*args)
                    self.on_exit()

            def on_exit():
                h5f.close()
                if staging_area is not None:
                    assert h5f.staging_store is not None

                    def on_upload_main(fname: str):
                        import shutil
                        shutil.copy(fname, path)
                        return path

                    if not h5f.staging_store.is_empty:
                        if on_upload_blob is None:
                            raise ValueError("on_upload_blob is required when staging store is not empty")

                    h5f.staging_store.upload(
                        on_upload_blob=on_upload_blob,
                        on_upload_main=on_upload_main,
                        consolidate_chunks=True
                    )

            return NWBHDF5IOWithCleanupOnExit(**kwargs, on_exit=on_exit)
    else:
        raise Exception(f"Unsupported file extension: {path}")
