from typing import Literal
import os
import h5py
from ..LindiH5pyFile.LindiH5pyFile import LindiH5pyFile
from ..LindiStagingStore.StagingArea import StagingArea
from ..LocalCache.LocalCache import LocalCache


class File(h5py.File):
    """
    A drop-in replacement for h5py.File that is either a lindi.LindiH5pyFile or
    h5py.File depending on whether the file name ends with .lindi.json or not.
    """
    def __new__(cls, name, mode: Literal['r', 'r+', 'w', 'w-', 'x', 'a'] = 'r', **kwds):
        if isinstance(name, str) and name.endswith('.lindi.json'):
            # should we raise exceptions on select unsupported kwds? or just go with the flow?
            if mode != 'r':
                staging_area = StagingArea.create(dir=name + '.d')
            else:
                staging_area = None
            local_cache_dir = os.environ.get('LINDI_LOCAL_CACHE_DIR', None)
            if local_cache_dir is not None:
                local_cache = LocalCache(cache_dir=local_cache_dir)
            else:
                local_cache = None

            return LindiH5pyFile.from_lindi_file(
                name,
                mode=mode,
                staging_area=staging_area,
                local_cache=local_cache
            )
        else:
            return h5py.File(name, mode=mode, **kwds)
