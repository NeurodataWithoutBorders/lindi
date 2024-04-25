from typing import Literal
import os
import h5py
from h5py import File as OriginalH5pyFile
from ..LindiH5pyFile.LindiH5pyFile import LindiH5pyFile
from ..LindiStagingStore.StagingArea import StagingArea
from ..LocalCache.LocalCache import LocalCache


# We need to use this metaclass so that isinstance(f, h5py.File) works when f is
# a LindiH5pyFile or OriginalH5pyFile
class InstanceCheckMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, OriginalH5pyFile) or isinstance(instance, LindiH5pyFile)


class File(metaclass=InstanceCheckMeta):
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
            return OriginalH5pyFile(name, mode=mode, **kwds)


def apply_h5py_patch():
    h5py.File = File


# by virtue of importing we apply the patch
apply_h5py_patch()
