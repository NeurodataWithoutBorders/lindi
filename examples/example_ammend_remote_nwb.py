import numpy as np
import lindi
import pynwb


def example_ammend_remote_nwb():
    url = 'https://api.dandiarchive.org/api/assets/2e6b590a-a2a4-4455-bb9b-45cc3d7d7cc0/download/'
    with lindi.LindiH5pyFile.from_hdf5_file(url) as f:
        f.write_lindi_file('example.nwb.lindi.tar')
    with lindi.LindiH5pyFile.from_lindi_file('example.nwb.lindi.tar', mode='r+') as f:

        # Can't figure out how to modify something using pyNWB
        # with pynwb.NWBHDF5IO(file=f, mode='r+') as io:
        #     nwbfile = io.read()
        #     print(nwbfile)
        #     nwbfile.session_description = 'Modified session description'
        #     io.write(nwbfile)

        f['session_description'][()] = 'new session description'

        # Create something that will become a new file in the tar
        ds = f.create_dataset('new_dataset', data=np.random.rand(10000, 1000), chunks=(1000, 200))
        ds[20, 20] = 42

    with lindi.LindiH5pyFile.from_lindi_file('example.nwb.lindi.tar', mode='r') as f:
        with pynwb.NWBHDF5IO(file=f, mode='r') as io:
            nwbfile = io.read()
            print(nwbfile)
        print(f['new_dataset'][20, 20])


if __name__ == '__main__':
    example_ammend_remote_nwb()
