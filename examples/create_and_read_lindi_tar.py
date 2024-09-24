import numpy as np
import lindi

# Create a new lindi binary file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi.tar', mode='w') as f:
    f.attrs['attr1'] = 'value1'
    f.attrs['attr2'] = 7
    ds = f.create_dataset('dataset1', shape=(1000, 1000), dtype='f')
    ds[...] = np.random.rand(1000, 1000)

# Later read the file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi.tar', mode='r') as f:
    print(f.attrs['attr1'])
    print(f.attrs['attr2'])
    print(f['dataset1'][...])
