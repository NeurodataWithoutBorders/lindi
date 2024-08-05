import lindi

# Create a new lindi.json file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi.json', mode='w') as f:
    f.attrs['attr1'] = 'value1'
    f.attrs['attr2'] = 7
    ds = f.create_dataset('dataset1', shape=(10,), dtype='f')
    ds[...] = 12

# Later read the file
with lindi.LindiH5pyFile.from_lindi_file('example.lindi.json', mode='r') as f:
    print(f.attrs['attr1'])
    print(f.attrs['attr2'])
    print(f['dataset1'][...])