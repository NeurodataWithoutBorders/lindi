import numpy as np
import lindi


def write_lindi_binary():
    with lindi.LindiH5pyFile.from_lindi_file('test.lindi', mode='w') as f:
        f.attrs['test'] = 42
        ds = f.create_dataset('data', shape=(1000, 1000), dtype='f4')
        ds[...] = np.random.rand(1000, 1000)


def test_read():
    f = lindi.LindiH5pyFile.from_lindi_file('test.lindi', mode='r')
    print(f.attrs['test'])
    print(f['data'][0, 0])
    f.close()


if __name__ == "__main__":
    write_lindi_binary()
    test_read()
