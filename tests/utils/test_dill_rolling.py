import os

from experiment.utils import dill_dump_rolling, dill_load_rolling

def test_rolling_save(tmpdir):
    directory = tmpdir.mkdir('output')
    filenames = [os.path.join(directory,fn) for fn in ['file1','file2','file3']]

    assert len(os.listdir(directory)) == 0

    obj = {'a': 1, 'b': 2}
    dill_dump_rolling(obj, filenames)
    assert len(os.listdir(directory)) == 1

    obj2 = dill_load_rolling(filenames)
    assert obj == obj2

    dill_dump_rolling(obj, filenames)
    assert len(os.listdir(directory)) == 2

    obj2 = dill_load_rolling(filenames)
    assert obj == obj2

    dill_dump_rolling(obj, filenames)
    assert len(os.listdir(directory)) == 3

    obj2 = dill_load_rolling(filenames)
    assert obj == obj2

    dill_dump_rolling(obj, filenames)
    assert len(os.listdir(directory)) == 3

    obj2 = dill_load_rolling(filenames)
    assert obj == obj2

def test_rolling_save_missing_first_file(tmpdir):
    directory = tmpdir.mkdir('output')
    filenames = [os.path.join(directory,fn) for fn in ['file1','file2','file3']]

    assert len(os.listdir(directory)) == 0

    obj1 = {'a': 1, 'b': 2}
    dill_dump_rolling(obj1, filenames)
    obj2 = {'a': 1, 'b': 2, 'c': 3}
    dill_dump_rolling(obj2, filenames)

    os.remove(filenames[0])
    obj = dill_load_rolling(filenames)
    assert obj == obj1

def test_rolling_save_corrupt_first_file(tmpdir):
    directory = tmpdir.mkdir('output')
    filenames = [os.path.join(directory,fn) for fn in ['file1','file2','file3']]

    assert len(os.listdir(directory)) == 0

    obj1 = {'a': 1, 'b': 2}
    dill_dump_rolling(obj1, filenames)
    obj2 = {'a': 1, 'b': 2, 'c': 3}
    dill_dump_rolling(obj2, filenames)

    with open(filenames[0], "r+b") as f:
        f.seek(5)
        f.write(b'asdf')
    obj = dill_load_rolling(filenames)
    assert obj == obj1
