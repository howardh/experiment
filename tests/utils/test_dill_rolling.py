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
    """ Check that if the first file is corrupt, the second file will be loaded, and the first file is deleted. """
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

    assert not os.path.isfile(os.path.join(directory,'file1'))
    assert os.path.isfile(os.path.join(directory,'file2'))
    assert not os.path.isfile(os.path.join(directory,'file3'))

def test_rolling_save_fills_missing_slots(tmpdir):
    """ Check that if a file is missing, the slot gets filled with the new save without the following files rolling down.

    e.g. If the file names used are "file1", "file2", "file3", and we have "file1" and "file3", then "file1" will be moved to "file2", but "file3" doesn't change.
    """
    directory = tmpdir.mkdir('output')
    filenames = [os.path.join(directory,fn) for fn in ['file1','file2','file3','file4']]

    assert len(os.listdir(directory)) == 0

    obj1 = {'a': 1, 'b': 2}
    obj2 = {'a': 1, 'b': 2, 'c': 3}
    obj3 = {'b': 2, 'c': 3}
    obj4 = {'a': 1, 'c': 3}
    dill_dump_rolling(obj1, filenames)
    dill_dump_rolling(obj2, filenames)
    dill_dump_rolling(obj3, filenames)

    # filename[0] contains obj3
    # filename[1] contains obj2
    # filename[2] contains obj1

    os.remove(filenames[1])

    # filename[0] contains obj3
    # filename[2] contains obj1

    assert os.path.isfile(os.path.join(directory,'file1'))
    assert not os.path.isfile(os.path.join(directory,'file2'))
    assert os.path.isfile(os.path.join(directory,'file3'))
    assert not os.path.isfile(os.path.join(directory,'file4'))

    dill_dump_rolling(obj4, filenames)

    # filename[0] contains obj4
    # filename[1] contains obj3
    # filename[2] contains obj1

    assert os.path.isfile(os.path.join(directory,'file1'))
    assert os.path.isfile(os.path.join(directory,'file2'))
    assert os.path.isfile(os.path.join(directory,'file3'))
    assert not os.path.isfile(os.path.join(directory,'file4'))

    obj = dill_load_rolling(filenames)
    assert obj == obj4

    os.remove(filenames[0])
    obj = dill_load_rolling(filenames)
    assert obj == obj3

    os.remove(filenames[1])
    obj = dill_load_rolling(filenames)
    assert obj == obj1
