import os

from experiment.utils import find_next_free_file, find_next_free_dir

def test_find_next_free_file(tmpdir):
    directory = tmpdir.mkdir('output')
    path = find_next_free_file(prefix='file',suffix='txt',directory=directory)
    assert os.path.isfile(path)

def test_find_next_free_dir(tmpdir):
    directory = tmpdir.mkdir('output')
    path = find_next_free_dir(root_directory=directory,template='foo-%d')
    assert os.path.isdir(path)

def test_find_next_free_file_non_existant_root(tmpdir):
    directory = os.path.join(tmpdir,'output')
    path = find_next_free_file(prefix='file',suffix='txt',directory=directory)
    assert os.path.isfile(path)
