import pytest
import os

from experiment.logger import FileBackedList

def test_empty(tmpdir):
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=3, max_file_length=5)
    assert len(fblist) == 0
    with pytest.raises(IndexError):
        fblist[0]

def test_one_object(tmpdir):
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=3, max_file_length=5)
    fblist.append(1)
    assert len(fblist) == 1
    assert fblist[0] == 1
    assert fblist[-1] == 1

def test_data_in_memory(tmpdir):
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=None)
    for i in range(10):
        fblist.append(i)
    assert len(fblist) == 10
    assert fblist[0] == 0
    assert fblist[-1] == 9
    assert fblist[5] == 5
    assert fblist[-5] == 5
    assert fblist[-6] == 4

def test_modify_data_in_memory(tmpdir):
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=None)
    for i in range(10):
        fblist.append(i)
    assert len(fblist) == 10
    assert fblist[0] == 0
    assert fblist[-1] == 9
    assert fblist[5] == 5
    assert fblist[-5] == 5
    assert fblist[-6] == 4
    fblist[0] = 100
    assert fblist[0] == 100
    fblist[-1] = 200
    assert fblist[-1] == 200
    fblist[5] = 300
    assert fblist[5] == 300
    fblist[-5] = 400
    assert fblist[-5] == 400
    fblist[-6] = 500
    assert fblist[-6] == 500

def test_data_in_file(tmpdir):
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=3)
    for i in range(10):
        fblist.append(i)
    assert len(fblist) == 10
    assert fblist[0] == 0
    assert fblist[-1] == 9
    assert fblist[5] == 5
    assert fblist[-5] == 5
    assert fblist[-6] == 4
    # Check that the file has been written to disk
    assert len(os.listdir(output_dir)) == 1

def test_data_in_multiple_files(tmpdir):
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=3, max_file_length=5)
    for i in range(10):
        fblist.append(i)
    assert len(fblist) == 10
    assert fblist[0] == 0
    assert fblist[-1] == 9
    assert fblist[5] == 5
    assert fblist[-5] == 5
    assert fblist[-6] == 4
    # Check that the files have been written to disk
    assert len(os.listdir(output_dir)) == 2
    assert os.path.exists(os.path.join(output_dir, 'output.pkl.0'))
    assert os.path.exists(os.path.join(output_dir, 'output.pkl.1'))
    # Check that the files are not empty
    assert os.stat(os.path.join(output_dir, 'output.pkl.0')).st_size > 0
    assert os.stat(os.path.join(output_dir, 'output.pkl.1')).st_size > 0

def test_iterator(tmpdir):
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=3, max_file_length=5)
    for i in range(10):
        fblist.append(i)
    output = []
    for i in fblist:
        output.append(i)
    assert output == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def test_iterator_mem_longer_than_file_length(tmpdir):
    """
    Test that the iterator properly obtains all data when the max memory length is longer than the max file length.
    """
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=5, max_file_length=3)
    for i in range(10):
        fblist.append(i)
    output = []
    for i in fblist:
        output.append(i)
    assert output == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def test_indexed(tmpdir):
    output_dir = tmpdir.mkdir('output')
    filename = os.path.join(output_dir, 'output.pkl')
    fblist = FileBackedList(filename, max_memory_length=3, max_file_length=5, index_enabled=True)
    for i in range(10):
        fblist.append(i)
    assert fblist[0] == 0
    assert fblist[-1] == 9
    assert fblist[5] == 5
    assert fblist[-5] == 5
    assert fblist[-6] == 4
