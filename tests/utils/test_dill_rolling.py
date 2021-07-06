import os
from typing import Mapping

import pytest
import torch
import dill

from experiment.utils import dill_dump_rolling, dill_load_rolling, dump, load

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
        f.seek(0)
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

    obj1 = {'a': 1, 'b': 2        }
    obj2 = {'a': 1, 'b': 2, 'c': 3}
    obj3 = {        'b': 2, 'c': 3}
    obj4 = {'a': 1,         'c': 3}
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

@pytest.mark.parametrize('obj',[
    1,
    'b',
    'asdf',
    r'asdf',
    b'123',
    [],
    [1,2,3],
    [1,2,'foo'],
    {},
    {'a': 1},
    {'a': [1,'foo'], 'b': {'c': 5, 'd': 'asdf'}},
])
def test_dump_primitives(obj, tmpdir):
    directory = tmpdir.mkdir('output')
    filename = os.path.join(directory,'file')
    with open(filename,'wb') as f:
        dump(obj,f)
    with open(filename,'rb') as f:
        loaded_obj = load(f)
    assert loaded_obj == obj

def test_dump_torch_in_dict(tmpdir):
    directory = tmpdir.mkdir('output')
    filename = os.path.join(directory,'file')
    class Network(torch.nn.Module):
        def __init__(self, num_actions):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=4,out_channels=16,kernel_size=8,stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=16,out_channels=16,kernel_size=4,stride=2),
                torch.nn.ReLU()
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=16*9*9,out_features=128),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=128,out_features=num_actions),
            )
        def forward(self, obs):
            x = obs
            x = self.conv(x)
            x = x.view(-1,9*9*16)
            x = self.fc(x)
            return x
    net = Network(5)
    optim = torch.optim.Adam(net.parameters(),lr=1e-3)
    with open(filename,'wb') as f:
        dump({'net': net, 'optim': optim},f)
    with open(filename,'rb') as f:
        loaded_obj = load(f)

    assert isinstance(loaded_obj,Mapping)

    original_state_dict = net.state_dict()
    loaded_state_dict = loaded_obj['net']
    assert loaded_state_dict.keys() == original_state_dict.keys()
    for k in loaded_state_dict.keys():
        assert (loaded_state_dict[k] == original_state_dict[k]).all()

    original_state_dict = optim.state_dict()
    loaded_state_dict = loaded_obj['optim']
    assert loaded_state_dict.keys() == original_state_dict.keys()
    for k in loaded_state_dict.keys():
        assert loaded_state_dict[k] == original_state_dict[k]

def test_dump_torch_in_list(tmpdir):
    directory = tmpdir.mkdir('output')
    filename = os.path.join(directory,'file')
    class Network(torch.nn.Module):
        def __init__(self, num_actions):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=4,out_channels=16,kernel_size=8,stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=16,out_channels=16,kernel_size=4,stride=2),
                torch.nn.ReLU()
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=16*9*9,out_features=128),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=128,out_features=num_actions),
            )
        def forward(self, obs):
            x = obs
            x = self.conv(x)
            x = x.view(-1,9*9*16)
            x = self.fc(x)
            return x
    net = Network(5)
    optim = torch.optim.Adam(net.parameters(),lr=1e-3)
    with open(filename,'wb') as f:
        dump([net, optim],f)
    with open(filename,'rb') as f:
        loaded_obj = load(f)

    assert isinstance(loaded_obj,list)

    original_state_dict = net.state_dict()
    loaded_state_dict = loaded_obj[0]
    assert loaded_state_dict.keys() == original_state_dict.keys()
    for k in loaded_state_dict.keys():
        assert (loaded_state_dict[k] == original_state_dict[k]).all()

    original_state_dict = optim.state_dict()
    loaded_state_dict = loaded_obj[1]
    assert loaded_state_dict.keys() == original_state_dict.keys()
    for k in loaded_state_dict.keys():
        assert loaded_state_dict[k] == original_state_dict[k]

@pytest.mark.parametrize('data',[
    [['invalid magic number',1]],
    [['d36f3c96-e6f8-417c-be78-26275b53ccad',-1]],
    [['d36f3c96-e6f8-417c-be78-26275b53ccad',1],'foo'],
    [['d36f3c96-e6f8-417c-be78-26275b53ccad',1],{'type': 'invalid type'}],
])
def test_load_invalid_formats(data,tmpdir):
    directory = tmpdir.mkdir('output')
    filename = os.path.join(directory,'file')

    with open(filename,'wb') as f:
        for x in data:
            dill.dump(x,f)
    with open(filename,'rb') as f:
        with pytest.raises(Exception):
            load(f)
