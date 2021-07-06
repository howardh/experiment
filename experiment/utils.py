import os
import itertools
from typing import List, Any, Mapping, BinaryIO, Iterable
import io

import torch
import dill

def find_next_free_file(prefix, suffix, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    while True:
        path = ''
        for i in itertools.count():
            path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
            if not os.path.isfile(path):
                break
        # Create the file to avoid a race condition.
        # Will give an error if the file already exists.
        try:
            f = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(f)
        except FileExistsError:
            # Trying to create a file that already exists.
            # Try a new file name
            continue
        # Success
        break
    return path

def find_next_free_dir(root_directory, template):
    if not os.path.isdir(root_directory):
        os.makedirs(root_directory, exist_ok=True)
    while True:
        directory = ''
        for i in itertools.count():
            directory=os.path.join(root_directory, template % i)
            if not os.path.isdir(directory):
                break
        # Create the directory to avoid a race condition.
        # Will give an error if the directory already exists.
        try:
            os.makedirs(directory, exist_ok=False)
        except FileExistsError:
            # Trying to create a directory that already exists.
            # Try a new directory name
            continue
        # Success
        break
    return directory

def dill_dump_rolling(obj : Any, filenames : List[str]):
    """
    Save the provided object to file with one of the given filenames in a rolling manner.

    Args:
        filenames: A list of paths, where the first element is the name given to the most recent save.
    """
    rename_queue = []
    if os.path.isfile(filenames[0]):
        for src,dst in zip(reversed(filenames[:-1]),reversed(filenames)):
            if os.path.isfile(src):
                rename_queue.append((src,dst))
            else:
                rename_queue.clear()
    for src,dst in rename_queue:
        os.rename(src=src,dst=dst)
    with open(filenames[0],'wb') as f:
        dill.dump(obj,f)

def dill_load_rolling(filenames : List[str]):
    """
    Load the most recent file from the list of filenames. If loading fails, then try the next file until one of them works.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            continue
        try:
            with open(filename,'rb') as f:
                return dill.load(f)
        except dill.UnpicklingError:
            # Delete corrupted files so they don't overwrite valid saves
            os.remove(filename)
    raise Exception('Unable to find a valid file to load.')

##################################################
# Pickling
##################################################

MAGIC_NUMBER = 'd36f3c96-e6f8-417c-be78-26275b53ccad'

def _can_dump(obj):
    """ Check if the object can be dumped as is, or if it requires extra handling before dumping. """
    if isinstance(obj,torch.nn.Module):
        return False
    if isinstance(obj,torch.optim.Optimizer):
        return False
    if isinstance(obj,Mapping):
        return all((_can_dump(v) for v in obj.values()))
    if isinstance(obj,str): # str is also an Iterable, so catch it before that branch
        return True
    if isinstance(obj,Iterable):
        return all((_can_dump(v) for v in obj))
    return True

def dump(obj : Any, f : BinaryIO):
    """ Pickle objects with torch object handling. """
    VERSION = 1
    def dump_rec(obj : Any, f : BinaryIO):
        if _can_dump(obj):
            dill.dump({'type': 'raw'}, f)
            dill.dump(obj, f)
        elif isinstance(obj,dict):
            keys = list(obj.keys())
            dill.dump({'type': 'dict', 'keys': keys}, f)
            for k in keys:
                dump_rec(obj[k], f)
        elif isinstance(obj,list):
            dill.dump({'type': 'list', 'length': len(obj)}, f)
            for o in obj:
                dump_rec(o, f)
        elif isinstance(obj,torch.nn.Module) or isinstance(obj,torch.optim.Optimizer):
            dill.dump({'type': 'torch_state_dict'}, f)
            buff = io.BytesIO()
            torch.save(obj.state_dict(),buff)
            buff.seek(0)
            byte_str = buff.read()
            dill.dump(byte_str, f)
        else:
            raise NotImplementedError('Unable to handle object of type %s.' % type(obj)) # pragma: no cover
    dill.dump([MAGIC_NUMBER,VERSION], f)
    dump_rec(obj, f)

def load(f : BinaryIO, torch_map_location=None):
    VERSION = 1
    header = dill.load(f)
    if header[0] != MAGIC_NUMBER:
        raise Exception('invalid file format.')
    if header[1] != VERSION:
        raise Exception('Unable to handle file version.')
    def load_rec(f : BinaryIO):
        meta = dill.load(f)
        if not isinstance(meta,Mapping):
            raise Exception('Invalid file format')
        if meta['type'] == 'raw':
            return dill.load(f)
        elif meta['type'] == 'dict':
            keys = meta['keys']
            return {k:load_rec(f) for k in keys}
        elif meta['type'] == 'list':
            length = meta['length']
            return [load_rec(f) for _ in range(length)]
        elif meta['type'] == 'torch_state_dict':
            byte_str = dill.load(f)
            buff = io.BytesIO(byte_str)
            buff.seek(0)
            return torch.load(buff, map_location=torch_map_location)
        else:
            raise Exception('Invalid file format')
    return load_rec(f)
