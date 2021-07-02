import os
import itertools
from typing import List, Any

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
    if os.path.isfile(filenames[0]):
        for src,dst in zip(reversed(filenames[:-1]),reversed(filenames)):
            if os.path.isfile(src):
                os.rename(src=src,dst=dst)
    with open(filenames[0],'wb') as f:
        dill.dump(obj,f)

def dill_load_rolling(filenames : List[str]):
    """
    Load the most recent file from the list of filenames. If loading fails, then try the next file until one of them works.
    """
    for filename in filenames:
        try:
            with open(filename,'rb') as f:
                return dill.load(f)
        except:
            continue
    raise Exception('Unable to find a valid file to load.')
