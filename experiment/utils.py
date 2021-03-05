import os
import itertools

def find_next_free_file(prefix, suffix, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    while True:
        for i in itertools.count():
            path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
            if not os.path.isfile(path):
                break
        # Create the file to avoid a race condition.
        # Will give an error if the file already exists.
        try:
            f = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(f)
        except FileExistsError as e:
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
        for i in itertools.count():
            directory=os.path.join(root_directory, template % i)
            if not os.path.isdir(directory):
                break
        # Create the directory to avoid a race condition.
        # Will give an error if the directory already exists.
        try:
            os.makedirs(directory, exist_ok=False)
        except FileExistsError as e:
            # Trying to create a directory that already exists.
            # Try a new directory name
            continue
        # Success
        break
    return directory

