from abc import ABC, abstractmethod
from typing import Union, List
import warnings

import dill
import numpy as np

try:
    import wandb
except:
    pass

class BaseLogger(ABC):
    @abstractmethod
    def __getitem__(self, index): ...
    @abstractmethod
    def __len__(self): ...
    @abstractmethod
    def __iter__(self): ...
    @abstractmethod
    def __reversed__(self): ...
    @abstractmethod
    def log(self,**data): ...
    @abstractmethod
    def append(self,**data): ...

class Logger(BaseLogger):
    def __init__(self,
            key_name : Union[str, List[str]] = None,
            manual_iteration : bool = False,
            overwrite : bool = False,
            allow_implicit_key : bool = False,
            in_memory : bool = True,
            filename : str = None,
            max_file_length : int = None,
            wandb_params : dict = None):
        """
        Args:
            key_name: Name of a logged value that is unique for each iteration.
                Consecutive calls to `Logger.log` with the same value for this key will be logged together.
                If not specified, then each call to `Logger.log` is assumed to be a distinct iteration.
                Can be a list of strings if multiple keys form the iteration's identifier, in which case, at least one of the listed values have to be different between iterations.
            overwrite: If set to `True`, then calls to `Logger.log` can overwrite existing logged data with the same key. If `False`, then an exception will be raised instead.
            allow_implicit_key (bool): If set to `True`, then the key needs only be specified when it changes. If unspecified, then it is assumed to be unchanged. If set to `False`, then the key must always be present.
            in_memory (bool): If set to `True`, then the logged data is stored in memory. If set to `False`, then the logged data is stored on disk.
        """
        self.key_name = key_name
        self.manual_iteration = manual_iteration
        self.overwrite = overwrite
        self.allow_implicit_key = allow_implicit_key

        if not in_memory:
            self.data = FileBackedList(
                    filename=filename,
                    max_memory_length=3,
                    max_file_length=max_file_length)
        else:
            self.data = []
        self.keys = set()

        self.init_wandb(wandb_params)
        self._multival_keys = set()

    def init_wandb(self, wandb_params):
        self._wandb_params = wandb_params

        if wandb_params is not None:
            self._wandb_run = wandb.init(**wandb_params)
        else:
            self._wandb_run = None
    def finish_wandb(self, *args, **kwargs):
        if self._wandb_run is not None:
            self._wandb_run.finish(*args, **kwargs)

    def __getitem__(self, index : Union[str,int,slice]):
        if type(index) is str:
            if index not in self.keys:
                raise KeyError('Key %s not found in logs.' % index)
            if type(self.key_name) is not str and self.key_name is not None:
                raise NotImplementedError('')
            x = []
            y = []
            for i,d in enumerate(self.data):
                if index not in d:
                    continue
                y.append(d[index])
                if self.key_name is None:
                    x.append(i)
                else:
                    x.append(d[self.key_name])
            return x,y
        elif type(index) is int:
            return self.data[index]
        elif type(index) is slice:
            return self.data[index]
        else:
            raise TypeError('Unable to handle index of type %s' % type(index))
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        for x in self.data:
            yield x
    def __reversed__(self):
        return reversed(self.data)

    def __repr__(self):
        return '<%s.%s at %s %s>' % (
                self.__class__.__module__,
                self.__class__.__name__,
                hex(id(self)),
                self.keys,
        )

    def _is_logger_key(self, k):
        if type(self.key_name) is str and k == self.key_name:
            return True
        if type(self.key_name) is list and k in self.key_name:
            return True
        return False

    def _did_key_change(self, data):
        if len(self.data) == 0:
            return True
        if self.key_name is None:
            return True
        last_point = self.data[-1]
        assert last_point is not None
        if type(self.key_name) is str:
            if self.key_name not in data.keys():
                if self.allow_implicit_key:
                    return False
                else:
                    raise Exception('`key_name` is specified as "%s", but this key is not present in the logged data.' % self.key_name)
            assert self.data is not None
            assert self.data[-1] is not None
            return data[self.key_name] != last_point[self.key_name]
        if type(self.key_name) is list:
            for k in self.key_name:
                if k not in data.keys():
                    if self.allow_implicit_key:
                        return False
                    else:
                        raise Exception('`key_name` is specified as "%s", but "%s" is not present in the logged data.' % (self.key_name, k))
                if data[k] != last_point[k]:
                    return True
            return False

    def next_iteration(self):
        if not self.manual_iteration:
            raise Exception('`manual_iteration` has to be set to `True` to use next_iteration.')
        self._on_key_change()

    def log(self,**data):
        if self._did_key_change(data) and not self.manual_iteration:
            self._on_key_change()
        last_point = self.data[-1]
        assert last_point is not None
        for k,v in data.items():
            if k in last_point:
                if not self._is_logger_key(k) and not self.overwrite:
                    raise Exception('Key "%s" already exists for this iteration.' % k)
            last_point[k] = v
            self.keys.add(k)
        # W&B logging
        if self._wandb_run is not None:
            if type(self.key_name) is str:
                if self.key_name in data:
                    key_val = data[self.key_name]
                elif self.allow_implicit_key:
                    key_val = last_point[self.key_name]
                else:
                    raise Exception('Key not found')
                self._wandb_run.log(data, step=key_val)
            else:
                self._wandb_run.log(data)

    def append(self,**data):
        if self._did_key_change(data) and not self.manual_iteration:
            self._on_key_change()
        last_point = self.data[-1]
        assert last_point is not None
        for k,v in data.items():
            if self._is_logger_key(k):
                last_point[k] = v
                continue
            if k in last_point:
                if type(last_point[k]) is not list:
                    if self.overwrite:
                        last_point[k] = []
                    else:
                        raise Exception('Key "%s" has already been assigned (probably via `Logger.log()`). The same key cannot be used in both `log()` and `append()`.' % k)
                last_point[k].append(v)
            else:
                last_point[k] = [v]
            self.keys.add(k)
            self._multival_keys.add(k)
        if self._wandb_run is not None:
            warnings.warn('`append` does not support logging to W&B.')

    def _sync_wandb_multival(self):
        """ Aggregate and sync multivalue data (i.e. data added using `append`) with W&B. """
        if self._wandb_run is None:
            return
        if type(self.key_name) is not str:
            return
        if len(self.data) == 0:
            return
        last_point = self.data[-1]
        assert last_point is not None
        if self.key_name not in last_point:
            return

        key_val = last_point[self.key_name]
        data = {}
        for k in self._multival_keys:
            try:
                data[k] = np.mean(last_point[k])
            except:
                pass
        self._wandb_run.log(data, step=key_val)

    def _on_key_change(self):
        # Make appropriate aggregations for W&B
        self._sync_wandb_multival()
        # New data point
        self.data.append({})

    def mean(self, key) -> float:
        total = 0
        count = 0
        for d in self.data:
            if key in d:
                total += d[key]
                count += 1
        return total/count

    def make_sublogger(self, prefix):
        return SubLogger(self, prefix)

    def state_dict(self):
        output = {
                'data': self.data,
                'manual_iteration': self.manual_iteration,
                'key_name': self.key_name,
                'keys': self.keys,
        }
        if isinstance(self.data, FileBackedList):
            output['data'] = self.data.state_dict()
        if self._wandb_run is not None:
            output['wandb_params'] = self._wandb_params
            output['wandb_run_id'] = self._wandb_run.id
        return output
    def load_state_dict(self, state, include_wandb=True):
        data = state.get('data', [])
        if isinstance(data, dict):
            self.data = FileBackedList(filename=None)
            self.data.load_state_dict(data)
        else:
            self.data = data
        self.key_name = state.get('key_name', None)
        self.manual_iteration = state.get('manual_iteration', False)
        if 'keys' in state:
            self.keys = state['keys']
        else:
            for d in self.data:
                for k in d.keys():
                    self.keys.add(k)
        if include_wandb:
            # We cannot currently resume W&B logging. If the run has logged data between the last checkpoint and when it was killed, that data will still be logged on W&B, and we can't overwrite them.
            pass
            #if 'wandb_params' in state:
            #    wandb_params = state['wandb_params']
            #    self._wandb_run = wandb.init(
            #            project=wandb_params['project'],
            #            id=state['wandb_run_id'],
            #            resume='must')

class SubLogger(BaseLogger):
    """ A logger """
    def __init__(self, parent_logger : Logger, prefix : str):
        if not parent_logger.allow_implicit_key:
            raise Exception('SubLoggers will not work unless the parent logger allows implicit keys. Set `allow_implicit_key` to True.')
        self.parent_logger = parent_logger
        self.prefix = prefix
    def __getitem__(self, index : Union[str,int,slice]):
        if isinstance(index,str):
            return self.parent_logger[f'{self.prefix}{index}']
        return self.parent_logger[index]
    def __len__(self):
        raise NotImplementedError('Not sure how I want this to behave yet.')
    def __iter__(self):
        raise NotImplementedError('Not sure how I want this to behave yet.')
    def __reversed__(self):
        raise NotImplementedError('Not sure how I want this to behave yet.')

    def log(self, **data):
        prefixed_data = {f'{self.prefix}{k}':v for k,v in data.items()}
        self.parent_logger.log(**prefixed_data)
    def append(self, **data):
        prefixed_data = {f'{self.prefix}{k}':v for k,v in data.items()}
        self.parent_logger.append(**prefixed_data)

    def state_dict(self):
        return {
            'prefix': self.prefix,
        }
    def load_state_dict(self, state):
        self.prefix = state['prefix']

class DummyLogger(BaseLogger):
    def __init__(self,*args,**kwargs) -> None:
        args=args
        kwargs=kwargs
        super().__init__()
    def __getitem__(self, index : Union[str,int,slice]):
        index=index
        return None
    def __len__(self):
        return 0
    def __iter__(self):
        raise NotImplementedError('Not sure how I want this to behave yet.')
    def __reversed__(self):
        raise NotImplementedError('Not sure how I want this to behave yet.')
    def log(self,**data):
        data=data
    def append(self, **data):
        data=data
    def state_dict(self):
        return {}
    def load_state_dict(self, state):
        state=state
    def make_sublogger(self, prefix):
        return DummyLogger(prefix)

class FileBackedList:
    """
    A list whose data is stored on disk, and the metadata in memory. This is designed to be used with `Logger`.
    """
    def __init__(self, filename, max_memory_length=None, max_file_length=None, index_enabled=False):
        self._base_filename = filename
        self._max_memory_length = max_memory_length
        self._max_file_length = max_file_length

        self._files = []
        self._file_handle = None
        self._length = 0
        self._data = []
        self._index_enabled = index_enabled # If True, then keep track the position of each data point
        self._indices = [] # Position of each data point in their respective files
        self.iterate_past_end = False # If True, then iterate past the end of the list as described by the metadata. This is useful if you're loading the metadata from a checkpoint on a running experiment and you want to see the latest data before the next checkpoint is saved. Note that this might miss some data that is saved in memory.
    def _new_file(self):
        if self._base_filename is None:
            raise Exception('Cannot create new file without a base filename.')
        index = len(self._files)
        self._files.append({
            'filename': f'{self._base_filename}.{index}',
            'num_points': 0,
        })
        if self._file_handle is not None:
            self._file_handle.close()
        self._file_handle = open(self._files[-1]['filename'],'wb')
    def __len__(self) -> int:
        return self._length
    def __getitem__(self,index):
        if index < 0:
            if -index <= len(self._data):
                return self._data[index]
            else:
                return self._read_point_from_file(self._length+index)
        if index >= 0:
            if index >= self._length-len(self._data):
                return self._data[index-self._length]
            else:
                return self._read_point_from_file(index)
    def __setitem__(self, index, value):
        if index < 0:
            if -index < len(self._data):
                self._data[index] = value
            else:
                raise IndexError(f'Index {index} is out of range (Already written to file).')
        if index >= 0:
            if index >= self._length-len(self._data):
                self._data[index-self._length] = value
            else:
                raise IndexError(f'Index {index} is out of range (Already written to file).')
    def __iter__(self):
        # Ensure that everything is written to disk
        assert self._file_handle is not None
        self._file_handle.flush()
        # Retrieve data from files
        for file_metadata in self._files[:-1]:
            with open(file_metadata['filename'],'rb') as f:
                for _ in range(file_metadata['num_points']):
                    yield dill.load(f)
        # Last file does not have all the points, so we need to handle this separately
        file_metadata = self._files[-1]
        if self.iterate_past_end:
            with open(file_metadata['filename'],'rb') as f:
                try:
                    while True:
                        yield dill.load(f)
                except EOFError:
                    pass
        else:
            with open(file_metadata['filename'],'rb') as f:
                for _ in range(file_metadata['num_points']):
                    yield dill.load(f)
            for x in self._data:
                yield x
    def __reversed__(self):
        raise NotImplementedError()
    def _read_point_from_file(self, index):
        if index < 0:
            raise IndexError(f'Index {index} is out of range.')
        if self._file_handle is not None:
            self._file_handle.flush()
        total_points = 0
        filename = None
        points_in_file = None
        for x in self._files:
            total_points += x['num_points']
            if total_points > index:
                filename = x['filename']
                points_in_file = x['num_points']
                break
        if filename is None or points_in_file is None:
            raise IndexError(f'Index {index} is out of bounds.')
        with open(filename,'rb') as f:
            if self._index_enabled:
                pos = self._indices[index]
                f.seek(pos)
                return dill.load(f)
            else:
                index_in_file = points_in_file-(total_points-index)
                print(index_in_file)
                for i in range(index_in_file):
                    print(f'skipping {i}')
                    dill.load(f)
                return dill.load(f)
    def _write_point_to_file(self) -> None:
        if self._file_handle is None:
            self._new_file()
        if self._index_enabled:
            assert self._file_handle is not None
            pos = self._file_handle.tell()
            self._indices.append(pos)
        dill.dump(self._data[0],self._file_handle)
        self._files[-1]['num_points'] += 1
        self._data = self._data[1:]
        if self._max_file_length is not None and self._files[-1]['num_points'] >= self._max_file_length:
            self._new_file()
    def append(self, obj) -> None:
        """ Append an object to the list. """
        self._length += 1
        self._data.append(obj)
        if self._max_memory_length is not None and len(self._data) > self._max_memory_length:
            self._write_point_to_file()
    def flush(self) -> None:
        """ Flush the data to disk. """
        if self._file_handle is not None:
            while len(self._data) > 0:
                self._write_point_to_file()
            self._file_handle.flush()
    def close(self) -> None:
        """ Flush data and close the file handle. """
        self.flush()
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
    def state_dict(self) -> dict:
        """ Return a dictionary containing the state of the list. """
        return {
            '_base_filename': self._base_filename,
            '_max_memory_length': self._max_memory_length,
            '_max_file_length': self._max_file_length,
            'files': self._files,
            'data': self._data,
            'length': self._length,
            'index_enabled': self._index_enabled,
            'indices': self._indices,
        }
    def load_state_dict(self, state):
        """ Load the state of the list from a dictionary. """
        self._base_filename = state['_base_filename']
        self._max_memory_length = state['_max_memory_length']
        self._max_file_length = state['_max_file_length']
        self._files = state['files']
        self._data = state['data']
        self._length = state['length']
        self._index_enabled = state['index_enabled']
        self._indices = state['indices']
        self._new_file()
        if self._file_handle is not None:
            self._file_handle.close()
        self._file_handle = open(self._files[-1]['filename'],'wb')
