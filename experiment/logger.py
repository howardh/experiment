from typing import Union, List

class Logger:
    def __init__(self,
            key_name : Union[str, List[str]] = None,
            manual_iteration : bool = False,
            overwrite : bool = False,
            allow_implicit_key : bool = False):
        """
        Args:
            key_name: Name of a logged value that is unique for each iteration.
                Consecutive calls to `Logger.log` with the same value for this key will be logged together.
                If not specified, then each call to `Logger.log` is assumed to be a distinct iteration.
                Can be a list of strings if multiple keys form the iteration's identifier, in which case, at least one of the listed values have to be different between iterations.
            overwrite: If set to `True`, then calls to `Logger.log` can overwrite existing logged data with the same key. If `False`, then an exception will be raised instead.
            allow_implicit_key (bool): If set to `True`, then the key needs only be specified when it changes. If unspecified, then it is assumed to be unchanged. If set to `False`, then the key must always be present.
        """
        self.key_name = key_name
        self.manual_iteration = manual_iteration
        self.overwrite = overwrite
        self.allow_implicit_key = allow_implicit_key

        self.data = []
        self.keys = set()

    def __getitem__(self,index : Union[str,int]):
        if type(index) is str:
            if index not in self.keys:
                raise KeyError('Key %s not found in logs.' % index)
            if type(self.key_name) is not str:
                raise NotImplementedError('')
            x = []
            y = []
            for d in self.data:
                if index not in d:
                    continue
                y.append(d[index])
                x.append(d[self.key_name])
            return x,y
        elif type(index) is int:
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
        if type(self.key_name) is str:
            if self.key_name not in data.keys():
                if self.allow_implicit_key:
                    return False
                else:
                    raise Exception('`key_name` is specified as "%s", but this key is not present in the logged data.' % self.key_name)
            return data[self.key_name] != self.data[-1][self.key_name]
        if type(self.key_name) is list:
            for k in self.key_name:
                if k not in data.keys():
                    if self.allow_implicit_key:
                        return False
                    else:
                        raise Exception('`key_name` is specified as "%s", but "%s" is not present in the logged data.' % (self.key_name, k))
                if data[k] != self.data[-1][k]:
                    return True
            return False

    def next_iteration(self):
        if not self.manual_iteration:
            raise Exception('`manual_iteration` has to be set to `True` to use next_iteration.')
        self.data.append({})

    def log(self,**data):
        if self._did_key_change(data) and not self.manual_iteration:
            self.data.append({})
        for k,v in data.items():
            if k in self.data[-1]:
                if not self._is_logger_key(k) and not self.overwrite:
                    raise Exception('Key "%s" already exists for this iteration.' % k)
            self.data[-1][k] = v
            self.keys.add(k)
            
    def append(self,**data):
        if self._did_key_change(data) and not self.manual_iteration:
            self.data.append({})
        for k,v in data.items():
            if self._is_logger_key(k):
                self.data[-1][k] = v
                continue
            if k in self.data[-1]:
                if type(self.data[-1][k]) is not list:
                    if self.overwrite:
                        self.data[-1][k] = []
                    else:
                        raise Exception('Key "%s" has already been assigned (probably via `Logger.log()`). The same key cannot be used in both `log()` and `append()`.' % k)
                self.data[-1][k].append(v)
            else:
                self.data[-1][k] = [v]
            self.keys.add(k)

    def mean(self, key) -> float:
        total = 0
        count = 0
        for d in self.data:
            if key in d:
                total += d[key]
                count += 1
        return total/count

    def state_dict(self):
        return {
                'data': self.data,
                'manual_iteration': self.manual_iteration,
                'key_name': self.key_name,
                'keys': self.keys,
        }
    def load_state_dict(self, state):
        self.data = state.get('data', [])
        self.key_name = state.get('key_name', None)
        self.manual_iteration = state.get('manual_iteration', False)
        if 'keys' in state:
            self.keys = state['keys']
        else:
            for d in self.data:
                for k in d.keys():
                    self.keys.add(k)
