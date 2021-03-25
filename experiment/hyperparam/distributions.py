
"""
- Constant
- Discrete space
    - Integers (i.e. with ordinal relationship)
    - Categorical (i.e. no ordinal relationship)
- Continuous
    - Bounded
        - Uniform
        - Log uniform
    - Unbounded
        - Normal
"""

import numpy as np
from abc import ABC, abstractmethod

class Distribution(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def sample(self):
        pass

class Constant(Distribution):
    def __init__(self, val):
        self.val = val
    def __len__(self):
        return 1
    def __repr__(self):
        return 'Constant(%s)' % self.val
    def sample(self):
        return self.val
    def linspace(self):
        return [self.val]

class Categorical(Distribution):
    def __init__(self, vals):
        self.vals = vals
    def __len__(self):
        return len(self.vals)
    def __repr__(self):
        return 'Categorical(%s)' % self.vals
    def sample(self):
        return np.random.choice(self.vals)
    def linspace(self):
        return self.vals

class Uniform(Distribution):
    def __init__(self, min_val, max_val, n=None):
        if max_val < min_val:
            raise Exception('max_val must be less than min_val. Received max_val=%f and min_val=%f.' % (max_val,min_val))
        self.min_val = min_val
        self.max_val = max_val
        self.n = n
    def __len__(self):
        if self.n is None:
            raise Exception('No length defined.')
        return self.n
    def __repr__(self):
        return 'Uniform(%f,%f,n=%s)' % (self.min_val, self.max_val, self.n)
    def sample(self):
        return np.random.rand()*(self.max_val-self.min_val)+self.min_val
    def linspace(self, n=None):
        if n is None:
            n = self.n
        if n is None:
            raise Exception('`n` not specified.')
        return np.linspace(self.min_val, self.max_val, n)

class IntUniform(Uniform):
    def __init__(self, min_val, max_val, n=None):
        super().__init__(min_val,max_val,n)
    def __len__(self):
        if self.n is None:
            return int(max_val-min_val)
        return self.n
    def __repr__(self):
        return 'IntUniform(%f,%f,n=%s)' % (self.min_val, self.max_val, self.n)
    def sample(self):
        return int(super().sample())
    def linspace(self, n=None):
        if n is None:
            n = self.n
        if n is None:
            return np.arange(self.min_val, self.max_val)
        else:
            return np.floor(np.linspace(self.min_val, self.max_val, n))

class LogUniform(Uniform):
    def __init__(self, min_val, max_val, n=None):
        super().__init__(np.log(min_val),np.log(max_val),n)
    def __repr__(self):
        return 'LogUniform(%f,%f,n=%s)' % (np.exp(self.min_val), np.exp(self.max_val,self.n))
    def sample(self):
        return np.exp(super().sample())
    def linspace(self, n=None):
        return np.exp(super().linspace(n))
