
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

class Discrete(Distribution):
    def __init__(self, vals):
        self.vals = vals
    def __len__(self):
        return len(self.vals)
    def __repr__(self):
        return 'Discrete(%s)' % self.vals
    def sample(self):
        return np.random.choice(self.vals)
    def linspace(self):
        return vals

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
        return np.linspace(self.min_val, self.max_val, n)

class LogUniform(Distribution):
    def __init__(self, min_val, max_val, n=None):
        self.min_val = np.log(min_val)
        self.max_val = np.log(max_val)
        self.n = n
    def __len__(self):
        if self.n is None:
            raise Exception('No length defined.')
        return self.n
    def __repr__(self):
        return 'LogUniform(%f,%f,n=%s)' % (np.exp(self.min_val), np.exp(self.max_val,self.n))
    def sample(self):
        return np.exp(np.random.rand()*(self.max_val-self.min_val)+self.min_val)
    def linspace(self, n=None):
        if n is None:
            n = self.n
        return np.exp(np.linspace(self.min_val, self.max_val, n))
