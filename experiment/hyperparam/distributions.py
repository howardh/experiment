from typing import Optional, List
from abc import ABC, abstractmethod

import numpy as np
import skopt

class Distribution(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def sample(self):
        pass
    @abstractmethod
    def linspace(self) -> List:
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
    #def skopt_space(self):
    #    return skopt.space.Categorical([self.val])

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
    def skopt_space(self):
        return skopt.space.Categorical(self.vals, transform='label')

class Uniform(Distribution):
    """ Range from `min_val` inclusive to `max_val` exclusive.  """
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
    def skopt_space(self):
        return skopt.space.Real(self.min_val,self.max_val,prior='uniform', transform='normalize')

class IntUniform(Uniform):
    """ Range from `min_val` to `max_val` inclusive.  """
    def __init__(self, min_val, max_val, n=None):
        super().__init__(min_val,max_val+1,n)
    def __len__(self):
        if self.n is None:
            return int(self.max_val-self.min_val)
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
    def skopt_space(self):
        return skopt.space.Integer(self.min_val,self.max_val-1,prior='uniform', transform='normalize')

class LogUniform(Uniform):
    """ Range from `min_val` to `max_val` inclusive.  """
    def __init__(self, min_val : float, max_val : float, n : Optional[int] = None):
        super().__init__(np.log(min_val),np.log(max_val),n)
        if min_val <= 0 or max_val <= 0:
            raise Exception('Range must be strictly positive.')
    def __repr__(self):
        return 'LogUniform(%f,%f,n=%s)' % (np.exp(self.min_val), np.exp(self.max_val),self.n)
    def sample(self):
        return np.exp(super().sample())
    def linspace(self, n=None):
        return np.exp(super().linspace(n))
    def skopt_space(self):
        return skopt.space.Real(np.exp(self.min_val),np.exp(self.max_val),prior='log-uniform', transform='normalize')

class LogIntUniform(Uniform):
    """ Range from `min_val` to `max_val` inclusive.  """
    def __init__(self, min_val, max_val, n=None):
        super().__init__(np.log(min_val),np.log(max_val),n)
        self._int_range = (min_val,max_val)
        if min_val <= 0 or max_val <= 0:
            raise Exception('Range must be strictly positive.')
        self._cum_probs = self._compute_cum_probs()
    def __repr__(self):
        return 'LogIntUniform(%f,%f,n=%s)' % (np.exp(self.min_val), np.exp(self.max_val),self.n)
    def sample(self):
        r = np.random.rand()
        for i,p in enumerate(self._cum_probs):
            if r <= p:
                return self._int_range[0]+i
    def linspace(self, n=None):
        if n is None:
            n = self.n
        output = []
        j = 0
        for i,v in enumerate(range(self._int_range[0], self._int_range[1]+1)):
            if self._cum_probs[i] >= j/(n-1):
                output.append(v)
                j += 1
        return np.array(output)
    def _compute_cum_probs(self):
        a,b = self._int_range
        if a == b:
            return np.array([1.])
        else:
            loga = np.log(a)
            logb = np.log(b+1)
            cum_probs = (np.log(np.arange(a,b+2))-loga)/(logb-loga)
            cum_probs = cum_probs[1:]
            return cum_probs
    def skopt_space(self):
        return skopt.space.Integer(self._int_range[0],self._int_range[1],prior='log-uniform')
