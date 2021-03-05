from abc import ABC, abstractmethod
from typing import Callable
from collections import defaultdict
from collections.abc import Mapping
import itertools
import numpy as np
import os
import dill

from experiment import Experiment, ExperimentRunner
from experiment.utils import find_next_free_dir
from .distributions import Distribution, Constant, LogUniform

##################################################

def normalize_search_space(search_space):
    def normalize(x):
        if isinstance(x,Distribution):
            return x
        else:
            return Constant(x)
    return {k:normalize(v) for k,v in search_space.items()}

def serializable_search_space(search_space):
    def serialize(x):
        if isinstance(x,Distribution):
            return str(x)
        else:
            return str(Constant(x))
    return {k:serialize(v) for k,v in search_space.items()}

def unserializable_search_space(search_space):
    def unserialize(x):
        return exec(x)
    return {k:unserialize(v) for k,v in search_space.items()}

def search_space_vector_keys(search_space):
    keys = []
    for k,v in search_space.items():
        if isinstance(v,Distribution) and not isinstance(v,Constant):
            keys.append(k)
    keys = sorted(keys)
    return keys

def search_space_bounds(search_space):
    keys = search_space_vector_keys(search_space)
    bounds = []
    for k in keys:
        dist = search_space[k]
        bounds.append((dist.min_val,dist.max_val))
    return bounds

def search_space_sample(search_space):
    return {k:v.sample() for k,v in search_space.items()}

def config_dict_to_vector(search_space, config):
    keys = search_space_vector_keys(search_space)
    return [config[k] for k in keys]

def vector_to_config_dict(search_space, vec):
    keys = search_space_vector_keys(search_space)
    config = {
            **search_space
    }
    for k,v in search_space.items():
        if isinstance(v,Constant):
            config[k] = v.sample()
    for k,v in zip(keys,vec):
        if isinstance(search_space[k], LogUniform):
            config[k] = np.exp(v)
        else:
            config[k] = v
    return config

##################################################

class Search(ABC):
    def __init__(self, cls,
            search_space: Mapping,
            maximize: bool = False,
            **exp_runner_kwargs):
        """
        Args:
            cls: Experiment class for the experiment to be run.
            search_space (collections.abc.Mapping): Search space
            maximize (bool): If set to True, then the search will seek to maximize the score function instead of minimizing.
        """
        self.cls = cls
        self.search_space = normalize_search_space(search_space)
        self.maximize = maximize
        self.exp_runner_kwargs = exp_runner_kwargs

class GridSearch(Search):
    def __init__(self, cls,
            search_space: Mapping,
            root_directory: str = './results',
            output_directory: str = None,
            **exp_runner_kwargs):
        """
        Args:
            root_directory: If specified, then a subdirectory will be created within the root directory, and all output for the gridsearch will be placed in the newly-created subdirectory.
            output_directory: If specified, then all output for the gridsearch will be placedin this directory.
                Takes precedence over `root_directory` if both are specified.
        """
        super().__init__(cls, search_space, **exp_runner_kwargs)
        self.root_directory = root_directory
        self.directory = find_next_free_dir(
                self.root_directory, 'Gridsearch-{}-%d'.format(cls.__name__))
    def run(self):
        keys = self.search_space.keys()
        all_vals = list(itertools.product(*[self.search_space[k].linspace() for k in keys]))
        for i,vals in enumerate(all_vals):
            config = {k:v for k,v in zip(keys,vals)}
            exp = ExperimentRunner(self.cls,
                    root_directory=os.path.join(self.directory,'Experiments'),
                    config=config, **self.exp_runner_kwargs)
            exp.run()

class RandomSearch(Search):
    def __init__(self, cls,
            search_space: Mapping,
            name: str = None,
            root_directory: str = './results',
            output_directory: str = None,
            **exp_runner_kwargs):
        """
        Args:
            root_directory: If specified, then a subdirectory will be created within the root directory, and all output for the gridsearch will be placed in the newly-created subdirectory.
            output_directory: If specified, then all output for the gridsearch will be placedin this directory.
                Takes precedence over `root_directory` if both are specified.
        """
        super().__init__(cls, search_space, **exp_runner_kwargs)
        if name is None:
            name = 'RandomSearch-%s' % cls.__name__
        self.root_directory = root_directory
        self.directory = find_next_free_dir(
                self.root_directory, '{}-%d'.format(name))
        self.search_space = normalize_search_space(search_space)
    def run(self):
        for _ in range(5):
            config = {k:v.sample() for k,v in self.search_space.items()}
            exp = ExperimentRunner(self.cls,
                    root_directory=os.path.join(self.directory,'Experiments'),
                    config=config, **self.exp_runner_kwargs)
            exp.run()

class BayesianOptimizationSearch(Search):
    def __init__(self, cls,
            search_space: Mapping,
            score_fn: Callable[[Experiment],int],
            name: str = None,
            search_budget: int = 5,
            root_directory: str = './results',
            output_directory: str = None,
            **exp_runner_kwargs):
        super().__init__(cls, search_space, **exp_runner_kwargs)
        if name is None:
            name = 'BayesianOptimizationSearch-%s' % cls.__name__
        self.score_fn = score_fn
        self.search_budget = search_budget
        self.root_directory = root_directory
        self.directory = find_next_free_dir(
                self.root_directory, '{}-%d'.format(name))
        self.search_space = normalize_search_space(search_space)
    def run(self):
        import skopt
        from skopt import gp_minimize
        # Bounds
        bounds = search_space_bounds(self.search_space)
        # Objective function
        def objective_fn(x):
            config = vector_to_config_dict(self.search_space, x)
            exp = ExperimentRunner(self.cls,
                    root_directory=os.path.join(self.directory,'Experiments'),
                    config=config, **self.exp_runner_kwargs)
            exp.run()
            return self.score_fn(exp.exp)
        # Perform search
        results = gp_minimize(
                func=objective_fn,
                dimensions=bounds,
                acq_func='EI',
                n_calls=self.search_budget,
                n_random_starts=3
        )

##################################################

class Analysis(ABC):
    def best_config(self):
        pass
    def best_score(self):
        pass

class SimpleAnalysis(Analysis):
    """ Treat all runs as independent runs. """
    def __init__(self, cls,
            score_fn: Callable[[Experiment],int],
            maximize=False,
            directory: str = None):
        self.directory = directory
        self.score_fn = score_fn
        self.maximize = maximize
        self.cls = cls
        self.results = None
        self.sorted_results = None
    def _load_results(self):
        self.results = []
        for name in os.listdir(self.directory):
            checkpoint_filename = os.path.join(self.directory,name,'checkpoint.pkl')
            with open(checkpoint_filename,'rb') as f:
                checkpoint = dill.load(f)
            config = checkpoint['args']['config']
            exp = self.cls()
            exp.load_state_dict(checkpoint['exp'])
            self.results.append((self.score_fn(exp),config))
    def _sort_results(self):
        if self.results is None:
            self._load_results()
        if self.sorted_results is not None:
            return
        sorted_results = sorted(
                self.results,
                key=lambda x: x[0],
                reverse=self.maximize
        )
        self.sorted_results = sorted_results
    def get_best_config(self):
        self._sort_results()
        score,config = self.sorted_results[0]
        return config
    def get_best_score(self):
        self._sort_results()
        score,config = self.sorted_results[0]
        return score

class GroupedAnalysis(Analysis):
    """ Group together runs that use the same hyperparameters. """
    def __init__(self, cls,
            score_fn: Callable[[Experiment],int],
            maximize=False,
            directory: str = None):
        self.directory = directory
        self.score_fn = score_fn
        self.maximize = maximize
        self.cls = cls
        self.results = None
        self.sorted_results = None
    def _load_results(self):
        self.results = defaultdict(lambda: [])
        for name in os.listdir(self.directory):
            checkpoint_filename = os.path.join(self.directory,name,'checkpoint.pkl')
            with open(checkpoint_filename,'rb') as f:
                checkpoint = dill.load(f)
            config = checkpoint['args']['config']
            exp = self.cls()
            exp.load_state_dict(checkpoint['exp'])
            key = frozenset(config.items())
            self.results[key].append(self.score_fn(exp))
    def _sort_results(self):
        if self.results is None:
            self._load_results()
        if self.sorted_results is not None:
            return
        sorted_results = sorted(
                self.results.items(),
                key=lambda x: np.mean(x[1]),
                reverse=self.maximize
        )
        self.sorted_results = sorted_results
    def get_best_config(self):
        self._sort_results()
        config,score = self.sorted_results[0]
        return dict(config)
    def get_best_score(self):
        self._sort_results()
        config,score = self.sorted_results[0]
        return score

class GaussianProcessAnalysis(Analysis):
    """ Fit a Gaussian Process to the results. """
    def __init__(self, cls,
            score_fn: Callable[[Experiment],int],
            search_space: Mapping,
            maximize=False,
            directory: str = None):
        self.directory = directory
        self.score_fn = score_fn
        self.search_space = search_space
        self.maximize = maximize
        self.cls = cls
        self.results = None
        self.best_result = None
    def _load_results(self):
        self.results = []
        for name in os.listdir(self.directory):
            checkpoint_filename = os.path.join(self.directory,name,'checkpoint.pkl')
            with open(checkpoint_filename,'rb') as f:
                checkpoint = dill.load(f)
            config = checkpoint['args']['config']
            exp = self.cls()
            exp.load_state_dict(checkpoint['exp'])
            self.results.append((self.score_fn(exp), config))
    def _find_optimum(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from scipy.optimize import minimize
        # Load if needed
        if self.results is None:
            self._load_results()
        # Bounds
        bounds = search_space_bounds(self.search_space)
        # Initial Guess
        x0 = config_dict_to_vector(
                self.search_space,
                search_space_sample(self.search_space)
        )
        # Extract data
        x = [[config[k] for k in keys] for _,config in self.results]
        y = [score for score,_ in self.results]
        # Fit GP
        kernel = Matern()
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(x,y)
        # Find optimum
        opt_result = minimize(
                fun=lambda x: gpr.predict([x]),
                method='L-BFGS-B',
                bounds=bounds,
                x0=x0
        )
        # Convert back to config
        best_config = vector_to_config_dict(self.search_space, opt_result.x)
        # Save optimum
        self.best_result = (opt_result.fun.item(), best_config)
    def get_best_config(self):
        self._find_optimum()
        score,config = self.best_result
        return config
    def get_best_score(self):
        self._find_optimum()
        score,config = self.best_result
        return score
