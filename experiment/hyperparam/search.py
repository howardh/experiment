from abc import ABC
from typing import Callable, Generic, Type, TypeVar, Union, Optional
from typing_extensions import TypedDict
from collections import defaultdict
from collections.abc import Mapping
import itertools
import os

import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import dill
import skopt
import scipy
import scipy.optimize
from skopt.learning.gaussian_process import GaussianProcessRegressor
#from skopt.learning.gaussian_process.kernels import Matern, RBF
from skopt.learning.gaussian_process.kernels import Matern

from experiment import Experiment, ExperimentRunner, load_checkpoint
from experiment.utils import find_next_free_dir
#from .distributions import Distribution, Constant, Uniform, IntUniform, LogUniform, LogIntUniform, Categorical
from .distributions import Distribution, Constant, Uniform, Categorical

##################################################

ExpType = TypeVar('ExpType', bound=Experiment) # Must be a subclass of `Experiment`

class ProjectedPoint(TypedDict):
    length: np.ndarray
    distance: Optional[np.ndarray]

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

def search_space_vector_keys(search_space, include_non_categorical=True, include_categorical=True):
    keys = []
    for k,v in search_space.items():
        if isinstance(v,Distribution) and not isinstance(v,Constant):
            if include_non_categorical and isinstance(v,Uniform):
                keys.append(k)
            elif include_categorical and isinstance(v,Categorical):
                keys.append(k)
    keys = sorted(keys)
    return keys

def search_space_bounds(search_space, include_non_categorical=True, include_categorical=True):
    keys = search_space_vector_keys(search_space, include_non_categorical=include_non_categorical, include_categorical=include_categorical)
    bounds = []
    for k in keys:
        dist = search_space[k]
        try:
            bounds.append((dist.min_val,dist.max_val))
        except:
            bounds.append(None)
    return bounds

def search_space_skopt_dimensions(search_space):
    keys = search_space_vector_keys(search_space, include_non_categorical=True, include_categorical=True)
    bounds = []
    for k in keys:
        dist = search_space[k]
        bounds.append(dist.skopt_space())
    return bounds

def search_space_sample(search_space):
    return {k:v.sample() for k,v in search_space.items()}

def config_dict_to_vector(search_space, config, normalized=False):
    keys = search_space_vector_keys(search_space)
    def convert(k):
        if normalized:
            return search_space[k].skopt_space().transform(config[k])
        else:
            return config[k]
    return [convert(k) for k in keys]

def vector_to_config_dict(search_space, vec, normalized=False):
    keys = search_space_vector_keys(search_space)
    config = {
            **search_space
    }
    for k,v in search_space.items():
        if isinstance(v,Constant):
            config[k] = v.sample()
    for k,v in zip(keys,vec):
        if normalized:
            config[k] = search_space[k].skopt_space().inverse_transform(v)
        else:
            config[k] = v
    return config

def project_point(
        x : np.ndarray,
        x0 : np.ndarray,
        x1 : np.ndarray,
        include_distance : bool = False
    ) -> ProjectedPoint:
    """
    Return the length of the projection of point `x` onto the line formed by the line between `x0` and `x1`
    """
    v = (x1-x0).reshape(1,-1)
    u = (x-x0).reshape(-1,len(x0))
    v_len = np.sqrt(np.sum(v**2))
    scalar_proj_u = (u@v.T/v_len).reshape(-1,1)/v_len

    if include_distance:
        proj_u = x0+scalar_proj_u*v
        dist = np.sqrt(np.sum((u-proj_u)**2, axis=1))
        if dist.min() == dist.max():
            dist[:] = 1
        return {'length':scalar_proj_u, 'distance': dist}
    else:
        return {'length':scalar_proj_u, 'distance': None}

def expected_minimum_bfgs(res, random_state=None): 
    # Random starting guesses
    random_samples = res.space.rvs(random_state=random_state)
    random_samples = res.space.transform(random_samples)
    # Optimize
    model = res.models[-1]
    opt_result = scipy.optimize.minimize(
            fun=lambda x: model.predict(x.reshape(1, -1))[0],
            method='L-BFGS-B',
            bounds=res.space.transformed_bounds,
            x0=random_samples
    )
    return res.space.inverse_transform(
            opt_result.x.reshape(1,res.space.n_dims)
    )

def plot_gaussian_process(res, **kwargs):
    """Plots the optimization results and the gaussian process
    for any objective functions. Points are all projected onto the 1D line spanning from one corner of the search space to the other.
    Code copied and adapted from [here](https://github.com/scikit-optimize/scikit-optimize/blob/de32b5fd2205a1e58526f3cacd0422a26d315d0f/skopt/plots.py#L108).

    Parameters
    ----------
    res :  `OptimizeResult`
        The result for which to plot the gaussian process.
    ax : `Axes`, optional
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.
    n_calls : int, default: -1
        Can be used to evaluate the model at call `n_calls`.
    objective : func, default: None
        Defines the true objective function. Must have one input parameter.
    n_points : int, default: 1000
        Number of data points used to create the plots
    noise_level : float, default: 0
        Sets the estimated noise level
    show_legend : boolean, default: True
        When True, a legend is plotted.
    show_title : boolean, default: True
        When True, a title containing the found minimum value
        is shown
    show_acq_func : boolean, default: False
        When True, the acquisition function is plotted
    show_next_point : boolean, default: False
        When True, the next evaluated point is plotted
    show_observations : boolean, default: True
        When True, observations are plotted as dots.
    show_mu : boolean, default: True
        When True, the predicted model is shown.
    Returns
    -------
    ax : `Axes`
        The matplotlib axes.
    """
    ax = kwargs.get("ax", None)
    n_calls = kwargs.get("n_calls", -1)
    objective = kwargs.get("objective", None)
    noise_level = kwargs.get("noise_level", 0)
    show_legend = kwargs.get("show_legend", True)
    show_title = kwargs.get("show_title", True)
    show_acq_func = kwargs.get("show_acq_func", False)
    #show_next_point = kwargs.get("show_next_point", False)
    show_observations = kwargs.get("show_observations", True)
    show_mu = kwargs.get("show_mu", True)
    n_points = kwargs.get("n_points", 1000)

    if ax is None:
        ax = plt.gca()
    x0 = np.array([b[0] for b in res.space.transformed_bounds])
    x1 = np.array([b[1] for b in res.space.transformed_bounds])
    x = np.linspace(0,1,n_points)
    x_model = np.linspace(x0,x1,n_points)
    x = x.reshape(-1, 1)
    x_model = x_model.reshape(-1, res.space.n_dims)

    acq_func_kwargs = None
    acq_func = None
    n_random = None
    if res.specs is not None and "args" in res.specs:
        n_random = res.specs["args"].get('n_random_starts', None)
        acq_func = res.specs["args"].get("acq_func", "EI")
        acq_func_kwargs = res.specs["args"].get("acq_func_kwargs", {})

    if acq_func_kwargs is None:
        acq_func_kwargs = {}
    if acq_func is None or acq_func == "gp_hedge":
        acq_func = "EI"
    if n_random is None:
        n_random = len(res.x_iters) - len(res.models)

    fx = None
    if objective is not None:
        fx = np.array([objective(x_i) for x_i in x_model])
    if n_calls < 0:
        model = res.models[-1]
        curr_x_iters = res.x_iters
        curr_func_vals = res.func_vals
    else:
        model = res.models[n_calls]

        curr_x_iters = res.x_iters[:n_random + n_calls]
        curr_func_vals = res.func_vals[:n_random + n_calls]
    curr_x_iters = res.space.transform(curr_x_iters)
    curr_x_iters_proj,curr_x_iters_dist = project_point(curr_x_iters,x0,x1,include_distance=True)

    # Plot true function.
    if objective is not None:
        ax.plot(x, fx, "r--", label="True (unknown)")
        ax.fill(np.concatenate(
            [x, x[::-1]]),
            np.concatenate(([fx_i - 1.9600 * noise_level
                             for fx_i in fx],
                            [fx_i + 1.9600 * noise_level
                             for fx_i in fx[::-1]])),
            alpha=.2, fc="r", ec="None")

    # Plot GP(x) + contours
    if show_mu:
        y_pred, sigma = model.predict(x_model, return_std=True)
        ax.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
        ax.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                alpha=.2, fc="g", ec="None")

    # Plot sampled points
    if show_observations:
        MIN_OPACITY = 0.1
        ax.scatter(curr_x_iters_proj, curr_func_vals,
                c=[(1,0,0,1-(d/np.max(curr_x_iters_dist))*(1-MIN_OPACITY)) for d in curr_x_iters_dist],
                s=8*4, label="Observations")
    if (show_mu or show_observations or objective is not None)\
            and show_acq_func:
        ax_ei = ax.twinx()
        ax_ei.set_ylabel(str(acq_func) + "(x)")
        plot_both = True
    else:
        ax_ei = ax
        plot_both = False
    if show_acq_func:
        raise NotImplementedError()
        #acq = _gaussian_acquisition(x_model, model,
        #                            y_opt=np.min(curr_func_vals),
        #                            acq_func=acq_func,
        #                            acq_func_kwargs=acq_func_kwargs)
        #next_x = x[np.argmin(acq)]
        #next_acq = acq[np.argmin(acq)]
        #acq = - acq
        #next_acq = -next_acq
        #ax_ei.plot(x, acq, "b", label=str(acq_func) + "(x)")
        #if not plot_both:
        #    ax_ei.fill_between(x.ravel(), 0, acq.ravel(),
        #                       alpha=0.3, color='blue')

        #if show_next_point and next_x is not None:
        #    ax_ei.plot(next_x, next_acq, "bo", markersize=6,
        #               label="Next query point")

    if show_title:
        ax.set_title(r"x* = %s, f(x*) = %.4f" % (res.x, res.fun))
    # Adjust plot layout
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    if show_legend:
        if plot_both:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_ei.get_legend_handles_labels()
            ax_ei.legend(lines + lines2, labels + labels2, loc="best",
                         prop={'size': 6}, numpoints=1)
        else:
            ax.legend(loc="best", prop={'size': 6}, numpoints=1)

    return ax

##################################################

def load_past_experiment_results(cls, root_dir, search_space, score_fn, normalized=False):
    x0 = []
    y0 = []
    for exp_dir in os.listdir(root_dir):
        # Load them into Experiment objects
        exp = load_checkpoint(cls, os.path.join(root_dir,exp_dir))
        # Extract config
        config = exp.config
        config_vec = config_dict_to_vector(search_space, config, normalized=normalized)
        # Apply score_fn to them
        score = score_fn(exp.exp)
        # Save as starting data for gp_minimize
        x0.append(config_vec)
        y0.append(score)
    return x0,y0

##################################################

class Search(ABC, Generic[ExpType]):
    def __init__(self,
            cls : Type[ExpType],
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
            name = 'GridSearch-%s' % cls.__name__
        self.root_directory = root_directory
        if output_directory is None:
            self.directory = find_next_free_dir(
                    self.root_directory, '{}-%d'.format(name))
        else:
            self.directory = output_directory
    def run(self):
        keys = self.search_space.keys()
        all_vals = list(itertools.product(*[self.search_space[k].linspace() for k in keys]))
        for vals in all_vals:
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
            output_directory: Optional[str] = None,
            search_budget: int = 10,
            **exp_runner_kwargs):
        """
        Args:
            root_directory: If specified, then a subdirectory will be created within the root directory, and all output for the gridsearch will be placed in the newly-created subdirectory.
            output_directory: If specified, then all output for the gridsearch will be placedin this directory.
                Takes precedence over `root_directory` if both are specified.
            search_budget: Number of random experiments to execute.
        """
        super().__init__(cls, search_space, **exp_runner_kwargs)
        if name is None:
            name = 'RandomSearch-%s' % cls.__name__
        self.root_directory = root_directory
        if output_directory is None:
            self.directory = find_next_free_dir(
                    self.root_directory, '{}-%d'.format(name))
        else:
            self.directory = output_directory
        self.search_space = normalize_search_space(search_space)
        self.search_budget = search_budget
    def run(self):
        for _ in range(self.search_budget):
            config = {k:v.sample() for k,v in self.search_space.items()}
            exp = ExperimentRunner(self.cls,
                    root_directory=os.path.join(self.directory,'Experiments'),
                    config=config, **self.exp_runner_kwargs)
            exp.run()

class BayesianOptimizationSearch(Search):
    def __init__(self,
            cls : Type[ExpType],
            search_space: Mapping,
            score_fn: Callable[[ExpType],Union[int,float]],
            name: str = None,
            search_budget: int = 5,
            kernel = Matern(length_scale=1,nu=2.5,length_scale_bounds='fixed'),
            root_directory: str = './results',
            output_directory: str = None,
            **exp_runner_kwargs):
        super().__init__(cls, search_space, **exp_runner_kwargs)
        if name is None:
            name = 'BayesianOptimizationSearch-%s' % cls.__name__
        self.score_fn = score_fn
        self.kernel = kernel
        self.search_budget = search_budget
        self.root_directory = root_directory
        if output_directory is None:
            self.directory = find_next_free_dir(
                    self.root_directory, '{}-%d'.format(name))
        else:
            self.directory = output_directory
        self.search_space = normalize_search_space(search_space)
        self.results = None
        self.expected_min = None
    def run(self):
        from skopt import gp_minimize
        # Parameters
        exp_runner_root_dir = os.path.join(self.directory,'Experiments')
        os.makedirs(exp_runner_root_dir,exist_ok=True)
        # Check if there's already existing results to load
        x0,y0 = load_past_experiment_results(self.cls, exp_runner_root_dir, self.search_space, self.score_fn, normalized=False)
        # Objective function
        def objective_fn(x):
            config = vector_to_config_dict(self.search_space, x)
            exp = ExperimentRunner(self.cls,
                    root_directory=exp_runner_root_dir,
                    config=config, **self.exp_runner_kwargs)
            exp.run()
            return self.score_fn(exp.exp)
        # Perform search
        gpr = GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=5)
        self.results = gp_minimize(
                func=objective_fn,
                base_estimator=gpr,
                dimensions=search_space_skopt_dimensions(self.search_space),
                acq_func='EI',
                n_calls=self.search_budget,
                x0=x0 if len(x0) > 0 else None,
                y0=y0 if len(x0) > 0 else None,
                n_initial_points=max(3-len(x0),0)
        )
        results = self.results
        self.expected_min = expected_minimum_bfgs(results)
        print(self.expected_min)
    def plot_gp(self, filename=None):
        if self.results is None:
            raise Exception('Search must be run before GP can be plotted.')
        plt.figure()
        plot_gaussian_process(self.results)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()

##################################################

class Analysis(ABC):
    def best_config(self):
        pass
    def best_score(self):
        pass

class SimpleAnalysis(Analysis):
    """ Treat all runs as independent runs. """
    def __init__(self,
            cls : Type[ExpType],
            score_fn: Callable[[ExpType],Union[int,float]],
            maximize : bool = False,
            directory : Optional[str] = None):
        if directory is None:
            raise Exception('Output directory must be specified.')
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
        if self.results is None:
            raise Exception('This should never happen. This is only here to make pyright happy.')
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
        _,config = self.sorted_results[0]
        return config
    def get_best_score(self):
        self._sort_results()
        score,_ = self.sorted_results[0]
        return score

class GroupedAnalysis(Analysis):
    """ Group together runs that use the same hyperparameters. """
    def __init__(self,
            cls : Type[ExpType],
            score_fn: Callable[[ExpType],Union[int,float]],
            maximize=False,
            directory: Optional[str] = None):
        if directory is None:
            raise Exception('Output directory must be specified.')
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
        config,_ = self.sorted_results[0]
        return dict(config)
    def get_best_score(self):
        self._sort_results()
        _,score = self.sorted_results[0]
        return score

class GaussianProcessAnalysis(Analysis):
    """ Fit a Gaussian Process to the results. """
    def __init__(self,
            cls : Type[ExpType],
            score_fn: Callable[[ExpType],Union[int,float]],
            search_space: Mapping,
            maximize=False,
            kernel = Matern(length_scale=1,nu=2.5,length_scale_bounds='fixed'),
            directory: str = None):
        self.directory = directory
        self.score_fn = score_fn
        self.search_space = normalize_search_space(search_space)
        self.skopt_search_space = skopt.utils.normalize_dimensions(search_space_skopt_dimensions(self.search_space))
        self.maximize = maximize
        self.cls = cls
        self.kernel = kernel

        self.results = None
        self.best_result = None
        self.model = None
    def _fit_model(self):
        if self.model is not None:
            return self.model
        # Load data
        x,y = load_past_experiment_results(self.cls, self.directory, self.search_space, self.score_fn, normalized=False)
        x = self.skopt_search_space.transform(x)
        # Fit GP
        gpr = GaussianProcessRegressor(kernel=self.kernel,normalize_y=False)
        gpr.fit(x,y)
        print(gpr.score(x,y))
        self.model = gpr
        return self.model
    def _find_optimum(self):
        from scipy.optimize import minimize
        gpr = self._fit_model()
        # Initial Guess
        x0 = self.skopt_search_space.transform(self.skopt_search_space.rvs())
        # Find optimum
        opt_result = minimize(
                fun=lambda x: gpr.predict([x]),
                method='L-BFGS-B',
                bounds=self.skopt_search_space.transformed_bounds,
                x0=x0
        )
        # Convert back to config
        best_config = vector_to_config_dict(
                self.search_space,
                self.skopt_search_space.inverse_transform([opt_result.x])[0]
        )
        # Save optimum
        self.best_result = (opt_result.fun.item(), best_config)
    def get_best_config(self):
        self._find_optimum()
        if self.best_result is None:
            raise Exception('This should never happen. This is only here to make pyright happy.')
        _,config = self.best_result
        return config
    def get_best_score(self):
        self._find_optimum()
        if self.best_result is None:
            raise Exception('This should never happen. This is only here to make pyright happy.')
        score,_ = self.best_result
        return score
    def compute_scores(self, config, normalized=False):
        gpr = self._fit_model()
        if isinstance(config, Mapping):
            # x = self.skopt_search_space.transform(config_dict_to_vector(config)) # FIXME: wtf was I trying to do here?
            raise NotImplementedError()
        else:
            if normalized:
                x = config
            else:
                x = self.skopt_search_space.transform(config)
        return gpr.predict(x)
    def plot(self, show_points=True, show_gp_mean=True, show_gp_std=False, acq_func=None, filename='plot.png'):
        """
        Args:
            show_points (bool): If `True`, create a scatter plot of each experiment result.
            show_gp_mean (bool): If `True`, plot a line representing the Gaussian Process fitted to the data.
            show_gp_std (bool): 
            acq_func (str): None, "ei", or "lcb"
            filename (str): Path to the file where the plot is to be saved.
        """
        gpr = self._fit_model()

        # Bounds
        bounds = self.skopt_search_space.transformed_bounds
        x0 = np.array([b[0] for b in bounds])
        x1 = np.array([b[1] for b in bounds])

        # Points
        x,y = load_past_experiment_results(self.cls, self.directory, self.search_space, self.score_fn, normalized=False)
        x = self.skopt_search_space.transform(x)
        x_proj = project_point(x,x0,x1)
        if show_points:
            plt.scatter(x_proj,y)

        # Gaussian Process
        x_plot = np.linspace(0,1,1000)
        x = np.linspace(x0,x1,1000)

        # Acquisition function
        if acq_func == 'ei':
            from skopt.acquisition import gaussian_ei
            y = gaussian_ei(x,gpr)
            plt.plot(x_plot,y,color='red', label='EI Acquisition Function')
        elif acq_func == 'lcb':
            from skopt.acquisition import gaussian_lcb
            y = gaussian_lcb(x,gpr)
            plt.plot(x_plot,y,color='red', label='LCB Acquisition Function')

        # mean
        prediction = gpr.predict(x, return_std=True)
        y = prediction[0]
        std = prediction[1]
        if show_gp_std:
            plt.fill_between(x_plot,y-std,y+std, alpha=0.2)
        if show_gp_mean:
            plt.plot(x_plot,y)

        # Save
        plt.savefig(filename)
        print('File saved at %s' % os.path.abspath(filename))
