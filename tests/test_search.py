import pytest
import os
import numpy as np

from experiment.hyperparam import GridSearch, RandomSearch, BayesianOptimizationSearch, GaussianProcessAnalysis
from experiment.hyperparam import Categorical, Uniform, IntUniform, LogUniform
from experiment.hyperparam.search import project_point

from .test_experiment import DummyExp

search_space = {
        'const': 1,
        'uniform': Uniform(0,1,3),
        'int_uniform': IntUniform(0,3),
        'log_uniform': LogUniform(1e-3,1e-1,3),
        'cat': Categorical(['a','b','c']),
}

search_space_continuous = {
        'const': 1,
        'uniform': Uniform(0,1,3),
        'int_uniform': IntUniform(0,3),
        'log_uniform': LogUniform(1e-3,1e-1,3),
}

def test_gridsearch(tmpdir):
    root_dir = tmpdir.mkdir('results')
    assert len(os.listdir(root_dir)) == 0

    search = GridSearch(DummyExp, search_space,
            root_directory=root_dir,
            epoch=3,
            checkpoint_frequency=100,
            max_iterations=10,
    )
    search.run()

    assert len(os.listdir(root_dir)) == 1
    assert len(os.listdir(os.path.join(search.directory,'Experiments'))) == 3*3*3*3

def test_random_search(tmpdir):
    root_dir = tmpdir.mkdir('results')
    assert len(os.listdir(root_dir)) == 0

    search = RandomSearch(DummyExp, search_space,
            root_directory=root_dir,
            epoch=3,
            checkpoint_frequency=100,
            max_iterations=10,
            search_budget=10
    )
    search.run()

    assert len(os.listdir(root_dir)) == 1
    assert len(os.listdir(os.path.join(search.directory,'Experiments'))) == 10

def test_bo_search_continuous(tmpdir):
    return
    root_dir = tmpdir.mkdir('results')
    assert len(os.listdir(root_dir)) == 0

    search = BayesianOptimizationSearch(DummyExp, search_space_continuous,
            score_fn=lambda exp: exp.logger.mean('val'),
            root_directory=root_dir,
            epoch=3,
            checkpoint_frequency=100,
            max_iterations=10,
            search_budget=10
    )
    search.run()

    assert len(os.listdir(root_dir)) == 1
    assert len(os.listdir(os.path.join(search.directory,'Experiments'))) == 10

def test_bo_search(tmpdir):
    search_space = {
            'const': 1,
            'uniform': Uniform(0,1,3),
            'int_uniform': IntUniform(0,3),
            'log_uniform': LogUniform(1e-3,1e-1,3),
            'cat': Categorical(['a','b','c']),
    }

    root_dir = tmpdir.mkdir('results')
    assert len(os.listdir(root_dir)) == 0

    search = BayesianOptimizationSearch(DummyExp, search_space,
            score_fn=lambda exp: exp.logger.mean('val'),
            root_directory=root_dir,
            epoch=3,
            checkpoint_frequency=100,
            max_iterations=10,
            search_budget=10
    )
    search.run()

    assert len(os.listdir(root_dir)) == 1
    assert len(os.listdir(os.path.join(search.directory,'Experiments'))) == 10

    analysis = GaussianProcessAnalysis(DummyExp,
            score_fn=lambda exp: exp.logger.mean('val'),
            search_space=search_space,
            directory=os.path.join(search.directory,'Experiments'))
    print(analysis.get_best_config())

def test_project_point():
    x0 = np.array([0,0])
    x1 = np.array([1,1])
    x = np.array([0.5,0.5])
    x_proj = project_point(x,x0,x1)
    assert pytest.approx(x_proj.item(), 0.5)
