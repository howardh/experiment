import pytest
import os
import numpy as np

from experiment import Experiment
from experiment.logger import Logger
from experiment.hyperparam import GridSearch, RandomSearch, BayesianOptimizationSearch, GaussianProcessAnalysis
from experiment.hyperparam import Categorical, Uniform, IntUniform, LogUniform
from experiment.hyperparam.search import project_point

EPS = 1e-6

class DummyExp(Experiment):
    def setup(self, config, output_directory=None):
        print(config)
        assert config['const'] == 1
        assert 0 <= config['uniform'] <= 1
        assert 0 <= config['int_uniform'] <= 2
        assert 1e-3-EPS <= config['log_uniform'] <= 1e-1+EPS
        assert config['cat'] in ['a','b','c']
        self.mean = np.sqrt(
                (config['uniform']-0.5)**2 + 
                (config['int_uniform']-1)**2 +
                np.exp(np.log(config['log_uniform'])-np.log(1e-2))**2
        )
        self.logger = Logger()
    def run_step(self,iteration):
        self.logger.log(val=self.mean)
    def state_dict(self):
        return {
                'logger': self.logger.state_dict()
        }
    def load_state_dict(self,state):
        self.logger.load_state_dict(state['logger'])

search_space = {
        'const': 1,
        'uniform': Uniform(0,1,3),
        'int_uniform': IntUniform(0,2),
        'log_uniform': LogUniform(1e-3,1e-1,3),
        'cat': Categorical(['a','b','c']),
}

def test_gridsearch(tmpdir):
    root_dir = tmpdir.mkdir('results')
    assert len(os.listdir(root_dir)) == 0

    search = GridSearch(DummyExp, search_space,
            root_directory=root_dir,
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
            checkpoint_frequency=100,
            max_iterations=10,
            search_budget=10
    )
    search.run()

    assert len(os.listdir(root_dir)) == 1
    assert len(os.listdir(os.path.join(search.directory,'Experiments'))) == 10

def test_bo_search_and_analysis_match(tmpdir):
    root_dir = tmpdir.mkdir('results')
    assert len(os.listdir(root_dir)) == 0

    search = BayesianOptimizationSearch(DummyExp, search_space,
            score_fn=lambda exp: exp.logger.mean('val'),
            root_directory=root_dir,
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

    # Check that we have the same model in the search and the analysis
    config = search.expected_min[0]
    search_score = search.results.models[-1].predict(search.results['space'].transform([config]))
    analysis_score = analysis.compute_scores([config], normalized=False)
    assert analysis_score == pytest.approx(search_score)

def test_bo_search_add_budget(tmpdir):
    root_dir = tmpdir.mkdir('results')
    assert len(os.listdir(root_dir)) == 0

    search = BayesianOptimizationSearch(DummyExp, search_space,
            score_fn=lambda exp: exp.logger.mean('val'),
            root_directory=root_dir,
            checkpoint_frequency=100,
            max_iterations=10,
            search_budget=10
    )
    search.run()

    assert len(os.listdir(root_dir)) == 1
    assert len(os.listdir(os.path.join(search.directory,'Experiments'))) == 10

    search = BayesianOptimizationSearch(DummyExp, search_space,
            score_fn=lambda exp: exp.logger.mean('val'),
            output_directory=search.directory,
            checkpoint_frequency=100,
            max_iterations=10,
            search_budget=10
    )
    search.run()

    assert len(os.listdir(root_dir)) == 1
    assert len(os.listdir(os.path.join(search.directory,'Experiments'))) == 20

def test_project_point():
    x0 = np.array([0,0])
    x1 = np.array([1,1])
    x = np.array([0.5,0.5])
    x_proj = project_point(x,x0,x1)
    assert pytest.approx(x_proj.item()) == 0.5

def test_vector_to_config_dict():
    pass
