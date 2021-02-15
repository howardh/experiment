import pytest
import numpy as np

from experiment import Experiment, ExperimentRunner, load_checkpoint
from experiment.logger import Logger

def test_find_next_free_file_concurrency(tmpdir):
    # TODO: Use multiprocessing to create 100 files, and check that there are 100 files in this directory.
    pass

class DummyExp(Experiment):
    def setup(self, config):
        self.logger = Logger()
        self.rng = np.random.default_rng()
    def run_step(self,iteration):
        self.logger.log(val=self.rng.random())
    def state_dict(self):
        return { 'rng': self.rng.bit_generator.state, 'logger': self.logger.state_dict() }
    def load_state_dict(self,state):
        self.rng.bit_generator.state = state.get('rng')

class DummyExpInterrupt(DummyExp):
    def run_step(self,iteration):
        super().run_step(iteration)
        if iteration >= 3:
            raise KeyboardInterrupt()

def test_experiment_runs_without_error():
    exp = ExperimentRunner(DummyExp, max_iterations=10)
    exp.run()

def test_checkpoint_created(tmpdir):
    """ If an experiment is interrupted, then the checkpoint should still be in the checkpoint directory. """
    checkpoint_dir = tmpdir.mkdir('checkpoints')
    results_dir = tmpdir.mkdir('results')
    assert len(checkpoint_dir.listdir()) == 0
    assert len(results_dir.listdir()) == 0

    exp = ExperimentRunner(DummyExpInterrupt, max_iterations=10, checkpoint_directory=checkpoint_dir, results_directory=results_dir)
    exp.run()

    assert len(results_dir.listdir()) == 0
    assert len(checkpoint_dir.listdir()) == 1

def test_results_saved(tmpdir):
    """ In an experiment finishes, then the results should be in the results directory. """
    checkpoint_dir = tmpdir.mkdir('checkpoints')
    results_dir = tmpdir.mkdir('results')
    assert len(checkpoint_dir.listdir()) == 0
    assert len(results_dir.listdir()) == 0

    exp = ExperimentRunner(DummyExp, max_iterations=10, checkpoint_directory=checkpoint_dir, results_directory=results_dir)
    exp.run()

    assert len(results_dir.listdir()) == 1
    assert len(checkpoint_dir.listdir()) == 0

def test_load_checkpoint(tmpdir):
    checkpoint_dir = tmpdir.mkdir('checkpoints')
    results_dir = tmpdir.mkdir('results')

    exp = ExperimentRunner(DummyExpInterrupt, max_iterations=10, checkpoint_directory=checkpoint_dir, results_directory=results_dir)
    exp.run()

    assert len(results_dir.listdir()) == 0
    assert len(checkpoint_dir.listdir()) == 1

    exp2 = load_checkpoint(DummyExpInterrupt, checkpoint_dir.listdir()[0])
    exp2.run()

    assert exp2.state_dict() == exp.state_dict()

    assert len(results_dir.listdir()) == 0
    assert len(checkpoint_dir.listdir()) == 1
