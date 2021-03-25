import pytest
import numpy as np

from experiment import Experiment, ExperimentRunner, load_checkpoint
from experiment.logger import Logger

def test_find_next_free_file_concurrency(tmpdir):
    # TODO: Use multiprocessing to create 100 files, and check that there are 100 files in this directory.
    pass

class DummyExp(Experiment):
    def setup(self, config, output_directory=None):
        print(config)
        self.logger = Logger()
        self.rng = np.random.default_rng()
    def run_step(self,iteration):
        self.logger.log(val=self.rng.random())
    def state_dict(self):
        return {
                'rng': self.rng.bit_generator.state,
                'logger': self.logger.state_dict()
        }
    def load_state_dict(self,state):
        self.rng.bit_generator.state = state['rng']
        self.logger.load_state_dict(state['logger'])

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
    root_dir = tmpdir.mkdir('results')
    assert len(root_dir.listdir()) == 0

    exp = ExperimentRunner(DummyExpInterrupt, max_iterations=10, root_directory=root_dir)
    try:
        exp.run()
    except KeyboardInterrupt:
        pass

    assert len(root_dir.listdir()) == 1

def test_results_saved(tmpdir):
    """ In an experiment finishes, then the results should be in the results directory. """
    root_dir = tmpdir.mkdir('results')
    assert len(root_dir.listdir()) == 0

    exp = ExperimentRunner(DummyExp, max_iterations=10, root_directory=root_dir)
    exp.run()

    assert len(root_dir.listdir()) == 1

def test_load_checkpoint(tmpdir):
    root_dir = tmpdir.mkdir('results')
    assert len(root_dir.listdir()) == 0

    exp = ExperimentRunner(
            DummyExp, max_iterations=10, root_directory=root_dir
    )
    exp.run()

    assert len(root_dir.listdir()) == 1, 'Checkpoint was not created'

    exp2 = load_checkpoint(DummyExp, root_dir.listdir()[0])
    exp2.run() # This should do nothing, since the experiment was already run

    assert exp2.state_dict() == exp.state_dict()
    assert len(root_dir.listdir()) == 1
