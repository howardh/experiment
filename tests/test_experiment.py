import os

from experiment.experiment import sub_env_var
import numpy as np

from experiment import Experiment, ExperimentRunner, load_checkpoint, make_experiment_runner
from experiment.logger import Logger

##################################################
# Test Utils
##################################################

class Callback:
    def __init__(self):
        self.steps_run = []
    def __call__(self, step):
        self.steps_run.append(step)
        print(self.steps_run)

class DummyExp(Experiment):
    def setup(self, config, output_directory=None):
        print(config)
        self.logger = Logger()
        self.rng = np.random.default_rng(seed=config.get('seed'))
        self.output_directory = output_directory
        self._interrupt_at = config.get('interrupt_at')
        self._run_step_callback = config.get('run_step_callback', lambda _: None)
    def run_step(self,iteration):
        print('`run_step(%d)`' % iteration)
        self._run_step_callback(iteration)
        self.logger.log(val=self.rng.random())
        print(self.logger.data)
        if self._interrupt_at is not None and iteration == self._interrupt_at:
            raise KeyboardInterrupt()
    def state_dict(self):
        return {
                'rng': self.rng.bit_generator.state,
                'logger': self.logger.state_dict()
        }
    def load_state_dict(self,state):
        self.rng.bit_generator.state = state['rng']
        self.logger.load_state_dict(state['logger'])

##################################################
# sub_env_var
##################################################

def test_sub_env_var_none_input():
    assert sub_env_var(None) is None

def test_sub_env_var_no_subs():
    assert sub_env_var('asdf') == 'asdf'

def test_sub_env_var_not_found():
    assert sub_env_var('asdf {$THING_THAT_DOESNT_EXIST}') == 'asdf {$THING_THAT_DOESNT_EXIST}'

def test_sub_env_var_single_var():
    os.environ['FOO'] = 'boop'
    assert sub_env_var('asdf {$FOO}') == 'asdf boop'

def test_sub_env_var_two_vars():
    os.environ['FOO'] = 'boop'
    os.environ['BAR'] = 'beep'
    assert sub_env_var('asdf {$FOO}{$BAR}') == 'asdf boopbeep'

##################################################
# Experiment Runner
##################################################

def test_experiment_runs_without_error():
    exp = ExperimentRunner(DummyExp, max_iterations=10)
    exp.run()

def test_experiment_number_of_steps():
    exp_runner = ExperimentRunner(DummyExp, max_iterations=2)
    exp_runner.run()
    assert len(exp_runner.exp.logger.data) == 2

def test_checkpoint_created(tmpdir):
    """ If an experiment is interrupted, then the checkpoint should still be in the checkpoint directory. """
    root_dir = tmpdir.mkdir('results')
    assert len(root_dir.listdir()) == 0

    exp = ExperimentRunner(DummyExp, max_iterations=10,
            root_directory=str(root_dir),
            config={
                'interrupt_at': 3
            })
    try:
        exp.run()
    except KeyboardInterrupt:
        pass

    assert len(root_dir.listdir()) == 1

def test_results_saved(tmpdir):
    """ In an experiment finishes, then the results should be in the results directory. """
    root_dir = tmpdir.mkdir('results')
    assert len(root_dir.listdir()) == 0

    exp = ExperimentRunner(DummyExp, max_iterations=10, root_directory=str(root_dir))
    exp.run()

    assert len(root_dir.listdir()) == 1

def test_load_checkpoint(tmpdir):
    """
    Initialize an experiment and save a checkpoint at the start. Initialize a second experiment and load that first checkpoint. Run both and check that the state of both experiments are the same at the end.

    Note: A checkpoint is saved at the end of an experiment regardless of the value of `checkpoint_frequency`.
    """

    root_dir = tmpdir.mkdir('results')
    assert len(root_dir.listdir()) == 0

    exp = ExperimentRunner(
            DummyExp, max_iterations=10, root_directory=str(root_dir)
    )
    exp.run()

    assert len(root_dir.listdir()) == 1, 'Checkpoint was not created'

    exp2 = load_checkpoint(DummyExp, root_dir.listdir()[0])
    exp2.run() # This should do nothing, since the experiment was already run

    assert exp2.state_dict() == exp.state_dict()
    assert len(root_dir.listdir()) == 1

def test_same_id_loads_checkpoint(tmpdir):
    """ Verify that running an interrupted experiment with the same `trial_id` will load the checkpoint and continue from there. """
    root_dir = tmpdir.mkdir('results')
    assert len(root_dir.listdir()) == 0

    run_step_callback = Callback()

    exp_runner = make_experiment_runner(
            DummyExp,
            trial_id='dummy',
            max_iterations=2,
            root_directory=str(root_dir),
            checkpoint_frequency=10,
            config={
                'seed': 0,
                'interrupt_at': 0,
                'run_step_callback': run_step_callback,
            }
    )
    try:
        exp_runner.run()
    except KeyboardInterrupt:
        pass

    assert len(root_dir.listdir()) == 1
    assert run_step_callback.steps_run == [0]

    exp_runner = make_experiment_runner(
            DummyExp,
            trial_id='dummy',
            max_iterations=2,
            root_directory=str(root_dir),
            checkpoint_frequency=10,
            config={}
    )
    exp_runner.exp._interrupt_at = None
    exp_runner.exp._run_step_callback = run_step_callback

    assert exp_runner.exp.logger.data == [], 'Expected no data logged.'

    try:
        exp_runner.run()
    except KeyboardInterrupt:
        assert False, 'This should not happen'

    assert len(root_dir.listdir()) == 1, 'A new experiment was generated rather than loading the existing experiment.'

    assert run_step_callback.steps_run == [0,0,1], 'Experiment should\'ve restarted at the initial checkpoint and rerun every step of the experiment.'

    # Run again with the same seed, and we should get the same result
    exp_runner2 = make_experiment_runner(
            DummyExp,
            max_iterations=2,
            checkpoint_frequency=10,
            config={
                'seed': 0,
            }
    )
    exp_runner2.run()

    # Compare results
    assert exp_runner.exp.logger.data == exp_runner2.exp.logger.data
