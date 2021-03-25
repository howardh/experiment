import os
import time
import dill
import itertools
import pprint
from tqdm import tqdm
from abc import ABC, abstractmethod

from .utils import find_next_free_dir, find_next_free_file

class Experiment(ABC):
    @abstractmethod
    def setup(self, config, output_directory=None):
        pass
    @abstractmethod
    def run_step(self,iteration):
        pass

    def state_dict(self):
        return {}
    def load_state_dict(self,state):
        pass

    def before_epoch(self,iteration):
        pass
    def after_epoch(self,iteration):
        pass

class ExperimentRunner:
    def __init__(self, cls,
            experiment_name=None,
            root_directory='./results',
            trial_id=None, results_directory=None,
            epoch=50, max_iterations=None, verbose=False,
            checkpoint_frequency=10000,
            config={}):
        """
        Args:
            cls (Experiment): The class defining the experiment to run.
            root_directory (str): If specified, then a directory is created in this directory for all files generated by this experiment.
            experiment_name (str): Name given to the experiment, to be used in naming the directory in which results are saved. Defaults to the name of the Experiment class provided.
            results_directory (str): Directory which holds all files generated by the current experiment trial. If not specified, then a new directory is created in the root directory for this purpose.
                If an existing directory is specified, then it is loaded as a checkpoint.
            trial_id (str): An ID associated with this trial. If an existing checkpoint is found with the given ID in the root directory, then that checkpoint is loaded instead of starting a new run.
            epoch (int): Number of steps in an epoch
            max_iterations (int): The maximum number of iterations to run. If `None`, then there is no limit.
            checkpoint_frequency (int): Number of steps between each saved checkpoint.
            config (collections.abc.Mapping): Parameters that are passed to the experiment's `setup` method.
        """
        kwargs = locals()
        del kwargs['self']
        self.args = kwargs

        self.config = config
        self.experiment_name = experiment_name or cls.__name__
        self.root_directory = root_directory
        self.results_directory = results_directory
        self.epoch = epoch
        self.verbose = verbose
        self.trial_id = trial_id
        self.checkpoint_frequency = checkpoint_frequency
        self.max_iterations = max_iterations

        if self.results_directory is None:
            if self.trial_id is None:
                self.trial_id = time.strftime("%Y_%m_%d-%H_%M_%S")
            self.results_directory = find_next_free_dir(
                    self.root_directory,
                    '{}-{}-%d'.format(self.experiment_name, self.trial_id)
            )
        self.checkpoint_file_path = os.path.join(
                self.results_directory,'checkpoint.pkl')
        self.experiment_output_directory = os.path.join(
                self.results_directory, 'output')
        os.makedirs(self.experiment_output_directory, exist_ok=True)

        self.steps = 0
        if self.max_iterations is None:
            self.step_range = itertools.count(self.steps)
        else:
            self.step_range = range(self.steps,self.max_iterations)

        cls = kwargs['cls']
        self.exp = cls()
        self.exp.setup(
                config=self.config,
                output_directory=self.experiment_output_directory
        )

    def setup(self, **kwargs):
        pass

    def run(self):
        step_range = self.step_range
        if self.verbose:
            pprint.pprint(self.args)
            if self.max_iterations is None:
                step_range = tqdm(step_range)
            else:
                step_range = tqdm(step_range, total=self.max_iterations, initial=self.steps)
        for steps in step_range:
            self.steps = steps
            if steps % self.checkpoint_frequency == 0:
                self.save_checkpoint(self.checkpoint_file_path)
            if steps % self.epoch == 0:
                self.exp.before_epoch(steps)
            self.exp.run_step(steps)
            if steps % self.epoch == 0:
                self.exp.after_epoch(steps)
        # Save checkpoint
        self.save_checkpoint(self.checkpoint_file_path)

    def save_checkpoint(self, filename):
        results = self.state_dict()
        with open(filename,'wb') as f:
            dill.dump(results,f)
        if self.verbose:
            tqdm.write('Checkpoint saved at %s' % os.path.abspath(filename))

    def state_dict(self):
        output = {
            'args': {
                **self.args,
                'cls': str(self.args['cls']),
                'results_directory': self.results_directory,
            },
            'steps': self.steps,
            'exp': self.exp.state_dict()
        }
        return output

    def load_state_dict(self, state):
        # Experiment progress
        self.setup(**state.get('args'))
        self.exp.load_state_dict(state.get('exp'))

        self.steps = state['steps']
        if self.max_iterations is None:
            self.step_range = itertools.count(self.steps)
        else:
            self.step_range = range(self.steps+1,self.max_iterations)


def load_checkpoint(cls, path):
    if os.path.isfile(path):
        with open(path,'rb') as f:
            state = dill.load(f)
    if os.path.isdir(path):
        with open(os.path.join(path,'checkpoint.pkl'),'rb') as f:
            state = dill.load(f)
    else:
        raise Exception('Checkpoint does not exist: %s' % path)

    # Check that the given class matches with the checkpoint experiment class
    # Or it it is a child class of the original
    cls_name = state['args'].get('cls')
    parent_cls_names = (str(parent_cls_name) for parent_cls_name in cls.__bases__)
    if cls_name != str(cls) and cls_name not in parent_cls_names:
        raise Exception('Experiment class mismatch. Expected %s. Found %s.' % (cls_name,cls))
    state['args']['cls'] = cls

    # Create experiment runner with
    exp = ExperimentRunner(**state['args'])
    exp.load_state_dict(state)
    return exp
