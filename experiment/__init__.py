import os
import time
import dill
import itertools
import pprint
from tqdm import tqdm
from abc import ABC, abstractmethod

def find_next_free_file(prefix, suffix, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    while True:
        for i in itertools.count():
            path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
            if not os.path.isfile(path):
                break
        # Create the file to avoid a race condition.
        # Will give an error if the file already exists.
        try:
            f = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(f)
        except FileExistsError as e:
            # Trying to create a file that already exists.
            # Try a new file name
            continue
        # Success
        break
    return path

class Experiment(ABC):
    @abstractmethod
    def setup(self, config):
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
            results_directory='./results', epoch=50, max_iterations=None, verbose=False,
            checkpoint_file_path=None, checkpoint_directory='./checkpoints', checkpoint_id=None,
            checkpoint_frequency=10000,
            config={}):
        """
        Args:
            results_directory:
                Directory in which results of completed runs are saved.
            max_iterations:
                The maximum number of iterations to run. If `None`, then there is no limit.
            checkpoint_file_path:
                Full path to the file in which the checkpoint is saved. Overrides `checkpoint_directory` and `checkpoint_id` if set.
            checkpoint_directory:
                Directory in which checkpoints are saved.
            checkpoint_id: str
                An ID associated with this run. If an existing checkpoint is found with the given ID, then that checkpoint is loaded instead of starting a new run.
            checkpoint_frequency:
                Number of steps between each saved checkpoint.
        """
        kwargs = locals()
        del kwargs['self']
        self.args = kwargs

        self.config = kwargs.get('config')
        self.results_directory = kwargs.get('results_directory')
        self.epoch = kwargs.get('epoch')
        self.verbose = kwargs.get('verbose')
        self.checkpoint_file_path = kwargs.get('checkpoint_file_path')
        self.checkpoint_directory = kwargs.get('checkpoint_directory')
        self.checkpoint_id = kwargs.get('checkpoint_id')
        self.checkpoint_frequency = kwargs.get('checkpoint_frequency')
        self.max_iterations = kwargs.get('max_iterations')

        if self.checkpoint_file_path is None:
            if self.checkpoint_id is not None:
                self.checkpoint_file_path = os.path.join(self.checkpoint_directory, 'checkpoint-%s.pkl'%self.checkpoint_id)
            else:
                t = time.strftime("%Y_%m_%d-%H_%M_%S")
                self.checkpoint_file_path = find_next_free_file('checkpoint-%s'%t,'pkl',self.checkpoint_directory)

        self.steps = 0
        if self.max_iterations is None:
            self.step_range = itertools.count(self.steps)
        else:
            self.step_range = range(self.steps,self.max_iterations)

        cls = kwargs['cls']
        self.exp = cls()
        self.exp.setup(self.config)

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
        try:
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
            # Move checkpoint to results directory
            results_file_path = find_next_free_file('checkpoint','pkl',self.results_directory)
            os.replace(self.checkpoint_file_path, results_file_path)
        except KeyboardInterrupt:
            pass

    def save_checkpoint(self, filename):
        results = self.state_dict()
        with open(filename,'wb') as f:
            dill.dump(results,f)
        if self.verbose:
            tqdm.write('Checkpoint saved at %s' % filename)

    def state_dict(self):
        output = {
            'args': {
                **self.args,
                'cls': str(self.args['cls']),
                'checkpoint_file_path': self.checkpoint_file_path,
            },
            'steps': self.steps,
            'exp': self.exp.state_dict()
        }
        return output

    def load_state_dict(self, state):
        # Experiment progress
        self.setup(**state.get('args'))
        self.exp.load_state_dict(state.get('exp'))

def load_checkpoint(cls, file_name):
    if os.path.isfile(file_name):
        with open(file_name,'rb') as f:
            state = dill.load(f)
    else:
        raise Exception('Checkpoint does not exist: %s' % file_name)

    # Check that the given class matches with the checkpoint experiment class
    cls_name = state['args'].get('cls')
    if cls_name != str(cls):
        raise Exception('Experiment class mismatch. Expected %s. Found %s.' % (cls_name,cls))
    state['args']['cls'] = cls

    # Create experiment runner with
    exp = ExperimentRunner(**state['args'])
    exp.load_state_dict(state)
    return exp
