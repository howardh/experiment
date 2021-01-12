import os
import dill
import itertools

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

class Experiment:
    def __init__(self, directory='./results', epoch=50, max_iterations=None,
            verbose=False, keep_checkpoint=False, checkpoint_path=None,
            checkpoint_frequency=10000, config={}):
        """
        Args:
            max_iterations:
                The maximum number of iterations to run. If `None`, then there is no limit.
            checkpoint_frequency:
                Number of steps between each saved checkpoint.
        """
        self.args = locals()
        del self.args['self']

        self.directory = directory
        self.epoch = epoch
        self.verbose = verbose
        self.keep_checkpoint = keep_checkpoint
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency

        self.results_file_path = find_next_free_file('checkpoint','pkl',self.directory)

        self.state = {}

        self.steps = 0
        if max_iterations is None:
            self.step_range = itertools.count()
        else:
            self.step_range = range(max_iterations)

        self.setup(config)

    def setup(self, config):
        pass

    def before_epoch(self, iteration):
        pass

    def after_epoch(self, iteration):
        pass

    def run_step(self, iteration):
        raise NotImplementedError('`run_step` should be overridden.')

    def run(self):
        if self.verbose:
            pprint.pprint(self.args)
            step_range = tqdm(self.step_range)
        else:
            step_range = self.step_range

        try:
            for self.steps in step_range:
                if self.steps % self.checkpoint_frequency == 0:
                    self.save_checkpoint()
                if self.steps % self.epoch == 0:
                    self.before_epoch(self.steps)
                self.run_step(self.steps)
                if self.steps % self.epoch == 0:
                    self.after_epoch(self.steps)
            self.save_results()
        except KeyboardInterrupt:
            pass

    def save_checkpoint(self):
        results = self.state_dict()
        with open(self.results_file_path,'wb') as f:
            dill.dump(results,f)

    def state_dict(self):
        output = {
            'args': self.args,
            'steps': self.steps,
        }
        for k,v in self.state.items():
            output[k] = v
        return output

    def load_state_dict(self, state):
        # Experiment progress
        state.pop('args')
        self.steps = state.pop('steps')

        if self.max_iterations is None:
            self.step_range = itertools.count(self.steps)
        else:
            self.step_range = range(self.steps,self.max_iterations)

        for k,v in state.items():
            self.state[k] = v

    @staticmethod
    def from_checkpoint(file_name):
        if os.path.isfile(file_name):
            with open(file_name,'rb') as f:
                state = dill.load(f)
        else:
            raise Exception('Checkpoint does not exist: %s' % file_name)

        config = state['args'].pop('config')
        exp = Experiment(**state['args'],**config)
        exp.load_state_dict(state)
        return exp
