class Logger:
    def __init__(self, manual_iteration=False):
        self.manual_iteration = manual_iteration
        self.data = []
        self.iter = 0

    def next_iteration(self):
        if not self.manual_iteration:
            raise Exception('`manual_iteration` has to be set to `True` to use next_iteration.')
        self.iter += 1

    def log(self,**data):
        if self.iter == len(self.data):
            self.data.append(data)
            if not self.manual_iteration:
                self.iter += 1
        else:
            for k,v in data.items():
                if k in self.data[self.iter]:
                    raise Exception('Key "%s" already exists for this iteration.' % k)
                self.data[self.iter][k] = v

    def get_plot_data(self, key):
        x = []
        y = []
        for i,v in enumerate(self.data):
            if key in v:
                y.append(v[key])
                x.append(i)
        return x,y
