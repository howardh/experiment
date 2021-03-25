import collections
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import make_interp_spline

##################################################
# Extract data
##################################################

def get_xy_data(logger, key):
    x = []
    y = []
    for i,v in enumerate(logger.data):
        if key in v:
            y.append(v[key])
            x.append(i)
    return x,y

##################################################
# Smooth data
##################################################

from abc import ABC, abstractmethod
class SeriesTransform(ABC):
    """ Transform series data. """
    @abstractmethod
    def __call__(self,x,y):
        pass

class GaussianSmoothing(SeriesTransform):
    def __init__(self,sigma=2):
        self.sigma=sigma
    def __call__(self,x,y):
        sigma = self.sigma
        ysmoothed = gaussian_filter1d(y, sigma=sigma)
        return x,ysmoothed

class EMASmoothing(SeriesTransform):
    def __init__(self,weight=0.9,points=300):
        self.points = points
        self.weight=weight
    def __call__(self,x,y):
        weight = self.weight
        ysmoothed = []
        cur = y[0]
        ysmoothed.append(cur)
        for val in y[1:]:
            cur = (1-weight)*cur + weight*val
            ysmoothed.append(cur)
        return x,ysmoothed

class SplineSmoothing(SeriesTransform):
    def __init__(self,k=3,points=300):
        self.points = points
        self.k = k
    def __call__(self,x,y):
        xnew = np.linspace(x.min(), x.max(), self.points) # Resample
        spl = make_interp_spline(x, y, k=self.k)
        ysmoothed = spl(snew)
        return x,ysmoothed

class LinearInterpResample(SeriesTransform):
    def __init__(self, points=300):
        self.points = points
    def __call__(self,x,y):
        if len(x) == 0:
            return x,y
        xnew = np.linspace(x[0], x[-1], self.points) # Resample
        if len(x) == 1:
            ynew = np.ones_like(xnew)*y[0]
        else:
            ynew = np.empty_like(xnew)
            j = 0 # index on original data
            for i,xval in enumerate(xnew):
                while xval > x[j+1]:
                    j+=1
                slope = (y[j+1]-y[j])/(x[j+1]-x[j])
                ynew[i] = y[j]+slope*(xval-x[j])
        return xnew,ynew

class ComposeTransforms(SeriesTransform):
    def __init__(self, *transforms):
        self.transforms = transforms
    def __call__(self,x,y):
        for f in self.transforms:
            x,y = f(x,y)
        return x,y

##################################################
# Plot data
##################################################

def plot(logger, curves, filename, min_points=3):
    """
    Args:
        logger (experiment.logger.Logger): `experiment.logger.Logger` object containing the data to be plotted.
        curves: A list that can contain either
            - The names of the keys of the `logger` data to be plotted in the y axis
            - Mappings where each element contains information for a single curve. The mapping can contain the following:
                - key: Key of the value to plot in the y axis
                - smooth_fn: A `SmoothingFunction` to apply to the data
        filename (str): Where to save the plot image.
        min_points (int): Minimum number of points to be plotted. If fewer data points are available, then do nothing.
    """
    if not isinstance(curves, collections.abc.Iterable):
        raise Exception('`curves` must be an iterable.')
    if len(curves) == 0:
        plt.figure()
        plt.savefig(filename)

    elem = curves[0]
    if isinstance(elem, collections.abc.Mapping):
        plt.figure()
        has_labels = False
        for curve in curves:
            # Get data
            key = curve.get('key')
            x,y = get_xy_data(logger, key)
            # Smooth data
            smooth_fn = curve.get('smooth_fn', lambda x,y: (x,y))
            x,y = smooth_fn(x,y)
            # Plot data
            label = curve.get('label')
            if label is not None:
                has_labels = True
            plt.plot(x,y,label=label)
        if has_labels:
            plt.legend(loc='best')
        plt.grid()
        plt.savefig(filename)
    elif isinstance(elem, str):
        plt.figure()
        for k in keys:
            x,y = get_xy_data(logger, k)
            plt.plot(x,y)
        plt.savefig(filename)
    plt.close()
