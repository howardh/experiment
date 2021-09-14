import numpy as np
from typing import List, Mapping, Union, Tuple, Callable
from typing_extensions import TypedDict
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import make_interp_spline

from experiment.logger import Logger

##################################################
# Smooth data
##################################################

from abc import ABC, abstractmethod
class SeriesTransform(ABC):
    """ Transform series data. """
    @abstractmethod
    def __call__(self, x : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        pass

class GaussianSmoothing(SeriesTransform):
    def __init__(self,sigma=2):
        self.sigma=sigma
    def __call__(self, x : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        sigma = self.sigma
        ysmoothed = gaussian_filter1d(y, sigma=sigma, output=np.float32)
        return x,ysmoothed

class EMASmoothing(SeriesTransform):
    def __init__(self,weight=0.9,points=300):
        warnings.warn('I haven\'t figured out how I want EMA smoothing to behave yet. Expect this to change.')
        self.points = points
        self.weight=weight
    def __call__(self, x : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        weight = self.weight
        ysmoothed = []
        cur = y[0]
        ysmoothed.append(cur)
        for val in y[1:]:
            cur = (1-weight)*cur + weight*val
            ysmoothed.append(cur)
        return np.array(x),np.array(ysmoothed)

class SplineSmoothing(SeriesTransform):
    def __init__(self,k : int = 3, points : int = 300):
        """
        Args:
            k: B-spline degree. Default is cubic, k=3
        """
        self.points = points
        self.k = k
    def __call__(self, x : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        if len(x) < self.k:
            raise Exception('Must have at least k+1 points to interpolate. Found %d <= %d points.' % (len(x),self.k))
        spl = make_interp_spline(x, y, k=self.k)
        xnew = np.linspace(x.min(), x.max(), self.points) # Resample
        ysmoothed = spl(xnew)
        return xnew,ysmoothed

class LinearInterpResample(SeriesTransform):
    def __init__(self, points=300):
        self.points = points
    def __call__(self, x : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
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
    def __call__(self, x : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        for f in self.transforms:
            x,y = f(x,y)
        return x,y

##################################################
# Plot data
##################################################

class Curve(TypedDict, total=False):
    key : str
    label : str
    smooth_fn : Callable[[List[float],List[float]],Tuple[List[float],List[float]]]

def plot(logger : Logger, curves : List[Union[str,Curve]], filename : str, min_points : int = 3,
        xlabel : str = None, ylabel : str = None, title : str = None, yscale : str = 'linear'):
    """
    Args:
        logger (experiment.logger.Logger): `experiment.logger.Logger` object containing the data to be plotted.
        curves: A list that can contain either
            - The names of the keys of the `logger` data to be plotted in the y axis
            - Mappings where each element contains information for a single curve. The mapping can contain the following:
                - key: Key of the value to plot in the y axis
                - label: A string to identify this curve in the plot's legend
                - smooth_fn: A `SmoothingFunction` to apply to the data
        filename (str): Where to save the plot image.
        min_points (int): Minimum number of points to be plotted. If fewer data points are available, then do nothing.
    """
    if not isinstance(curves, list):
        raise Exception('`curves` must be a list.')
    if len(curves) == 0:
        return

    elem = curves[0]
    if isinstance(elem, Mapping):
        plt.figure()
        has_labels = False
        for i,curve in enumerate(curves):
            # Type checking
            if not isinstance(curve,Mapping):
                raise Exception('Expected a list of mappings. Found element of type %s at index %d.' % (type(curve),i))
            # Get data
            key = curve.get('key')
            if not isinstance(key,str):
                raise Exception('Expected a string as key. Found element of type %s.' % type(key))
            x,y = logger[key]
            # Check number of data points
            if len(x) < min_points:
                continue
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
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale(yscale)
        plt.savefig(filename)
        plt.close()
    elif isinstance(elem, str):
        plt.figure()
        for i,k in enumerate(curves):
            if not isinstance(k,str):
                raise Exception('Expected a list of strings. Found element of type %s at index %d.' % (type(k),i))
            x,y = logger[k]
            if len(x) < min_points:
                continue
            plt.plot(x,y)
        plt.grid()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale(yscale)
        plt.savefig(filename)
        plt.close()
