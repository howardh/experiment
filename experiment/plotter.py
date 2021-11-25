import numpy as np
from typing import List, Mapping, Union, Tuple, Callable
from typing_extensions import TypedDict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.axes
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
    def __init__(self,weight=0.9):
        self.weight=weight
    def __call__(self, x : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        weight = self.weight
        ysmoothed = []
        cur = y[0]
        ysmoothed.append(cur)
        for val in y[1:]:
            cur = weight*cur + (1-weight)*val
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
    smooth_fn : Callable[[np.ndarray,np.ndarray],Tuple[np.ndarray,np.ndarray]]

def plot(logger : Logger,
        curves : Union[List[str],List[Curve]],
        filename : str = None,
        min_points : int = 3,
        xlabel : str = None,
        ylabel : str = None,
        title : str = None,
        yscale : str = 'linear',
        aggregate : str = None,
        ax : matplotlib.axes.Axes = None,
        show_unaggregated : bool = True,
        show_unsmoothed : bool = True,
        colour_index = 0):
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
        aggregate (str): Aggregation method to use on the plotted data.

            - `None`: Plot the values as is with no aggregation.
            - "mean": Plot the mean of values in the list.

        ax (matplotlib.axes.Axes): Axis on which to render the plot. If `None`, a new axis will be created.
        colour_index (int): A number representing the colour to use for rendering this piece of data.
    """
    if not isinstance(curves, list):
        raise Exception('`curves` must be a list.')
    if len(curves) == 0:
        return
    if ax is None and filename is None:
        raise Exception('Either `filename` or `ax` must be provided.')

    # Type checking and consolidation
    normalized_curves = []
    for i,c in enumerate(curves):
        if isinstance(c, str):
            normalized_curves.append({'key': c})
        elif isinstance(c, Mapping):
            normalized_curves.append(c)
        else:
            raise Exception(f'Expected a list of mappings or strings. Found element of type {type(c)} at index {i}.')

    # Colours
    colour = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    # Plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = None

    has_labels = False
    for i,curve in enumerate(normalized_curves):
        # Get data
        key = curve.get('key')
        if not isinstance(key,str):
            raise Exception('Expected a string as key. Found element of type %s.' % type(key))
        x,y_raw = logger[key]
        # Check number of data points
        if len(x) < min_points:
            continue

        ####################
        # Aggregation

        # Plot unaggregated data
        if aggregate is not None and show_unaggregated:
            scatter_x = []
            scatter_y = []
            for a,b in zip(x,y_raw):
                for c in b:
                    scatter_x.append(a)
                    scatter_y.append(c)
            ax.scatter(scatter_x,scatter_y,c=colour[colour_index],alpha=0.3)
        # Aggregate
        if aggregate == 'mean':
            y = [float(np.mean(v)) for v in y_raw]
        else:
            y = y_raw

        ####################
        # Smoothing

        # Plot unsmoothed data
        if 'smooth_fn' in curve and show_unsmoothed:
            ax.plot(x,y,colour[colour_index],alpha=0.2)
        # Smooth data
        smooth_fn = curve.get('smooth_fn', lambda x,y: (x,y))
        x,y = smooth_fn(x,y)

        ####################
        # Final plot (processed data)

        label = curve.get('label')
        if label is not None:
            has_labels = True
        # Processed data
        ax.plot(x,y,colour[colour_index],label=label,alpha=1)

    if has_labels:
        ax.legend(loc='best')

    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if fig is not None:
        fig.savefig(filename)
        plt.close()

def stacked_area_plot(logger : Logger,
        key : str,
        filename : str = None,
        xlabel : str = None,
        ylabel : str = None,
        title : str = None,
        normalize : bool = True,
        ax : matplotlib.axes.Axes = None):
    if ax is None and filename is None:
        raise Exception('Either `filename` or `ax` must be provided.')

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = None

    data_x, data_y = logger[key]
    data_y = np.array(data_y)

    if normalize:
        data_y = data_y/data_y.sum(1,keepdims=True)

    plt.stackplot(data_x, *data_y.transpose())

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if fig is not None:
        fig.savefig(filename)
        plt.close()
