import pytest
import os

import numpy as np

from experiment.logger import Logger
from experiment.plotter import plot, LinearInterpResample, SplineSmoothing, EMASmoothing, GaussianSmoothing

##################################################
# Smooth data (Linear)
##################################################

def test_linear_interp():
    func = LinearInterpResample(3)
    x = np.array([0,2])
    y = np.array([1,3])
    x,y = func(x,y)
    assert (x == [0,1,2]).all()
    assert (y == [1,2,3]).all()

def test_linear_interp_nonuniform_input():
    func = LinearInterpResample(6)
    x = np.array([0,2,5])
    y = np.array([1,3,0])
    x,y = func(x,y)
    assert (x == [0,1,2,3,4,5]).all()
    assert (y == [1,2,3,2,1,0]).all()

def test_linear_interp_no_data():
    func = LinearInterpResample(6)
    x = np.array([])
    y = np.array([])
    x,y = func(x,y)
    assert (x == []).all()
    assert (y == []).all()

def test_linear_interp_1_point():
    func = LinearInterpResample(3)
    x = np.array([0])
    y = np.array([0])
    x,y = func(x,y)
    assert (x == [0,0,0]).all()
    assert (y == [0,0,0]).all()

def test_linear_interp_upsample_downsample_random_data():
    # Check that data is unchanged after upsampling and downsampling
    length = np.random.choice(10)+1
    upsample = LinearInterpResample(length*2-1)
    downsample = LinearInterpResample(length)
    x = np.arange(length)
    y = np.random.rand(length)
    print(length,x,y)
    new_x,new_y = upsample(x,y)
    assert len(new_x) == length*2-1
    assert len(new_y) == length*2-1
    new_x,new_y = downsample(new_x,new_y)
    assert (x == new_x).all()
    assert (y == new_y).all()

##################################################
# Smooth data (Spline)
##################################################

def test_quad_spline_interp_three_points():
    func = SplineSmoothing(k=2,points=4)
    x = np.array([0,2,3])
    y = np.array([1,3,2])
    x,y = func(x,y)
    assert (x == [0,1,2,3]).all()
    assert np.allclose(y,[1,8/3,3,2]) # Values taken from a spline interpolation calculator

def test_cubic_spline_interp_three_points():
    func = SplineSmoothing(k=3,points=4)
    x = np.array([0,2,3])
    y = np.array([1,3,2])
    with pytest.raises(Exception):
        x,y = func(x,y)

def test_cubic_spline_interp_four_points():
    func = SplineSmoothing(k=3,points=5)
    x = np.array([0,2,3,4])
    y = np.array([1,3,2,3])
    x,y = func(x,y)
    assert (x == [0,1,2,3,4]).all()
    #assert np.allclose(y,[1,?,3,2,3]) # TODO

##################################################
# Smooth data (Gaussian)
##################################################

def test_gaussian_smoothing_three_points_all_same():
    """ Check that it runs without errors """
    func = GaussianSmoothing(sigma=2)
    x = np.array([0,2,3])
    y = np.array([3,3,3])
    x,y = func(x,y)
    assert (x == [0,2,3]).all()
    assert (y == [3,3,3]).all()

def test_gaussian_smoothing_three_points():
    """ Check that it runs without errors """
    func = GaussianSmoothing(sigma=2)
    x = np.array([0,2,3])
    y = np.array([3,4,3])
    x,y = func(x,y)
    assert (x == [0,2,3]).all()
    assert y[0] > 3
    assert y[1] < 4
    assert y[2] > 3

##################################################
# Smooth data (EMA)
##################################################

def test_ema_smoothing_three_points():
    """ Check that it runs without errors """
    func = EMASmoothing(points=4)
    x = np.array([0,2,3])
    y = np.array([1,3,2])
    x,y = func(x,y)
    # TODO: Figure out how I want this to behave, then test it

##################################################
# Plot
##################################################

def test_plot_curves_key_only(tmpdir):
    """ Check that plotting runs without errors. """
    output_dir = tmpdir.mkdir('output')

    logger = Logger()
    logger.log(train_score=1, val_score=2)
    logger.log(train_score=2)
    logger.log(train_score=3)
    logger.log(train_score=1, val_score=3)
    logger.log(train_score=2)
    logger.log(train_score=3)

    filename=os.path.join(output_dir,'plot.png')
    plot(
        logger=logger,
        curves=['train_score','val_score'],
        filename=filename
    )
    assert os.path.isfile(filename)

def test_plot_curves_param(tmpdir):
    """ Check that plotting runs without errors. """
    output_dir = tmpdir.mkdir('output')

    logger = Logger()
    logger.log(train_score=1, val_score=2)
    logger.log(train_score=2)
    logger.log(train_score=3)
    logger.log(train_score=1, val_score=3)
    logger.log(train_score=2)
    logger.log(train_score=3)

    filename=os.path.join(output_dir,'plot.png')
    plot(
        logger=logger,
        curves=[
            {
                'key': 'train_score',
            },{
                'key': 'val_score',
            }
        ],
        filename=filename
    )
    assert os.path.isfile(filename)
