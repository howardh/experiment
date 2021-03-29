import pytest
import os
import numpy as np

from experiment.hyperparam import Categorical, Uniform, IntUniform, LogUniform, LogIntUniform

##################################################
# Uniform
##################################################

def test_uniform_positive():
    dist = Uniform(3,4)
    for _ in range(10):
        assert 3 <= dist.sample() <= 4

def test_uniform_negative():
    dist = Uniform(-2,-1)
    for _ in range(10):
        assert -2 <= dist.sample() <= 1

def test_uniform_size_0():
    dist = Uniform(5,5)
    for _ in range(10):
        assert dist.sample() == 5

def test_uniform_linspace():
    dist = Uniform(3,5)
    assert (dist.linspace(2) == np.array([3,5])).all()
    assert (dist.linspace(3) == np.array([3,4,5])).all()
    dist = Uniform(0,1)
    assert (dist.linspace(2) == np.array([0,1])).all()
    assert (dist.linspace(3) == np.array([0,0.5,1])).all()

##################################################
# LogUniform
##################################################

def test_log_uniform():
    dist = LogUniform(1e-5, 1e-3)
    for _ in range(10):
        assert 1e-5 <= dist.sample() <= 1e-3

def test_log_uniform_negative():
    with pytest.raises(Exception):
        dist = LogUniform(-5, 3)

def test_log_uniform_size_0():
    dist = LogUniform(1e-3, 1e-3)
    assert pytest.approx(dist.sample(),1e-3)

##################################################
# IntUniform
##################################################

def test_int_uniform():
    dist = IntUniform(1,3)

    vals = set()
    for _ in range(100):
        vals.add(dist.sample())
        if len(vals) == 5:
            break
    assert vals == set([1,2,3])

    vals = set()
    for _ in range(100):
        vals.add(dist.skopt_space().rvs().item())
        if len(vals) == 5:
            break
    assert vals == set([1,2,3])

def test_int_uniform_size_0():
    dist = IntUniform(3,3)
    assert dist.sample() == 3

##################################################
# LogIntUniform
##################################################

def test_log_int_uniform():
    dist = LogIntUniform(1, 5)

    vals = set()
    for _ in range(100):
        vals.add(dist.sample())
        if len(vals) == 5:
            break
    assert vals == set([1,2,3,4,5])

    vals = set()
    for _ in range(100):
        vals.add(dist.skopt_space().rvs().item())
        if len(vals) == 5:
            break
    assert vals == set([1,2,3,4,5])

def test_log_int_uniform_size_0():
    dist = LogIntUniform(5, 5)
    assert dist.sample() == 5

def test_log_int_uniform_linspace():
    dist = LogIntUniform(1, 16)
    assert (dist.linspace(2) == np.array([1,16])).all()
    assert (dist.linspace(5) == np.array([1,2,4,8,16])).all()

    dist = LogIntUniform(1, 15)
    assert (dist.linspace(2) == np.array([1,15])).all()
    assert (dist.linspace(3) == np.array([1,3,15])).all() # I didn't manually check if this is correct. Current implementation gives this output, and it looks reasonable. Putting this here as a regression test.
