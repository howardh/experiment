import pytest

from experiment.logger import Logger

def test_log_data():
    logger = Logger()
    assert len(logger) == 0

    logger.log(score=1)
    assert logger[-1] == {'score': 1}
    assert len(logger) == 1

    logger.log(score=1.2)
    assert logger[-1] == {'score': 1.2}
    assert len(logger) == 2

    logger.log(score=1.3)
    assert logger[-1] == {'score': 1.3}
    assert len(logger) == 3

def test_log_data_two_values():
    logger = Logger()
    assert len(logger) == 0

    logger.log(score=1, result=2)
    assert logger[-1] == {'score': 1, 'result': 2}
    assert len(logger) == 1

    logger.log(score=1.2, result=5)
    assert logger[-1] == {'score': 1.2, 'result': 5}
    assert len(logger) == 2

    logger.log(score=1.3, result=3)
    assert logger[-1] == {'score': 1.3, 'result': 3}
    assert len(logger) == 3
