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
    assert logger[-2] == {'score': 1}
    assert len(logger) == 2

    logger.log(score=1.3)
    assert logger[-1] == {'score': 1.3}
    assert logger[-2] == {'score': 1.2}
    assert logger[-3] == {'score': 1}
    assert len(logger) == 3

def test_log_data_two_values():
    logger = Logger()
    assert len(logger) == 0

    logger.log(score=1, result=2)
    assert logger[-1] == {'score': 1, 'result': 2}
    assert len(logger) == 1

    logger.log(score=1.2, result=5)
    assert logger[-1] == {'score': 1.2, 'result': 5}
    assert logger[-2] == {'score': 1, 'result': 2}
    assert len(logger) == 2

    logger.log(score=1.3, result=3)
    assert logger[-1] == {'score': 1.3, 'result': 3}
    assert logger[-2] == {'score': 1.2, 'result': 5}
    assert logger[-3] == {'score': 1, 'result': 2}
    assert len(logger) == 3

def test_log_list_data():
    logger = Logger(key_name='time')
    assert len(logger) == 0

    logger.append(time=0, score=1, result=2)
    assert logger[-1] == {'time': 0, 'score': [1], 'result': [2]}
    assert len(logger) == 1

    logger.append(time=0, score=1.2, result=5)
    assert logger[-1] == {'time': 0, 'score': [1,1.2], 'result': [2,5]}
    assert len(logger) == 1

    logger.append(time=0, score=1.3, result=3)
    assert logger[-1] == {'time': 0, 'score': [1,1.2,1.3], 'result': [2,5,3]}
    assert len(logger) == 1

    logger.append(time=1, score=1.3, result=3)
    assert logger[-1] == {'time': 1, 'score': [1.3], 'result': [3]}
    assert logger[-2] == {'time': 0, 'score': [1,1.2,1.3], 'result': [2,5,3]}
    assert len(logger) == 2

def test_repeat_key_error():
    logger = Logger(key_name='time')
    logger.log(time=0, score=1, result=2)
    with pytest.raises(Exception):
        logger.log(time=0, score=1, result=3)

def test_repeat_key_with_overwrite():
    logger = Logger(key_name='time', overwrite=True)

    logger.log(time=0, score=1, result=2)
    assert logger[-1] == {'time': 0, 'score': 1, 'result': 2}
    assert len(logger) == 1

    logger.log(time=0, score=1, result=3)
    assert logger[-1] == {'time': 0, 'score': 1, 'result': 3}
    assert len(logger) == 1

def test_log_data_implicit_key():
    logger = Logger(key_name='key', allow_implicit_key=True)
    assert len(logger) == 0

    # Key with data
    logger.log(key=0, score=1)
    assert logger[-1] == {'key': 0, 'score': 1}
    assert len(logger) == 1

    logger.log(foo=1.2)
    assert logger[-1] == {'key': 0, 'score': 1, 'foo': 1.2}
    assert len(logger) == 1

    logger.log(key=1, score=1.3)
    assert logger[-1] == {'key': 1, 'score': 1.3}
    assert logger[-2] == {'key': 0, 'score': 1, 'foo': 1.2}
    assert len(logger) == 2

    # Key separately from data
    logger.log(key=2)
    logger.log(score=1.4)
    assert logger[-1] == {'key': 2, 'score': 1.4}
    assert logger[-2] == {'key': 1, 'score': 1.3}
    assert logger[-3] == {'key': 0, 'score': 1, 'foo': 1.2}
    assert len(logger) == 3
