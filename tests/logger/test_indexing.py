import pytest

from experiment.logger import Logger

def test_empty_logger():
    logger = Logger()

    with pytest.raises(Exception):
        logger[0]

    with pytest.raises(Exception):
        logger[1]

    with pytest.raises(Exception):
        logger[-1]

def test_one_element():
    logger = Logger()
    logger.log(score=1)

    assert logger[0] == logger[-1]
    assert logger[0] == {'score': 1}

    with pytest.raises(Exception):
        logger[1]

def test_index_by_key():
    logger = Logger(key_name='step')
    logger.log(step=0,score=1)

    assert logger['score'] == ([0],[1])

    logger.log(step=1,score=5)
    assert logger['score'] == ([0,1],[1,5])

    logger.log(step=2,score=4,other='b')
    assert logger['score'] == ([0,1,2],[1,5,4])
    assert logger['other'] == ([2],['b'])

def test_index_by_slice():
    logger = Logger()
    logger.log(score=1)

    assert logger[:] == [{'score': 1}]

    logger.log(score=5)
    assert logger[:] == [{'score': 1}, {'score': 5}]
    assert logger[1:] == [{'score': 5}]
