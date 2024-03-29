import pytest
import subprocess
import textwrap

import dill

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

def test_index_by_string_no_key():
    """ Access elements under 'train_score' and 'val_score' with no key set on the logger.
    """
    logger = Logger()
    logger.log(train_score=1, val_score=2)
    logger.log(train_score=2)
    logger.log(train_score=3)
    logger.log(train_score=1, val_score=3)
    logger.log(train_score=2)
    logger.log(train_score=3)

    output = logger['train_score']
    assert output is not None
    assert len(output) == 2
    x,y = output
    assert y == [1,2,3,1,2,3]
    assert x == [0,1,2,3,4,5]

    output = logger['val_score']
    assert output is not None
    assert len(output) == 2
    x,y = output
    assert y == [2,3]
    assert x == [0,3]

def test_index_by_string_with_key():
    logger = Logger(key_name='iteration')
    logger.log(iteration=3,train_score=1, val_score=2)
    logger.log(iteration=4,train_score=2)
    logger.log(iteration=5,train_score=3)
    logger.log(iteration=6,train_score=1, val_score=3)
    logger.log(iteration=7,train_score=2)
    logger.log(iteration=8,train_score=3)

    output = logger['train_score']
    assert output is not None
    assert len(output) == 2
    x,y = output
    assert y == [1,2,3,1,2,3]
    assert x == [3,4,5,6,7,8]

    output = logger['val_score']
    assert output is not None
    assert len(output) == 2
    x,y = output
    assert y == [2,3]
    assert x == [3,6]

def test_with_file_backed_data(tmpdir):
    # Create a logger with a file-backed data store in a separate process
    code = textwrap.dedent(f"""
        import os
        import experiment.logger
        import dill

        logger = experiment.logger.Logger(
            key_name='iteration',
            in_memory=False,
            filename=os.path.join('{tmpdir}','data.pkl'),
            max_file_length=3)
        logger.log(iteration=3,train_score=1, val_score=2)
        logger.log(iteration=4,train_score=2)
        logger.log(iteration=5,train_score=3)
        logger.log(iteration=6,train_score=1, val_score=3)
        logger.log(iteration=7,train_score=2)
        logger.log(iteration=8,train_score=3)

        with open(os.path.join('{tmpdir}','state.pkl'), 'wb') as f:
            logger = dill.dump(logger.state_dict(), f)
    """)
    output = subprocess.run(['python','-c',code], capture_output=True)
    if output.returncode != 0:
        raise Exception(output.stderr.decode('utf-8'))

    # Load the logger from file
    logger = Logger()
    with open(tmpdir.join('state.pkl'), 'rb') as f:
        logger.load_state_dict(dill.load(f))

    output = logger['train_score']
    assert output is not None
    assert len(output) == 2
    x,y = output
    assert y == [1,2,3,1,2,3]
    assert x == [3,4,5,6,7,8]

    output = logger['val_score']
    assert output is not None
    assert len(output) == 2
    x,y = output
    assert y == [2,3]
    assert x == [3,6]

def test_with_file_backed_data_2(tmpdir):
    # Create a logger with a file-backed data store in a separate process
    code = textwrap.dedent(f"""
        import os
        import experiment.logger
        import dill

        logger = experiment.logger.Logger(
            key_name='iteration',
            in_memory=False,
            filename=os.path.join('{tmpdir}','data.pkl'),
            max_file_length=3)
        logger.log(iteration=3,train_score=1, val_score=2)
        logger.log(iteration=4,train_score=2)
        logger.log(iteration=5,train_score=3)
        logger.log(iteration=6,train_score=4)
        logger.log(iteration=7,train_score=1, val_score=3)
        logger.log(iteration=8,train_score=2)
        logger.log(iteration=9,train_score=3)
        logger.log(iteration=10,train_score=4)

        with open(os.path.join('{tmpdir}','state.pkl'), 'wb') as f:
            logger = dill.dump(logger.state_dict(), f)
    """)
    output = subprocess.run(['python','-c',code], capture_output=True)
    if output.returncode != 0:
        raise Exception(output.stderr.decode('utf-8'))

    # Load the logger from file
    logger = Logger()
    with open(tmpdir.join('state.pkl'), 'rb') as f:
        logger.load_state_dict(dill.load(f))

    output = logger['train_score']
    assert output is not None
    assert len(output) == 2
    x,y = output
    assert y == [1,2,3,4,1,2,3,4]
    assert x == [3,4,5,6,7,8,9,10]

    output = logger['val_score']
    assert output is not None
    assert len(output) == 2
    x,y = output
    assert y == [2,3]
    assert x == [3,7]
