import pytest

import wandb

from experiment.logger import Logger

#os.environ["WANDB_MODE"] = "dryrun" # Run W&B in offline mode

@pytest.fixture
def logger():
    wandb_params = {
            'project': 'logger-test',
    }
    logger = Logger(wandb_params=wandb_params)
    assert logger._wandb_run is not None

    yield logger

    logger._wandb_run.finish()

@pytest.fixture
def logger_with_impkey():
    wandb_params = {
            'project': 'logger-test',
    }
    logger = Logger(
            key_name='step',
            allow_implicit_key=True,
            wandb_params=wandb_params)
    assert logger._wandb_run is not None

    yield logger

    logger._wandb_run.finish()

def test_wandb_log(logger):
    val= [.1, .4, .5, .3, .4, .8, .5, .9, .8, .9]
    for i in range(10):
        logger.log(step=i, val=val[i])
        assert logger._wandb_run.summary['val'] == val[i]

def test_wandb_log_implicit_key(logger_with_impkey):
    logger = logger_with_impkey

    val1 = [.1, .4, .5, .3, .4, .8, .5, .9, .8, .9]
    val2 = [.5, .7, .3, .4, .2, .8, .2, .3, .1, .5]
    for i in range(10):
        logger.log(step=i)
        logger.log(val1=val1[i])
        logger.log(val2=val2[i])
    logger._wandb_run.finish()
    
    api = wandb.Api()
    run = api.run(logger._wandb_run.path)
    assert [x['val1'] for x in run.history()] == val1
    assert [x['val2'] for x in run.history()] == val2

def test_wandb_log_list(logger):
    val = [.1, .4, .5, .3, .4, .8, .5, .9, .8, .9]
    for i in range(10):
        logger.log(step=i, list_val=val)
        assert logger._wandb_run.summary['list_val'] == val
