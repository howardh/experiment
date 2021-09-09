import pytest

from experiment.logger import Logger

def test_wandb_log():
    wandb_params = {
            'project': 'logger-test',
    }
    logger = Logger(wandb_params=wandb_params)

    assert logger._wandb_run is not None

    val= [.1, .4, .5, .3, .4, .8, .5, .9, .8, .9]
    for i in range(10):
        logger.log(step=i, val=val[i])
        assert logger._wandb_run.summary['val'] == val[i]

@pytest.mark.skip(reason='Looks like W&B doesn\'t like lists?')
def test_wandb_log_list():
    wandb_params = {
            'project': 'logger-test',
    }
    logger = Logger(wandb_params=wandb_params)

    assert logger._wandb_run is not None

    val = [.1, .4, .5, .3, .4, .8, .5, .9, .8, .9]
    for i in range(10):
        logger.log(step=i, list_val=val)
        assert logger._wandb_run.summary['val'] == val
