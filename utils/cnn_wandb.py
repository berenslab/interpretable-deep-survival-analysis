import sys
import os
from types import SimpleNamespace
from typing import Union
import json

import wandb

# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")
from .helpers import get_config

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")
# from model.cnn import CNN

def get_runid_and_config_from_run(run, local_config_path: str = None):
    """ Loads run_id and config from file and wandb's run object, e.g. for complete
        config retrival after a sweep. 

    Args:
        run (wandb run object): wandb run object
        local_config_path (str, optional): Path to local config file. Defaults to None.

    Returns:
        run_id (str): run_id
        config (SimpleNamespace): config where config.cnn is the run's wandb config
    """
    if local_config_path is None:
        meta = json.load(run.file("wandb-metadata.json").download(f"meta_.json"))
        config_path = meta["args"][1]
        os.remove(f"meta_.json/wandb-metadata.json")
        os.rmdir(f"meta_.json")
    else:
        config_path = local_config_path

    # Overwrite c.cnn with wandb's config as it is the complete one
    # in order to not change a thing, e.g. in case of a sweep config where the tuned parameter is not set in the config
    c = get_config(config_path)
    c.cnn = SimpleNamespace(**run.config)

    return run.id, c

def get_run_id(c: SimpleNamespace) -> str:
    """Get run_id from config or create a new one using wandb

    Args:
        c (SimpleNamespace of dict): Config or config.cnn

    Returns:
        str: run_id
    """
    if "cnn" in vars(c):
        c = c.cnn

    if isinstance(c, dict):
        c = SimpleNamespace(**c)

    run_id = (
        c.run_id
        if "run_id" in vars(c) and c.run_id is not None
        else wandb.util.generate_id()
    )
    
    return run_id

