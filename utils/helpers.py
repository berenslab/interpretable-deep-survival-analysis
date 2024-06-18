import os
from typing import Tuple, Union
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import json
import yaml

import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")


class Parser(ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def init_sweep_id(c: SimpleNamespace) -> str:
    """Create new wandb sweep and return sweep_id"""
    import wandb

    conf = c.wandb_sweep
    proj = c.cnn.project if hasattr(c, "cnn") else c.project
    sweep_id = wandb.sweep(conf, project=proj)
    print("Sweep ID: ", sweep_id)
    return sweep_id

def get_config(config: Union[str, dict]):
    """Get config dict from config file (yaml or json), stringified dict, or dict

    Args:
        config (str, dict): config file name or dict or stringified dict

    Returns:
        config dict as SimpleNamespace
    """

    # Check if config is a stringified dict
    if isinstance(config, str) and "{" in config:
        config = json.loads(config)

        # Replace all stringified booleans by booleans in the config dict
        for key in config:
            val = str(config[key]).lower()
            if val in ["true", "false"]:
                config[key] = [True if val == "true" else False][0]

    # ... or the path to a config file
    elif isinstance(config, str):
        if config.startswith("/"):
            if config.endswith(".yaml") or config.endswith(".yml"):
                config = yaml.safe_load(open(config))
            elif config.endswith(".json"):
                config = json.load(open(config))
        else:
            # relative path or filename given, load from config folder
            config = load_config_file(config)

    elif isinstance(config, dict):
        pass

    else:
        raise TypeError(
            "Config must be a stringified dict or a path to a yaml or json file"
        )

    c = SimpleNamespace(**config)

    if hasattr(c, "cnn"):
        c.cnn = SimpleNamespace(**c.cnn)

    return c


def load_config_file(dir: str) -> dict:
    """Load config file from json or yaml file

    Args:
        path (str): Path to config starting from <project root>/models/config

    Returns:
        config (dict)
    """
    if "configs/" in dir:
        dir = dir.split("configs/")[1]
        
    path = os.path.join(get_project_dir(), "configs")

    if dir == "":
        print(
            "You have to specify a config file name using --config=<name.{json, yaml}>: \n"
        )
        only_files = [
            f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
        ]
        print(only_files)
        print("Exiting...")
        exit(1)

    # Read config dict from file
    if dir.endswith(".yaml") or dir.endswith(".yml"):
        config = yaml.safe_load(open(os.path.join(path, dir)))
    elif dir.endswith(".json"):
        config = json.load(open(os.path.join(path, dir)))

    return config


def set_seed(seed: int = 12345, silent=False) -> None:
    """Set seed for reproducibility."""
    try:
        import numpy as np

        np.random.seed(seed)
        if not silent:
            print(f"Numpy random seed was set as {seed}")
    except ImportError:
        print("Could not set seed for numpy: Numpy not found.")

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not silent:
            print(f"Torch random seed was set as {seed}")
    except ImportError:
        print("Could not set seed for torch: Torch not found.")

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if not silent:
        print(f"PYTHONHASHSEED was set as {seed}")
        print()


def arg_to_bool(arg: str) -> bool:
    """Converts string to bool

    Args:
        arg (str): string

    Returns:
        bool
    """
    arg = str(arg).lower()
    if arg in ["true", "t", "1"]:
        return True
    elif arg in ["false", "f", "0"]:
        return False
    else:
        raise Exception(f"Invalid argument {arg} for bool conversion")


def exists(paths: list):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


# Helper function to reduce list depth by 1
def flatten_one(l):
    """Flatten one level of nesting"""
    return [item for sublist in l for item in sublist]


def parse_list(arg: Union[list, str]):
    if isinstance(arg, list):
        return arg
    elif (
        isinstance(arg, np.ndarray)
        or isinstance(arg, tuple)
        or isinstance(arg, set)
        or isinstance(arg, pd.Series)
    ):
        return list(arg)
    else:
        try:
            arg = str(arg)
            arg = arg.replace(" ", "").replace("[", "").replace("]", "")
            if "," in arg:
                return arg.split(",")
            else:
                # not a list
                return arg
        except Exception as e:
            raise f"Could not parse {arg} of type {type(arg)} as a list: " + str(e)


def on_slurm():
    if "SLURM_JOB_ID" in os.environ:
        return True
    else:
        return False
    

def get_project_dir(path: str = "") -> str:
    """Get project directory.

    Reads project paths from yaml file and returns the first path that exists and contains a
    "models" subfolder.

    Args:
        path (str, optional): An absolute path to the git project directory

    Returns:
        path (str)

    Raises:
        Exception if not found
    """

    yaml_path = os.path.join(
        os.path.dirname(__file__), os.path.pardir, "configs", "dirs.yml"
    )

    with open(yaml_path) as f:
        yml = yaml.safe_load(f)

    paths = yml["project_dirs"]

    if path != "":
        paths.insert(0, path)
    for path in paths:
        if exists([path, os.path.join(path, "configs")]) and exists([path, os.path.join(path, "model")]):
            return path
    raise Exception(f"Areds git project directory not found among {paths}")


def get_areds_data_dir(scratch=True):

    yaml_path = os.path.join(
        os.path.dirname(__file__), os.path.pardir, "configs", "dirs.yml"
    )
    
    with open(yaml_path) as f:
        yml = yaml.safe_load(f)

    paths = yml["data_dirs"]

    def _is_slurm():
        return Path('/mnt/qb/areds').exists()
    
    if _is_slurm():
        # Use scratch path on slurm, if available
        areds_scratch_path = os.path.join(os.environ["SCRATCH"], "areds")
        if os.path.exists(areds_scratch_path) and scratch:
            data_path = areds_scratch_path
            return Path(data_path)
        else:
            return Path('/mnt/qb/areds')
    else:
        for path in paths:
            if os.path.exists(os.path.join(path, 'data_processed')):
                return Path(path)
        raise Exception("No data path found")
