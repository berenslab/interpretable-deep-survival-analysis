import json
import os
from types import SimpleNamespace
from typing import Tuple, Union
import torch
import yaml
import sys

# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")


def get_short_model_name(config: dict) -> str:

    name = "_".join(get_long_model_name(config).split("_")[1:])
    name = name.replace("efficientnet", "eff")
    name = name.replace("resnet", "res")
    name = name.replace("accuracy", "acc")
    name = name.replace("crossentropy_loss", "loss")
    name = name.replace(".pth", "")
    if "ID-" in name:
        name = name.split("ID-")[0]

    return name


def get_long_model_name(config: SimpleNamespace) -> str:
    """Get model name

        Returns model name for conventional save & load procedures:
            <project>_<network>_<metric>_<(un)bal>_<(no)aug>
            Additionally,
                _w for weight_decay (AdamW)
                _#classes for non-12-class models
                _<img_size> for non-512 image sizes
                _testrun
            are appended if applicaple.

    Args:
        config (SimpleNamespace of dict): config

    Returns:
        name.pth (str)
    """

    name = "_".join(
        [
            str(config.cnn.project).replace("_", "-"),
            str(config.cnn.network).replace("_", "-"),
            str(config.cnn.model_selection_metric).replace("_", "-"),
            "bal" if config.cnn.balancing else "unbal",
            "aug" if config.cnn.augmentation else "noaug",
        ]
    )

    optimizer = "_w" if config.cnn.weight_decay > 0.0 else ""
    classes = (
        "_" + str(config.cnn.num_classes) + "classes"
        if config.cnn.num_classes != 12
        else ""
    )
    size = "_" + str(config.cnn.img_size) if config.cnn.img_size != 512 else ""
    test_run = "_testrun" if config.cnn.test_run["enabled"] else ""

    return name + optimizer + classes + size + test_run


def get_model_filename(config: SimpleNamespace, run_id: str = None) -> str:
    if run_id is None:
        run_id = config.cnn.run_id
    return f"{run_id}.pth"


def get_cuda(gpu: Union[str, int] = ""):
    """Returns cuda string for given GPU

        e.g. 'cuda:1' or 'cuda'

    Args:
        gpu (str, otional): gpu number(s)
    Returns:
        str
    """
    gpu = str(gpu)
    if gpu == "":
        return "cuda"
    else:
        return f"cuda:{gpu}"

def set_parameter_requires_grad(self, fine_tune: bool = True):
    if fine_tune:
        for param in self.parameters():
            param.requires_grad = True
    elif not fine_tune:
        for param in self.parameters():
            param.requires_grad = False


def optimizer_to_device(optim, device: str) -> None:
    """Move optimizer to gpu device

    Args:
        optim: Optimizer
        device (str): Device to move to

    Returns:
        None
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_lr(optimizer):
    for p in optimizer.param_groups:
        return p["lr"]
