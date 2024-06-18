# Adapted from Huang et al., 2022
# https://github.com/YijinHuang/pytorch-classification/blob/master/data/transforms.py

from typing import Any, Dict
from torchvision import transforms


def get_transforms(img_size: int = 512, split: str = "train"):
    """Data Transformations

    Args:
        img_size (int): image size (width -or- height)
        split (str): train, val, validation, or test

    Returns:
        array: array of transforms to be applied to transforms.Compose()
    """

    if split != "train":
        transformations = [
            transforms.ToTensor(),
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
        ]

        return add_normalization(transformations)

    aug_args: Dict[str, Any] = {
        "random_crop": {"scale": [0.85, 1.15], "ratio": [0.75, 1.25], "p": 1.0},
        "flip": {"p": 0.5},
        "color_jitter": {
            "p": 0.5,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0,
            "hue": 0,
        },
        "rotation": {"p": 1.0, "degrees": [-180, 180]},
    }

    operations = {
        "random_crop": random_apply(
            transforms.RandomResizedCrop(
                size=(img_size, img_size),
                scale=aug_args["random_crop"]["scale"],
                ratio=aug_args["random_crop"]["ratio"],
                antialias=True,
            ),
            p=aug_args["random_crop"]["p"],
        ),
        "horizontal_flip": transforms.RandomHorizontalFlip(p=aug_args["flip"]["p"]),
        "vertical_flip": transforms.RandomVerticalFlip(p=aug_args["flip"]["p"]),
        "color_jitter": random_apply(
            transforms.ColorJitter(
                brightness=aug_args["color_jitter"]["brightness"],
                contrast=aug_args["color_jitter"]["contrast"],
                saturation=aug_args["color_jitter"]["saturation"],
                hue=aug_args["color_jitter"]["hue"],
            ),
            p=aug_args["color_jitter"]["p"],
        ),
        "rotation": random_apply(
            transforms.RandomRotation(degrees=aug_args["rotation"]["degrees"], fill=0),
            p=aug_args["rotation"]["p"],
        ),
    }

    augmentations = [transforms.ToTensor()]
    for operation in operations.items():
        augmentations.append(operation[1])

    return add_normalization(augmentations)


def random_apply(op, p):
    """Randomly apply a transformation"""
    return transforms.RandomApply([op], p=p)


def add_normalization(transformations: list):
    """Add normalization transformation using AREDS dataset wide mean and std after 
    resizing and center cropping."""

    normalization = transforms.Normalize(
        mean=[0.5723, 0.3242, 0.1655], std=[0.3039, 0.1998, 0.1466]
    )
    transformations.append(normalization)

    return transformations
