# Dataloader
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
from torchvision import transforms

# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")
from .cnn_surv_areds_dataset import AredsSurvivalDataset
from .cnn_transforms import get_transforms

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")
from utils.helpers import get_areds_data_dir, get_project_dir

# DATA_DIR = get_areds_data_dir()
PROJECT_DIR = Path(get_project_dir())
DATA_DIR = Path(get_areds_data_dir())

def get_train_loader_surv(c: SimpleNamespace):
    """Get train dataloader

    Args:
        c (SimpleNamespace of dict): config

    Returns:
        train_loader (torch.utils.data.DataLoader)
    """

    train_set = get_dataset(split="train", c=c)

    if c.cnn.test_run["enabled"]:
        train_set = get_subset(train_set, size=c.cnn.test_run["size"])
    elif hasattr(c.cnn, "train_set_fraction") and c.cnn.train_set_fraction < 1.0:
        train_set = get_subset(train_set, fraction=c.cnn.train_set_fraction)
        print(f"Using a {c.cnn.train_set_fraction*100}% fraction of the training data.")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=c.cnn.batch_size,
        shuffle=False if c.cnn.balancing else True,
        num_workers=c.cnn.num_workers,
        drop_last=True,
    )

    # print(
    #     "The number of images in the training set is: ",
    #     len(train_loader) * c.cnn["batch_size"],
    # )

    return train_loader


def get_val_loader_surv(c: SimpleNamespace):
    """Get train dataloader

    Args:
        c (SimpleNamespace of dict): config

    Returns:
        train_loader (torch.utils.data.DataLoader)
    """

    val_set = get_dataset(split="val", c=c)

    if c.cnn.test_run["enabled"]:
        val_set = get_subset(val_set, size=c.cnn.test_run["size"])

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=c.cnn.batch_size,
        shuffle=False,
        num_workers=c.cnn.num_workers,
        drop_last=True,
    )

    # print(
    #     "The number of images in the validation set is: ",
    #     len(val_loader) * c.cnn["batch_size"],
    # )

    return val_loader


def get_test_loader_surv(c: SimpleNamespace):
    """Get test dataloader

    Args:
        c (SimpleNamespace of dict): config

    Returns:
        test_loader (torch.utils.data.DataLoader)
    """

    test_set = get_dataset(split="test", c=c)

    if c.cnn.test_run["enabled"]:
        test_set = get_subset(test_set, size=c.cnn.test_run["size"])

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=c.cnn.batch_size,
        shuffle=False,
        num_workers=c.cnn.num_workers,
        drop_last=True,
    )

    # print(
    #     "The number of images in the test set is: ",
    #     len(test_loader) * c.cnn["batch_size"],
    # )

    return test_loader


def get_subset(set_: AredsSurvivalDataset, size: int = None, fraction: float = None):
    """If wanted, get a subset of the data for testing purposes or to run with a fraction of the data"""
    # Take a subset of the data (before optional balancing though, for testing purposes)
    if fraction is not None:
        size = int(fraction * len(set_))

    set_ = torch.utils.data.Subset(set_, range(0, size))

    return set_


def get_dataset(split: str, c: SimpleNamespace):
    """Get dataset of a given split. Note: If running a classification model, i.e. one that is
    trained for one prediction year x, this function excludes certain images from the training set
    that are invalid for this use case."""

    if c.cnn.augmentation and split == "train":
        transformations = get_transforms(img_size=c.cnn.img_size, split="validation")
    else:
        transformations = get_transforms(img_size=c.cnn.img_size, split=split)

    ## If using a classification approach like Yan 2020, i.e. when training one model for each
    # inquired year x, we need to exclude images that have no event and a relative
    # duration to censoring < x 
    if split == "train" and any([loss in c.cnn.loss.lower() for loss in ["clf", "celoss", "classification", "crossentropy"]]):
        assert len(c.cnn.survival_times) == 1, "When using a classification loss, only one survival time / evaluation time can be specified."
        clf_x = c.cnn.survival_times[0]   
    else: 
        clf_x = None   

    dataset = AredsSurvivalDataset(
        img_dir=DATA_DIR.joinpath(c.image_dir),
        metadata_csv=PROJECT_DIR.joinpath(c.metadata_csv),
        split=split,
        transform=transforms.Compose(transformations),
        exclude_imgs_for_classification_at=clf_x,
        use_stereo_pairs=c.cnn.use_stereo_pairs if hasattr(c.cnn, "use_stereo_pairs") else False,
    )

    return dataset
