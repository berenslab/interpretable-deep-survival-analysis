import torch
from torchvision import transforms
from PIL import Image
import sys, os
import pandas as pd
import numpy as np

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")
from utils.helpers import set_seed

from data.cnn_transforms import get_transforms

def inference(df, c, cnn, device="cuda:0"):
    """Run inference on images in a dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with "image_paths" column
        c (Config): config file
        cnn (CNN): model, where cnn.model is the pytorch model
    """
    set_seed(c.cnn.seed, silent=True)  # Set seed for reproducibility

    if isinstance(df, pd.Series):
        df = df.to_frame().T

    cnn.model.eval()

    # Set size
    img_size = c.cnn.img_size
    img_size_hires = 2 * img_size

    # Get image
    images = [Image.open(image_path) for image_path in df["image_path"]]

    # Transform images for model input and for plotting
    transforms_normalized = transforms.Compose(get_transforms(img_size=img_size, split="test"))
    transforms_ = transforms.Compose(get_transforms(img_size=img_size, split="test")[:-1])
    transforms_hires = transforms.Compose(get_transforms(img_size=img_size_hires, split="test")[:-1])

    images_cropped = [transforms_(image) for image in images]
    images_cropped_hires = [transforms_hires(image) for image in images]
    images_normalized = [transforms_normalized(image) for image in images]
    images_normalized = torch.stack(images_normalized).float().to(device)

    # Get predictions and feature maps
    with torch.no_grad():
        logits, activations = cnn.model(images_normalized)
        survival_curves = cnn.get_survival_predictions(logits).cpu().numpy().tolist()

    return (
        logits,
        activations,
        survival_curves,
        images_cropped,
        images_cropped_hires,
        images_normalized,
        img_size,
        img_size_hires,
    )