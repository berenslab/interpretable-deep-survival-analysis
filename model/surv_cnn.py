import sys
import os

import torch
from torch import nn
from torchvision import models

# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")
from .sparse_bagnet import create_sparse_bagnet

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")
from utils.cnn_utils import set_parameter_requires_grad

def create_survival_resnet50(num_times:int):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    res = models.resnet50(weights=weights)
    model = SurvivalCNN(res, num_times)
    set_parameter_requires_grad(model, fine_tune=True)

    return model

def create_survival_inceptionV3(num_times:int):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    inception = models.inception_v3(weights=weights)
    model = SurvivalCNN(inception, num_times)
    set_parameter_requires_grad(model, fine_tune=True)

    return model

def create_survival_sparsebagnet33(num_times:int):
    bagnet = create_sparse_bagnet(num_times)
    model = SurvivalCNN(bagnet, num_times)
    model.set_attribute("get_sparsity_loss", bagnet.get_sparsity_loss)
    set_parameter_requires_grad(model, fine_tune=True)

    return model

class SurvivalCNN(nn.Module):
    """CNN for survival prediction on one input image or a pair of input images."""

    def __init__(
        self, model, num_times=None
    ):
        super(SurvivalCNN, self).__init__()
        sequential = nn.Sequential(*model.children())

        # Determine features size: 
        #  find out if last layer is fc (ResNet, InceptionV3) or avgpool ((Sparse)Bagnet)
        if isinstance(sequential[-1], nn.Linear):
            size = sequential[-1].in_features
            model.fc = nn.Identity()
            self.cnn = model
            self.survival_head = nn.Linear(size, num_times)

        elif isinstance(sequential[-1], nn.Conv2d):
            self.cnn = sequential[:-1]
            self.conv1x1 = sequential[-1] # 1x1 conv layer
            self.survival_head = None

        else:
            raise NotImplementedError("Expecting a ResNet-50, Inception-v3 or (Sparse)Bagnet-33 model.")

    
    def set_attribute(self, attribute, value):
        setattr(self, attribute, value)
    
    
    def forward(self, x):   
        """Forward pass through the network. Accepts one or two input images. 
        For two images, they should be LS/RS stereo pairs and predictions should be averaged post
        softmax (Babenko et al. 2019)
        
        Args:
            x (torch.Tensor or torch.Tensor(torch.Tensor, torch.Tensor)): 
                input image or two input images (LS/RS stereo pair)

        Returns:
            if stereo pair: list of survival predictions and list of activations
            else: survival predictions and activations
        """

        if x.shape[1] == 2: # 0: batch-size, 1: channels(3) or stereo pair(2)
            stereo_pair = True
            x = x.unbind(dim=1) # Keep batch dim. but remove stereo pair dim.
        else:
            stereo_pair = False
            x = [x]

        survival_preds_list = []
        activations_list = []

        for i in range(len(x)):
            survival_preds, activations = self.one_forward_pass(x[i])
            survival_preds_list.append(survival_preds)
            activations_list.append(activations)
        
        if stereo_pair: # lists of length 2 with tensors of shape (bs, npreds)
            # Obtain prediction logits of shape (bs, 2, npreds)
            survival_preds_stereo = torch.stack(survival_preds_list).permute(1, 0, 2)

            # Average activation maps over stereo pair
            activations_mean = torch.stack(activations_list).mean(dim=0)

            return survival_preds_stereo, activations_mean
        
        else:
            return survival_preds_list[0], activations_list[0]

    def one_forward_pass(self, x):
        x = self.cnn(x)
        
        # Resnet, InceptionV3
        if isinstance(self.survival_head, nn.Linear):
            if "inception" in self.cnn.__class__.__name__.lower() and self.cnn.training:
                # Ignore auxiliary outputs that are passed during training
                x = x[0]
            activations = x.clone()
            x = x.view(x.size()[0], -1)
            survival_preds = self.survival_head(x)

        # (Sparse)Bagnet
        elif self.survival_head is None:
            n, m = x.shape[2], x.shape[3] # 28,28
            activations = self.conv1x1(x)

            # Spatial average pooling to get logits
            self.clf_avgpool = nn.AvgPool2d(kernel_size=(n, m), stride=(1, 1), padding=0)
            x = self.clf_avgpool(activations)
            survival_preds = x.view(x.shape[0], -1)


        return survival_preds, activations
        