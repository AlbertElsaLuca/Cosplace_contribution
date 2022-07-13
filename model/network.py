
import torch
import logging
import torchvision
from torch import nn
import torch.hub   #import the torch.hub in order to get the pretrained version of SE-resnet50

from model.layers import Flatten, L2Norm, GeM


CHANNELS_NUM_IN_LAST_CONV = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "vgg16": 512,
        "SE-resnet50": 2048      #add the resnet50 pretrained model 
    }


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim):
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_backbone(backbone_name):

    #instead of checking if backbone startwith, let's check if it contains "resnet"
    # backbone_name.startswith("resnet") ---> backbone_name.find("resnet")>0
    if backbone_name.find("resnet")>=0:
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone_name== "SE-resnet50":
            backbone = torch.hub.load('moskomule/senet.pytorch','se_resnet50', pretrained=True)
            #retrieve the pretrained version of SE-resnet from torch.hub 
            #we are using the SE-resnet50 because it's been the only one available pretrained on imageNet
            # we though to manually write the Squezze and excitation block into Resnet18 but it's not been straighforward.
            #for a sake of simplicity we decide to use The SE-resnet50. why did we mmodify the backbone is explained in the paper
    
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim

