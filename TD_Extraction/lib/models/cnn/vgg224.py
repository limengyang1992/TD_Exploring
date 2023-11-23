"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn
from torchvision.models import resnet18,vgg16_bn,swin_s

from ..registry import register_model

__all__ = ['vgg16_224']


class VGG(nn.Module):
    def __init__(self, num_classes=10,MC=False):
        super().__init__()
        net = vgg16_bn(pretrained=False)
        self.net = torch.nn.Sequential( *( list(net.children())[:-1]))
        self.head = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
            )
        self.linear = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        self.dropout = nn.Dropout(0.25)
        self.MCDROPOUT = MC

    def forward(self, x):
        feat1 = torch.flatten(self.net(x), 1)
        feat = self.head(feat1)
        if self.MCDROPOUT:
            feat = self.dropout(feat)
        out = self.linear(feat)
        return out,feat
    
@register_model
def vgg16_224(num_classes=10,MC=False):
    return VGG(num_classes=num_classes,MC=MC)