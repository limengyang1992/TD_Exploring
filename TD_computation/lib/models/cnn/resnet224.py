'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from ..registry import register_model

__all__ = ['ResNet18_224']


class ResNet(nn.Module):
    def __init__(self, num_classes=10,MC=False,pretrain=True):
        super().__init__()
        net = resnet18(pretrained=True)
        self.net = torch.nn.Sequential( *( list(net.children())[:-1] ) )
        self.linear = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.dropout = nn.Dropout(0.25)
        self.MCDROPOUT = MC

    def forward(self, x):
        feat = torch.flatten(self.net(x), 1)
        if self.MCDROPOUT:
            feat = self.dropout(feat)
        out = self.linear(feat)
        return out,feat
    
@register_model
def ResNet18_224(num_classes=10,MC=False,pretrain=False):
    return ResNet(num_classes=num_classes,MC=MC,pretrain=pretrain)