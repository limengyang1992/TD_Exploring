import numpy as np
import torch
from einops import rearrange
from torch import einsum, nn
from torchvision.models import resnet18,vgg16,swin_v2_t
from ..registry import register_model

__all__ = ['swin_224']


class Swin(nn.Module):
    def __init__(self, num_classes=10,MC=False):
        super().__init__()
        net = swin_v2_t(pretrained=True)
        self.net = torch.nn.Sequential( *( list(net.children())[:-1] ) )
        self.linear = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        self.dropout = nn.Dropout(0.25)
        self.MCDROPOUT = MC

    def forward(self, x):
        feat = torch.flatten(self.net(x), 1)
        if self.MCDROPOUT:
            feat = self.dropout(feat)
        out = self.linear(feat)
        return out,feat
    
@register_model
def swin_224(num_classes,MC):
    return Swin(num_classes=num_classes,MC=MC)



if __name__ == '__main__':
    m = swin_s224(num_classes=10)
    x = torch.randn(5, 3, 32, 32)

    print(m(x).shape)
