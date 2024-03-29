"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn

from ..registry import register_model

__all__ = ['vgg11','vgg16']

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        'M',
    ],
    'VGG19': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        512,
        'M',
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, MC=False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.MCDROPOUT = MC

    def forward(self, x):
        out = self.features(x)
        feat = out.view(out.size(0), -1)
        if self.MCDROPOUT:
            feat = self.dropout(feat)
        out = self.classifier(feat)
        return out,feat

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


@register_model
def vgg11(num_classes=10,MC=False):
    return VGG('VGG11', num_classes=num_classes,MC=MC)

@register_model
def vgg16(num_classes=10,MC=False):
    return VGG('VGG16', num_classes=num_classes,MC=MC)


def test():
    net = VGG('VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


# test()
