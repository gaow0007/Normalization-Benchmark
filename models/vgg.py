import torch
import torch.nn as nn
from torch.autograd import Variable
import LayerNorm
import GroupNorm
import InstanceNorm

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_norm(num_features, norm='bn'):
    if norm == 'bn':
        return nn.BatchNorm2d(num_features)
    elif norm == 'ln':
        return LayerNorm(num_features)
    elif norm == 'in':
        return InstanceNorm(num_features)
    elif norm == 'gn':
        return GroupNorm(num_features)

class VGG(nn.Module):
    def __init__(self, vgg_name, norm='bn'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.norm = norm

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.norm is not None:
                    if self.norm != 'wn'
                        normlayer = make_norm(x, self.norm)
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                    normlayer,
                                    nn.ReLU(inplace=True)]
                    else:
                        normconv = Conv2dWD(in_channels, x, kernel_size, padding=1)
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), 
                                    nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), 
                                nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
