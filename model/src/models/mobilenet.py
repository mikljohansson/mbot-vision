import math

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Sequential):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__(*args)

    def forward(self, x):
        return x + super().forward(x)


class UpsampleInterpolate2d(nn.Module):
    def __init__(self):
        super(UpsampleInterpolate2d, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class SegmentationHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.upsample = nn.Sequential(
            UpsampleInterpolate2d(),
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            UpsampleInterpolate2d(),
            nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.fuse(x)
        return x

class MobileNetModel(nn.Module):
    """
    Based on MobileNet v3 with an image segmentation head
    """
    def __init__(self, config):
        super().__init__()

        self.backbone = config.backbone
        del self.backbone.avgpool
        del self.backbone.classifier

        self.head = SegmentationHead(config.backbone_out_ch)
        self.out = nn.Identity()

        self.mean2x = torch.nn.Parameter(2. * torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1), requires_grad=False)

    def forward(self, x):
        # Normalize image values and convert to [-1, 1] range inside the network, to simplify deployment
        #image = (image - self.mean) / self.std
        #image = (image - 0.5) / 0.5
        # https://www.wolframalpha.com/input?i=simplify+%28%28x+-+m%29+%2F+s+-+0.5%29+%2F+0.5
        x = (2. * x - self.mean2x) / self.std - 1.

        x = self.backbone.features(x)
        x = self.head(x)
        x = self.out(x)
        return x

    def deploy(self):
        # Add the final sigmoid directly into the model
        self.out = nn.Sigmoid()
