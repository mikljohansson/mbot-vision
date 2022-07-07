import math

import torch
from torch import nn
import torch.nn.functional as F

from src.yolov6.models.efficientrep import EfficientRep
from src.yolov6.models.reppan import RepPANNeck
from src.yolov6.utils.torch_utils import fuse_model, fuse_conv_and_bn, initialize_weights


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


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
    def __init__(self):
        super().__init__()

        self.upsample = nn.Sequential(
            UpsampleInterpolate2d(),
            nn.Conv2d(576, 128, kernel_size=3, padding=1, groups=64, bias=False),
            nn.GroupNorm(16, 128),
            UpsampleInterpolate2d(),
            nn.Conv2d(128, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.GroupNorm(8, 32),
            UpsampleInterpolate2d(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.GroupNorm(4, 16),
        )

        self.fuse = nn.Sequential(
            ResidualBlock(
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
                nn.GroupNorm(4, 16),
                nn.Conv2d(16, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16, kernel_size=1),
            ),
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
        self.backbone = config.backbone(pretrained=config.pretrained)
        del self.backbone.avgpool
        del self.backbone.classifier

        self.head = SegmentationHead()
        self.out = nn.Identity()

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.head(x)
        x = self.out(x)
        return x

    def deploy(self):
        # Add the final sigmoid directly into the model
        self.out = nn.Sigmoid()
