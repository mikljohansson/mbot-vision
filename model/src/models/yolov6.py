import math

import torch
from torch import nn
import torch.nn.functional as F

from src.models.detection import DetectionHead
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


class DownsampleInterpolate2d(nn.Module):
    def __init__(self):
        super(DownsampleInterpolate2d, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='nearest')


class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.in1 = nn.Sequential(
            DownsampleInterpolate2d(),
            nn.GroupNorm(8, 16),
        )

        self.in2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
        )

        self.in3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            UpsampleInterpolate2d(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=1),
            ResidualBlock(
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
                nn.GroupNorm(8, 16),
                nn.Conv2d(16, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=1),
            ),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        x1, x2, x3 = x

        # Downsample to same resolution and channels
        x1 = self.in1(x1)
        x2 = self.in2(x2)
        x3 = self.in3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)

        return x

class YOLOv6Model(nn.Module):
    """
    Based on YOLOv6 with an image segmentation head and stripped down in size
    """
    def __init__(self, config, channels=3):
        super().__init__()

        depth_mul = config.depth_multiple
        width_mul = config.width_multiple
        num_repeat_backbone = config.backbone.num_repeats
        channels_list_backbone = config.backbone.out_channels
        num_repeat_neck = config.neck.num_repeats
        channels_list_neck = config.neck.out_channels
        num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
        channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

        self.backbone = EfficientRep(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat
        )

        self.neck = RepPANNeck(
            channels_list=channels_list,
            num_repeats=num_repeat
        )

        self.head = SegmentationHead()
        self.out = nn.Identity()

        initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        x = self.out(x)
        return x

    def detect(self, x):
        return DetectionHead()(x.to('cpu'))

    def deploy(self, finetuning=False):
        # Fuse batchnorm layers
        fuse_model(self)

        # Fuse 3x3, 1x1 and identity layers
        for layer in self.modules():
            if hasattr(layer, 'switch_to_deploy'):
                #print(f'Switching {type(layer)} to deployment configuration')
                layer.switch_to_deploy()

        if not finetuning:
            # Add the final detection head directly into the model
            self.out = DetectionHead()

            # Perform activation inplace
            for m in self.modules():
                if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True
