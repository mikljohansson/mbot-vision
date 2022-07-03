import math
import os.path

import torch
from torch import nn

from src.yolov6.models.efficientrep import EfficientRep
from src.yolov6.models.reppan import RepPANNeck
from src.yolov6.utils.config import Config


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor

class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.in1 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, 32),
        )

        self.in2 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
        )

        self.in3 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1)
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

class MBotVisionModel(nn.Module):
    """
    Based on YOLOv6 and stripped down in size
    """
    def __init__(self, config, channels=3):
        super().__init__()

        depth_mul = config.model.depth_multiple
        width_mul = config.model.width_multiple
        num_repeat_backbone = config.model.backbone.num_repeats
        channels_list_backbone = config.model.backbone.out_channels
        num_repeat_neck = config.model.neck.num_repeats
        channels_list_neck = config.model.neck.out_channels
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def deploy(self):
        pass

def create_model_cfg():
    cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), 'yolov6/configs/yolov6p.py'))
    model = MBotVisionModel(cfg)
    return model, cfg

def create_model():
    model, _ = create_model_cfg()
    return model

