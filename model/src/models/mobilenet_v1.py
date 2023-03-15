import torch
import torch.nn.functional as F
from torch import nn

from src.models.detection import DetectionHead


class UpsampleInterpolate2d(nn.Module):
    def __init__(self):
        super(UpsampleInterpolate2d, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class SegmentationNeck(nn.ModuleList):
    def __init__(self, in_ch, upsampling_factor, capture_channels):
        super().__init__()
        mid_ch = max(in_ch // 4, 4)
        self.append(nn.Conv2d(in_ch, mid_ch if upsampling_factor > 0 else 1, kernel_size=1))

        for i in range(upsampling_factor):
            next_ch = max(mid_ch // 4, 4)
            cap_ch = capture_channels[len(capture_channels) - i - 1] if capture_channels else 0

            self.append(UpsampleInterpolate2d())
            self.append(nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False))
            self.append(nn.BatchNorm2d(mid_ch + cap_ch))
            self.append(nn.Conv2d(mid_ch + cap_ch, mid_ch + cap_ch, kernel_size=1))
            self.append(nn.ReLU())
            self.append(nn.Conv2d(mid_ch + cap_ch, next_ch if i < upsampling_factor - 1 else 1, kernel_size=1))

            mid_ch = next_ch

    def forward(self, x, captures):
        concat_ix = len(captures) - 1
        concat_next = False

        for module in self:
            x = module(x)

            if concat_next:
                x = torch.cat([x, captures[concat_ix]], dim=1)
                concat_ix -= 1
                concat_next = False

            if isinstance(module, UpsampleInterpolate2d) and captures:
                concat_next = True

        return x


class CaptureTensor(nn.Module):
    def __init__(self, captures):
        super().__init__()
        self.captures = captures

    def forward(self, x):
        self.captures.append(x)
        return x


class MobileNetSegmentV1(nn.Module):
    """
    Based on MobileNet v1 retrofitted with an UNet-like image segmentation head
    https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/person_detection/training_a_model.md
    """
    def __init__(self, config):
        super().__init__()
        if hasattr(config.backbone, 'captures'):
            self.captures, capture_channels = config.backbone.captures
        else:
            self.captures = []
            capture_channels = []

        self.backbone = config.backbone
        self.neck = SegmentationNeck(config.backbone_out_ch, config.get('upsampling_factor', 1), capture_channels)
        self.head = nn.Identity()

    def forward(self, x):
        try:
            x = self.backbone.features(x)
            x = self.neck(x, self.captures)
            x = self.head(x)
            return x
        finally:
            self.captures.clear()

    def detect(self, x):
        return DetectionHead()(x.to('cpu'))

    def deploy(self, finetuning=False):
        if not finetuning:
            # Add the final detection head directly into the model
            self.head = DetectionHead()

            # Perform activation inplace
            for m in self.modules():
                if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True
