import torch.nn.functional as F
from torch import nn

from src.models.detection import DetectionHead


class UpsampleInterpolate2d(nn.Module):
    def __init__(self):
        super(UpsampleInterpolate2d, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class SegmentationNeck(nn.Sequential):
    def __init__(self, in_ch):
        mid_first_ch = 32
        mid_second_ch = mid_first_ch // 4

        super().__init__(
            nn.Conv2d(in_ch, mid_first_ch, kernel_size=1),

            UpsampleInterpolate2d(),
            nn.Conv2d(mid_first_ch, mid_first_ch, kernel_size=3, padding=1, groups=mid_first_ch, bias=False),
            nn.Conv2d(mid_first_ch, mid_second_ch, kernel_size=1),

            UpsampleInterpolate2d(),
            nn.Conv2d(mid_second_ch, mid_second_ch, kernel_size=3, padding=1, groups=mid_second_ch, bias=False),
            nn.Conv2d(mid_second_ch, 1, kernel_size=1),
        )


class MobileNetSegmentV1(nn.Module):
    """
    Based on MobileNet v1 with an image segmentation head and same config as the tflite-micro "person detection" example
    https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/person_detection/training_a_model.md
    """
    def __init__(self, config):
        super().__init__()
        self.backbone = config.backbone
        self.neck = SegmentationNeck(config.backbone_out_ch)
        self.head = nn.Identity()

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.neck(x)
        x = self.head(x)
        return x

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
