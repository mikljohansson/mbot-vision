import numpy as np
import torch
from scipy import signal
from torch import nn
import torch.nn.functional as F
from torchvision.ops.misc import ConvNormActivation


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


class SegmentationNeck(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.upsample = nn.Sequential(
            UpsampleInterpolate2d(),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.GroupNorm(16, in_ch),
            nn.Conv2d(in_ch, 16, kernel_size=1),

            UpsampleInterpolate2d(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

# https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch
# Define 2D Gaussian kernel
def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return torch.tensor(gkern2d)

class DilatedGaussianFilter(nn.Module):
    def __init__(self, kernel_size=3, dilation=1):
        super().__init__()

class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.Sigmoid()

        self.conv1 = self.create_conv(3, 1)
        self.conv2 = self.create_conv(5, 1)
        self.conv3 = self.create_conv(7, 1)

    def create_conv(self, kernel_size, dilation):
        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, dilation=dilation,
                         padding=(kernel_size * dilation - dilation) // 2, bias=False)
        conv.weight.data[:] = gkern(kernel_size, 2).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        return conv

    def forward(self, x):
        x = self.act(x)
        x = self.conv1(x) / (3**2) * 0.7 + self.conv2(x) / (5**2) * 0.2 + self.conv3(x) / (7**2) * 0.1
        x = torch.log(x + 1.)
        return x


class MobileNetModel(nn.Module):
    """
    Based on MobileNet v3 with an image segmentation head
    """
    def __init__(self, config):
        super().__init__()

        self.backbone = config.backbone

        # Remove the last channel expansion layer
        del self.backbone.features[-1]

        # Remove the object detection head
        del self.backbone.avgpool
        del self.backbone.classifier

        # Switch all activations to use ReLU
        for module in self.backbone.modules():
            if isinstance(module, ConvNormActivation):
                if type(module[-1]) in [nn.Hardswish, nn.LeakyReLU, nn.SiLU]:
                    del module[-1]
                    module.append(nn.ReLU())

        self.neck = SegmentationNeck(config.backbone_out_ch)
        self.head = nn.Identity()

        self.mean2x = torch.nn.Parameter(2. * torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1), requires_grad=False)

    def forward(self, x):
        # Normalize image values and convert to [-1, 1] range inside the network, to simplify deployment
        #image = (image - self.mean) / self.std
        #image = (image - 0.5) / 0.5
        # https://www.wolframalpha.com/input?i=simplify+%28%28x+-+m%29+%2F+s+-+0.5%29+%2F+0.5
        x = (2. * x - self.mean2x) / self.std - 1.

        x = self.backbone.features(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def deploy(self):
        # Add the final detection head directly into the model
        self.head = DetectionHead()

        # Perform activation inplace
        for m in self.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
