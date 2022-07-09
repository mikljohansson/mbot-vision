import numpy as np
import torch
from scipy import signal
from torch import nn
import torch.nn.functional as F

from src.models.gdunet_attention import AttentionBlock


class ResidualBlock(nn.Sequential):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__(*args)

    def forward(self, x):
        return x + super().forward(x)


class DenseBlock(nn.Sequential):
    def __init__(self, *args):
        super(DenseBlock, self).__init__(*args)

    def forward(self, x):
        y = super().forward(x)
        return x#torch.cat([x, y], dim=1)


class UpsampleInterpolate2d(nn.Module):
    def __init__(self):
        super(UpsampleInterpolate2d, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

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


class Downsample(nn.UpsamplingBilinear2d):
    def __init__(self):
        super().__init__(scale_factor=0.5)


class DownsampleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=in_ch, stride=2, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity(),
            nn.BatchNorm2d(out_ch),
        )


class UpsampleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=out_ch),
            nn.BatchNorm2d(out_ch),
        )


class LeakySigmoid(nn.Module):
    """
    Avoids vanishing gradients problem with regular sigmoid function
    https://arxiv.org/abs/1906.03504
    """
    def __init__(self):
        super(LeakySigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x) + x * 0.001


class BiasedSqueezeAndExcitation(torch.nn.Module):
    def __init__(self, in_ch, mid_ch) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv_map = nn.Conv2d(in_ch * 2, mid_ch, 1)
        self.conv_mul = nn.Conv2d(mid_ch, in_ch, 1)
        self.conv_bias = nn.Conv2d(mid_ch, in_ch, 1)
        self.mid_act = nn.ReLU6()
        self.out_act = LeakySigmoid()

    def forward(self, x):
        y1 = self.avgpool(x)
        y2 = self.maxpool(x)

        y = torch.cat([y1, y2], dim=1)
        y = self.conv_map(y)
        y = self.mid_act(y)

        ymul = self.out_act(self.conv_mul(y))
        ybias = self.conv_bias(y)

        return x * ymul + ybias


class ResidualConv(nn.Sequential):
    def __init__(self, in_ch):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.GroupNorm(8, in_ch),
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1),
            BiasedSqueezeAndExcitation(in_ch * 2, max(in_ch // 4, 4)),
            nn.ReLU6(),
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
        )

    def forward(self, x):
        return x + super().forward(x)


class SpatialPyramidPool(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        mid_ch = max(in_ch // 2, 4)

        self.convin = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.convmid = nn.Conv2d(mid_ch * 4, mid_ch, kernel_size=1)
        self.se = BiasedSqueezeAndExcitation(mid_ch, max(mid_ch // 4, 4))
        self.act = nn.ReLU6()
        self.convout = nn.Conv2d(mid_ch, in_ch, kernel_size=1)

    def forward(self, x):
        y1 = self.convin(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        y4 = self.pool(y3)

        y = torch.cat([y1, y2, y3, y4], 1)
        y = self.convmid(y)
        y = self.se(y)
        y = self.act(y)
        y = self.convout(y)
        return x + y


class MVNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
            nn.BatchNorm2d(4),
        )

        self.backbone = nn.Sequential(
            # 80x60 input

            DownsampleConv(4, 8),               # 40x30
            SpatialPyramidPool(8),

            DownsampleConv(8, 16),              # 20x15
            SpatialPyramidPool(16),
            ResidualConv(16),

            nn.BatchNorm2d(16),
            #AttentionBlock(16, 2, use_new_attention_order=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self.head = nn.Identity()

        self.mean2x = torch.nn.Parameter(2. * torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1), requires_grad=False)

    def forward(self, x):
        # Normalize image values and convert to [-1, 1] range inside the network, to simplify deployment
        #image = (image - self.mean) / self.std
        #image = (image - 0.5) / 0.5
        # https://www.wolframalpha.com/input?i=simplify+%28%28x+-+m%29+%2F+s+-+0.5%29+%2F+0.5
        x = (2. * x - self.mean2x) / self.std - 1.

        x = self.stem(x)
        x = self.backbone(x)
        x = self.head(x)
        return x

    def deploy(self):
        # Add the final detection head directly into the model
        self.head = DetectionHead()

        # Perform activation inplace
        for m in self.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
