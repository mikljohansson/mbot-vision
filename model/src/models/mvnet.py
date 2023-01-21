import numpy as np
import torch
from scipy import signal
from torch import nn
import torch.nn.functional as F
from einops import einsum

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
        return torch.cat([x, y], dim=1)


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
    def __init__(self, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, dilation=dilation,
                              padding=(kernel_size * dilation - dilation) // 2)
        self.conv.weight.data[:] = gkern(kernel_size, 2).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        self.divisor = kernel_size ** 2

    def forward(self, x):
        return self.conv(x) / self.divisor


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.Sigmoid()

        self.conv1 = DilatedGaussianFilter(3, 1)
        self.conv2 = DilatedGaussianFilter(3, 2)
        self.conv3 = DilatedGaussianFilter(3, 4)

    def forward(self, x):
        x = self.act(x)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return x


class Downsample(nn.UpsamplingBilinear2d):
    def __init__(self):
        super().__init__(scale_factor=0.5)


class DownsampleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, mix_ch=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), groups=in_ch, stride=2),
            nn.Conv2d(out_ch, out_ch, kernel_size=1) if mix_ch else nn.Identity(),
            nn.BatchNorm2d(out_ch),
        )


class UpsampleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, scale_factor=None, size=None):
        super().__init__(
            nn.Upsample(scale_factor=scale_factor, size=size),
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
    def __init__(self, in_ch, mul_ch=2):
        super().__init__()
        mid_ch = int(in_ch * mul_ch)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv_map = nn.Conv2d(in_ch * 2, mid_ch, kernel_size=1)
        self.conv_mul = nn.Conv2d(mid_ch, in_ch, kernel_size=1)
        self.conv_bias = nn.Conv2d(mid_ch, in_ch, kernel_size=1)
        self.mid_act = nn.ReLU6()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avgpool(x)
        y2 = self.maxpool(x)

        y = torch.cat([y1, y2], dim=1)
        y = self.conv_map(y)
        y = self.mid_act(y)

        ymul = self.out_act(self.conv_mul(y))
        ybias = self.conv_bias(y)

        return x * ymul + ybias


class GlobalAttention(nn.Module):
    """
    Modified version of Global Context Block
    https://arxiv.org/pdf/1904.11492.pdf
    https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
    https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py#L909
    """
    def __init__(self, in_ch, attention_heads=4):
        super().__init__()
        self.attention_heads = attention_heads
        self.key_conv = nn.Conv2d(in_ch, self.attention_heads, 1)

        # ConvNeXt uses an inverted bottleneck design
        mid_ch = in_ch * 4

        self.mul_term = nn.Sequential(
            nn.Conv2d(in_ch * self.attention_heads, mid_ch, 1),
            nn.ReLU6(),
            nn.Conv2d(mid_ch, in_ch, 1),
            nn.Sigmoid()
        )

        self.bias_term = nn.Sequential(
            nn.Conv2d(in_ch * self.attention_heads, mid_ch, 1),
            nn.ReLU6(),
            nn.Conv2d(mid_ch, in_ch, 1),
        )

    def forward(self, x):
        # [B, C, H, W]
        b, c, h, w = x.size()

        key = self.key_conv(x)
        # [B, A, H, W]

        key = key.reshape(b, self.attention_heads, h * w)
        key = key.softmax(dim=-1)
        # [B, A, H * W]

        value = x.reshape(b, c, h * w)
        # [B, C, H * W]

        attention = einsum(key, value, 'b a hw, b c hw -> b c a')
        # [B, C, A]

        attention = attention.reshape(b, c * self.attention_heads, 1, 1)
        # [B, C * A, 1, 1]

        return x * self.mul_term(attention) + self.bias_term(attention)


class ConvNeXt(nn.Sequential):
    def __init__(self, in_ch, mul_ch=4):
        mid_ch = int(in_ch * mul_ch)

        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            BiasedSqueezeAndExcitation(in_ch),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            BiasedSqueezeAndExcitation(mid_ch),
            nn.ReLU6(),
            nn.Conv2d(mid_ch, in_ch, kernel_size=1),
        )

    def forward(self, x):
        return x + super().forward(x)


class SpatialPyramidPool(nn.Module):
    def __init__(self, in_ch, channel_add=True):
        super().__init__()
        self.channel_add = channel_add
        self.in_ch = in_ch

        self.convin = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.att1 = BiasedSqueezeAndExcitation(in_ch * 4)
        self.norm = nn.BatchNorm2d(in_ch * 4)
        self.convmid = nn.Conv2d(in_ch * 4, in_ch, kernel_size=1)
        self.att2 = BiasedSqueezeAndExcitation(in_ch)
        self.act = nn.ReLU6()
        self.convout = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def forward(self, x):
        y1 = self.convin(x)
        y2 = self.pool1(y1)
        y3 = self.pool2(y2)
        y4 = self.pool3(y3)

        y = torch.cat([y1, y2, y3, y4], 1)
        y = self.att1(y)
        y = self.norm(y)

        if self.channel_add:
            y = y[:, 0:self.in_ch] + \
                y[:, self.in_ch:(self.in_ch * 2)] + \
                y[:, (self.in_ch * 2):(self.in_ch * 3)] + \
                y[:, (self.in_ch * 3):(self.in_ch * 4)]
        else:
            y = self.convmid(y)

        y = self.att2(y)
        y = self.act(y)
        y = self.convout(y)
        return x + y


class MVNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),
            nn.BatchNorm2d(3),
        )

        self.backbone = nn.Sequential(
            # 80x60 input

            DownsampleConv(3, 3, mix_ch=True),  # 40x30
            SpatialPyramidPool(3),

            DownsampleConv(3, 3, mix_ch=True),  # 20x15
            SpatialPyramidPool(3),

            DenseBlock(
                DownsampleConv(3, 9),          # 10x8
                ConvNeXt(9, mul_ch=2),

                ResidualBlock(
                    DownsampleConv(9, 18),     # 5x4
                    ConvNeXt(18, mul_ch=2),
                    UpsampleConv(18, 9, scale_factor=2)    # 10x8
                ),

                UpsampleConv(9, 3, size=(15, 20))          # 20x15
            ),

            nn.Conv2d(6, 1, kernel_size=1)
        )

        self.head = nn.Identity()
        self.detector = DetectionHead()

        #self.mean2x = torch.nn.Parameter(2. * torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1), requires_grad=False)
        #self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1), requires_grad=False)

    def forward(self, x):
        # Normalize image values and convert to [-1, 1] range inside the network, to simplify deployment
        #image = (image - self.mean) / self.std
        #image = (image - 0.5) / 0.5
        # https://www.wolframalpha.com/input?i=simplify+%28%28x+-+m%29+%2F+s+-+0.5%29+%2F+0.5
        #x = (2. * x - self.mean2x) / self.std - 1.

        x = self.stem(x)
        x = self.backbone(x)
        x = self.head(x)
        return x

    def detect(self, x):
        x = self.detector(x)
        return x / torch.amax(x, dim=(1, 2, 3))

    def deploy(self):
        # Add the final detection head directly into the model
        self.head = DetectionHead()

        # Perform activation inplace
        for m in self.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
