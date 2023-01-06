import numpy as np
import torch
from scipy import signal
from torch import nn


class ResidualBlock(nn.Sequential):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__(*args)

    def forward(self, x):
        return x + super().forward(x)


# https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch
# Define 2D Gaussian kernel
def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return torch.tensor(gkern2d)


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.create_conv(3, 1)
        #self.conv2 = self.create_conv(5, 1)
        #self.conv3 = self.create_conv(7, 1)

    def create_conv(self, kernel_size, dilation):
        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, dilation=dilation,
                         padding=(kernel_size * dilation - dilation) // 2, bias=False)
        #conv.weight.data[:] = gkern(kernel_size, 2).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        return conv

    def forward(self, x):
        #x = self.conv1(x) / (3**2) * 0.7 + self.conv2(x) / (5**2) * 0.2 + self.conv3(x) / (7**2) * 0.1
        #x = torch.log(x + 1.)
        x = self.conv1(x)# + self.conv2(x) + self.conv3(x)
        return x


class DownsampleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=in_ch, stride=2, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity(),
            nn.BatchNorm2d(out_ch),
        )


class ResidualConv(nn.Sequential):
    def __init__(self, in_ch):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            #nn.GroupNorm(8, in_ch),
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
        )

    def forward(self, x):
        return x + super().forward(x)


class SpatialPyramidPool(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        mid_ch = max(in_ch // 2, 4)

        self.convin = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.convmid = nn.Conv2d(mid_ch * 4, mid_ch, kernel_size=1)
        self.act = nn.ReLU()
        self.convout = nn.Conv2d(mid_ch, in_ch, kernel_size=1)

    def forward(self, x):
        y1 = self.convin(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        y4 = self.pool(y3)

        y = torch.cat([y1, y2, y3, y4], 1)
        y = self.convmid(y)
        y = self.act(y)
        y = self.convout(y)
        return x + y


class TMNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
            nn.BatchNorm2d(4),
        )

        self.backbone = nn.Sequential(
            # 80x60 input

            DownsampleConv(4, 8),               # 40x30
            #SpatialPyramidPool(8),

            DownsampleConv(8, 16),              # 20x15
            #SpatialPyramidPool(16),
            #ResidualConv(16),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, kernel_size=1),
            #nn.Softmax(dim=1)
        )

        self.head = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        #x = x[:, [0]]
        x = self.head(x)
        return x

    def deploy(self):
        # Add the final detection head directly into the model
        self.head = DetectionHead()

        # Perform activation inplace
        for m in self.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
