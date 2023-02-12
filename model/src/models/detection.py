import torch
import numpy as np

from torch import nn
from scipy import signal


# https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch
# Define 2D Gaussian kernel
def gkern(kernlen=3, std=2.):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return torch.tensor(gkern2d)


class DilatedGaussianFilter(nn.Module):
    def __init__(self, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, dilation=dilation,
                              padding=(kernel_size * dilation - dilation) // 2)
        self.conv.weight.data[:] = gkern(kernel_size, 2.).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

    def forward(self, x):
        return self.conv(x)


class DetectionHead(nn.Module):
    """
    Uses static and non-learnable gaussian kernels to find the center-of-mass for objects
    """
    def __init__(self):
        super().__init__()

        self.act = nn.Sigmoid()

        self.conv1 = DilatedGaussianFilter(3, 1)
        self.conv2 = DilatedGaussianFilter(3, 2)
        self.conv3 = DilatedGaussianFilter(3, 3)

    def forward(self, x):
        # Max probability at any location
        x = self.act(x)

        # Detect the center-of-mass of objects using gaussian kernels
        y = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = x * 0.8 + y * (0.2 / float(sum(self.conv1.conv.weight.data.reshape(-1)) * 3))

        # Avoid small overflows outside the range of [0, 1]
        return x.clamp(0., 1.)
