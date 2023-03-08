import operator
from functools import reduce

import torch
import torch.nn.functional as F

from torch import nn


def prod(x):
    """
    Multiply a sequence of numbers together
    https://stackoverflow.com/questions/595374/whats-the-function-like-sum-but-for-multiplication-product
    """
    return reduce(operator.mul, x, 1)


class MLP(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.in_conv = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.act = nn.ReLU()
        self.out_conv = nn.Conv2d(mid_ch, out_ch, kernel_size=1)

    def zero_init(self):
        torch.nn.init.constant_(self.out_conv.weight, self.eps)
        torch.nn.init.constant_(self.out_conv.bias, self.eps)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.act(x)
        x = self.out_conv(x)
        return x


class WorkingMemoryContext(object):
    def __init__(self, cls, mem, mid_ch, x, prev_timestep_state):
        super().__init__()
        self.cls = cls
        self.mem = mem
        self.mid_ch = mid_ch
        self.x = x
        self.prev_timestep_state = prev_timestep_state \
            if prev_timestep_state is not None \
            else torch.zeros((self.x.shape[0], self.mid_ch, 1, 1))

    def __enter__(self):
        self.cls.stack.append((self.mem, self.prev_timestep_state))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cls.stack.pop()

    def get(self):
        return self.cls.stack[-1][1]


class WorkingMemory(nn.Module):
    """
    Encapsulates a working memory that is accessible by child modules. The memory could persist over
    multiple time-steps if the state from the previous time-step is forwarded into the current step.
    """
    stack = []
    in_ch = 16
    out_ch = 16

    def __init__(self, mid_ch=64, enabled=True):
        super().__init__()
        self.is_enabled = enabled
        in_ch = WorkingMemory.in_ch
        self.mid_ch = mid_ch
        out_ch = WorkingMemory.out_ch
        self.query_proj = MLP(in_ch + mid_ch, in_ch * 4, out_ch)
        self.update_proj = MLP(in_ch + mid_ch, mid_ch * 2, mid_ch)

    def enter(self, x, prev_timestep_state):
        return WorkingMemoryContext(WorkingMemory, self, self.mid_ch, x, prev_timestep_state)

    @staticmethod
    def enabled():
        return len(WorkingMemory.stack) > 0 and WorkingMemory.stack[-1][0].is_enabled

    @staticmethod
    def query(x):
        self, state = WorkingMemory.stack[-1]
        if self.is_enabled:
            x = torch.cat([state, x], dim=1)
            x = self.query_proj(x)
            return x

        return torch.zeros((x.shape[0], WorkingMemory.out_ch, 1, 1))

    @staticmethod
    def update(x):
        self, state = WorkingMemory.stack.pop()
        if self.is_enabled:
            x = torch.cat([state, x], dim=1)
            x = self.update_proj(x)
            state = state + x
            WorkingMemory.stack.append((self, state))


class WorkingMemoryQuery(nn.Module):
    """
    Queries the working memory
    """
    def __init__(self, in_ch, out_ch, reshape_like_input=False):
        super().__init__()
        self.out_ch = out_ch
        self.reshape_like_input = reshape_like_input
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.in_proj = MLP(in_ch * 2, WorkingMemory.in_ch * 2, WorkingMemory.in_ch)
        self.out_proj = MLP(WorkingMemory.out_ch, out_ch * 2, out_ch)

    def zero_init(self):
        self.out_proj.zero_init()

    def forward(self, x):
        b, c, h, w = x.shape

        if not WorkingMemory.enabled():
            return torch.zeros(
                (b, self.out_ch, h if self.reshape_like_input else 1, w if self.reshape_like_input else 1),
                dtype=x.dtype, layout=x.layout, device=x.device, requires_grad=False)

        # Downsample to B, C*2, 1, 1
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        x = torch.cat([x1, x2], dim=1)

        # Access memory
        x = self.in_proj(x)
        x = WorkingMemory.query(x)
        x = self.out_proj(x)

        # Reshape to original H, W
        return x.repeat((1, 1, h, w)) if self.reshape_like_input else x


class WorkingMemoryUpdate(nn.Module):
    """
    Updates the working memory
    """
    def __init__(self, in_ch):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.in_proj = MLP(in_ch * 2, WorkingMemory.in_ch * 2, WorkingMemory.in_ch)

    def forward(self, x):
        if not WorkingMemory.enabled():
            return x

        # Downsample to B, C*2, 1, 1
        y1 = self.avgpool(x)
        y2 = self.maxpool(x)
        y = torch.cat([y1, y2], dim=1)

        # Update memory
        y = self.in_proj(y)
        WorkingMemory.update(y)

        return x


class WorkingMemoryConv2d(nn.Conv2d):
    """
    Applies a memory bias to the weights and bias terms of a convolution
    """
    def __init__(self, in_ch, out_ch, kernel_size, **kwargs):
        super().__init__(in_ch, out_ch, kernel_size, **kwargs)
        self.deployed = False

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, **kwargs)

        self.weight_ch = prod(self.weight.shape)
        self.weight_term = WorkingMemoryQuery(in_ch * 2, self.weight_ch)
        self.weight_term.zero_init()

        if self.bias is not None:
            self.bias_ch = prod(self.bias.shape)
            self.bias_term = WorkingMemoryQuery(in_ch * 2, self.bias_ch)
            self.bias_term.zero_init()

    def forward(self, x):
        if not WorkingMemory.enabled():
            return super().forward(x)

        b, c, h, w = x.shape

        # Downsample input to B, C, 1, 1
        y1 = self.avgpool(x)
        y2 = self.maxpool(x)
        y = torch.cat([y1, y2], dim=1)

        # Fetch modifications to conv weights from memory
        weight_mod = self.weight_term(y).reshape((b, *self.weight.shape))
        bias_mod = None

        if self.bias is not None:
            # Fetch modifications to conv biases from memory
            bias_mod = self.bias_term(y).reshape((b, *self.bias.shape))

        # Execute 2d convolution for each sample
        results = []
        for bi in range(b):
            weight = self.weight + weight_mod[bi]
            bias = self.bias + bias_mod[bi] if bias_mod is not None else None
            results.append(F.conv2d(x[bi], weight, bias, self.stride, self.padding, self.dilation, self.groups))

        return torch.stack(results)
