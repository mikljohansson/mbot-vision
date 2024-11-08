import math

import torch
import torch.nn.functional as F

from torch import nn
from einops import einsum

from src.models.detection import DetectionHead
from src.models.memory import WorkingMemory, WorkingMemoryConv2d, WorkingMemoryUpdate, WorkingMemoryQuery


class ResidualBlock(nn.Sequential):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__(*args)

    def forward(self, x):
        return x + super().forward(x)


class GatedBlock(nn.Sequential):
    def __init__(self, *args):
        super(GatedBlock, self).__init__(*args)

    def forward(self, x):
        return x * super().forward(x)


class DenseBlock(nn.Sequential):
    def __init__(self, *args):
        super(DenseBlock, self).__init__(*args)

    def forward(self, x):
        y = super().forward(x)
        return torch.cat([x, y], dim=1)


class ConcatParallel(nn.ModuleList):
    """
    Concats the results from multiple parallel execution branches
    """
    def __init__(self, *modules: nn.Module):
        super().__init__(modules)

    def forward(self, x):
        return torch.cat([module(x) for module in self], dim=1)


class AddParallel(nn.ModuleList):
    """
    Adds the results from multiple parallel execution branches
    """
    def __init__(self, *modules: nn.Module):
        super().__init__(modules)

    def forward(self, x):
        y = self[0](x)
        for module in self[1:]:
            y = y + module(x)
        return y


class MultiplyParallel(nn.ModuleList):
    """
    Multiplies the results from multiple parallel execution branches
    """
    def __init__(self, *modules: nn.Module):
        super().__init__(modules)

    def forward(self, x):
        y = self[0](x)
        for module in self[1:]:
            y = y * module(x)
        return y


class Sum(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.sum(x, *self.args, **self.kwargs)


class UpsampleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, scale_factor=None, size=None):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.UpsamplingNearest2d(size=size, scale_factor=scale_factor),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1, groups=math.gcd(in_ch, out_ch), bias=False),
            nn.BatchNorm2d(out_ch),
        )


class DownsampleNearest2d(nn.UpsamplingNearest2d):
    def __init__(self):
        super().__init__(scale_factor=0.5)


class DownsampleBilinear2d(nn.UpsamplingBilinear2d):
    def __init__(self):
        super().__init__(scale_factor=0.5)


class DownsampleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, memory=False):
        conv = WorkingMemoryConv2d if memory else nn.Conv2d
        super().__init__(
            conv(in_ch, out_ch, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), groups=math.gcd(in_ch, out_ch), stride=stride, bias=False),
            conv(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, attention=False, memory=False, **kwargs):
        conv = WorkingMemoryConv2d if memory else nn.Conv2d
        super().__init__(
            conv(in_ch, out_ch, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), **kwargs),
            nn.BatchNorm2d(out_ch),
            BiasedSqueezeAndExcitation(out_ch) if attention else nn.Identity(),
            nn.Mish(),
        )


class DWConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, attention=False, memory=False,**kwargs):
        conv = WorkingMemoryConv2d if memory else nn.Conv2d
        super().__init__(
            conv(in_ch, out_ch, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), groups=math.gcd(in_ch, out_ch), **kwargs),
            conv(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            BiasedSqueezeAndExcitation(out_ch) if attention else nn.Identity(),
            nn.Mish(),
        )


class MobileNet3Block(ResidualBlock):
    def __init__(self, in_ch, out_ch, kernel_size, attention=False, expansion_ratio=4.):
        mid_ch = max(int(in_ch * expansion_ratio), 4)
        super().__init__(
            ConvNormAct(in_ch, mid_ch, kernel_size=1),
            ConvNormAct(mid_ch, mid_ch, kernel_size=kernel_size, attention=attention, groups=mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )


class ChannelShuffle(nn.Module):
    """
    PyTorch nn.ChannelShuffle doesn't have CUDA support
    """
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        # https://github.com/MG2033/ShuffleNet/blob/master/layers.py#L238
        n, c, h, w = x.shape
        x = x.reshape((n, self.groups, c // self.groups, h, w))
        x = x.transpose(2, 1)
        x = x.reshape((n, c, h, w))
        return x


class MeanMax(nn.Module):
    def __init__(self, groups=1):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        group_size = x.size()[1] // self.groups
        result = []

        assert group_size > 0
        assert x.size()[1] % self.groups == 0

        for group in range(self.groups):
            group_slice = x[:, (group * group_size):((group + 1) * group_size)]
            result.append(torch.mean(group_slice, dim=1, keepdim=True))
            result.append(torch.max(group_slice, dim=1, keepdim=True)[0])

        return torch.cat(result, dim=1)


class ExpandChannels(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.out_ch = out_ch

    def forward(self, x):
        return x.expand(-1, self.out_ch, -1, -1)


class BiasedSqueezeAndExcitation(nn.Module):
    """
    Channel attention Squeeze-and-Excitation block modified with inverted bottleneck and bias term
    """
    def __init__(self, in_ch, out_ch=None, expansion_ratio=4.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        mid_ch = max(int(in_ch * expansion_ratio), 4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.memory = WorkingMemoryQuery(in_ch, in_ch) if WorkingMemory.enabled() else None
        self.conv_map = nn.Conv2d(in_ch * 3 if self.memory is not None else in_ch * 2, mid_ch, kernel_size=1)
        self.mid_act = nn.Mish()
        self.conv_bias = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
        self.conv_mul = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
        self.act_mul = nn.Sigmoid()

    def factors(self, x):
        # Downsample to B, C*3, 1, 1
        y1 = self.avgpool(x)
        y2 = self.maxpool(x)

        if self.memory is not None:
            y3 = self.memory(x)
            y = torch.cat([y1, y2, y3], dim=1)
        else:
            y = torch.cat([y1, y2], dim=1)

        y = self.conv_map(y)
        y = self.mid_act(y)

        ybias = self.conv_bias(y)
        ymul = self.act_mul(self.conv_mul(y))
        return ymul, ybias

    def forward(self, x, y=None):
        ymul, ybias = self.factors(y if y is not None else x)
        return x * ymul + ybias


class ChannelAndSpatialAttention(nn.Module):
    """
    Channel and spatial attention module. Mix of CBAM and DFC attention and modified with biases
    and multiple spatial attention heads. Inspired by Axial Attention, but not using matrix multiplication

    https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnetv2_pytorch/model/ghostnetv2_torch.py#L104
    https://openreview.net/pdf/6db544c65bbd0fa7d7349508454a433c112470e2.pdf
    https://medium.com/mlearning-ai/axial-self-attention-in-cnn-efficient-and-elegant-85d8ce2ca8eb
    https://github.com/lucidrains/axial-attention
    """
    def __init__(self, in_ch, kernel_size=7, dilation=1, heads=4):
        super().__init__()
        heads = min(in_ch // 2, heads)
        heads = math.gcd(in_ch, heads)

        self.sqe1 = BiasedSqueezeAndExcitation(in_ch)
        self.meanmax = MeanMax(heads)

        self.conv = nn.Sequential(
            nn.Conv2d(heads * 2, heads, kernel_size=(1, kernel_size), stride=1,
                      padding=(0, (kernel_size * dilation - dilation) // 2),
                      dilation=dilation, groups=heads, bias=False),
            nn.Conv2d(heads, heads, kernel_size=(kernel_size, 1), stride=1,
                      padding=((kernel_size * dilation - dilation) // 2, 0),
                      dilation=dilation, groups=heads, bias=False),
            nn.Conv2d(heads, heads, kernel_size=1),
            BiasedSqueezeAndExcitation(heads),
            nn.Mish(),
        )

        self.mul_term = nn.Sequential(
            nn.Conv2d(heads, 1, kernel_size=1),
            # Expand spatial attention map back to in_ch
            ExpandChannels(in_ch),
            # Moderate which channels get the spatial attention applied
            BiasedSqueezeAndExcitation(in_ch),
            nn.Sigmoid()
        )

        self.bias_term = nn.Sequential(
            nn.Conv2d(heads, 1, kernel_size=1),
            # Expand spatial attention map back to in_ch
            ExpandChannels(in_ch),
            # Moderate which channels get the spatial attention applied
            BiasedSqueezeAndExcitation(in_ch)
        )

    def forward(self, x):
        # Moderate which input channels to pay spatial attention to
        y = self.sqe1(x)

        # Reduce to mean and max channels, keeping the spatial dimension
        y = self.meanmax(y)

        # Downsample to improve performance
        #y = F.avg_pool2d(y, kernel_size=2, stride=2)

        # Calculate multi-head spatial attention maps
        y = self.conv(y)

        # Calculate mul and bias terms
        ymul = self.mul_term(y)
        ybias = self.bias_term(y)

        # Upsample again
        #xybias = F.interpolate(xybias, size=x.shape[-2:], mode='nearest')
        #xymul = F.interpolate(xymul, size=x.shape[-2:], mode='nearest')

        # Apply the spatial and channel attention
        return x * ymul + ybias


class GlobalContext(nn.Module):
    """
    Channel attention (Global Context Block) modified to support multi-head attention
    https://arxiv.org/pdf/1904.11492.pdf
    https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
    https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py#L940
    """
    def __init__(self, in_ch, attention_heads=4):
        super().__init__()
        self.attention_heads = attention_heads
        self.key_conv = nn.Conv2d(in_ch, self.attention_heads, kernel_size=1)

        # ConvNeXt uses an inverted bottleneck design
        mid_ch = in_ch * 4

        self.mul_term = nn.Sequential(
            nn.Conv2d(in_ch * self.attention_heads, mid_ch, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(mid_ch, in_ch, 1),
            nn.Sigmoid()
        )

        self.bias_term = nn.Sequential(
            nn.Conv2d(in_ch * self.attention_heads, mid_ch, kernel_size=1),
            nn.Mish(),
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

        mul = self.mul_term(attention)
        bias = self.bias_term(attention)
        # [B, C, 1, 1]

        return x * mul + bias


class GhostAttention(nn.Module):
    """
    Gated spatial attention module (DFC attention) using only convolutions
    https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, **kwargs):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, 1, bias=False, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, kernel_size), stride=1,
                      padding=(0, (kernel_size * dilation - dilation) // 2),
                      dilation=dilation, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size, 1), stride=1,
                      padding=((kernel_size * dilation - dilation) // 2, 0),
                      dilation=dilation, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

    def forward(self, x):
        return x * self.attention(x)


class ParallelGhostAttention(nn.Module):
    """
    Gated spatial attention module (DFC attention) with channel attention
    https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()

        mid_ch = max(in_ch // 2, out_ch // 2)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            DenseBlock(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                ConcatParallel(
                    nn.Conv2d(mid_ch, mid_ch, kernel_size=(1, kernel_size), stride=1,
                              padding=(0, (kernel_size * dilation - dilation) // 2),
                              dilation=dilation, groups=out_ch, bias=False),
                    nn.Conv2d(mid_ch, mid_ch, kernel_size=(kernel_size, 1), stride=1,
                              padding=((kernel_size * dilation - dilation) // 2, 0),
                              dilation=dilation, groups=out_ch, bias=False),
                ),
            ),
            BiasedSqueezeAndExcitation(in_ch + mid_ch * 2),
            nn.Conv2d(in_ch + mid_ch * 2, out_ch, 1),
            nn.Sigmoid(),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

    def forward(self, x):
        return x * self.attention(x)


class LinearAttention(nn.Module):
    """
    https://github.com/tatp22/linformer-pytorch
    https://github.com/lucidrains/linear-attention-transformer
    https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py#L871
    """
    def __init__(
            self,
            dim,
            dim_head = 32,
            heads = 8,
            dropout = 0.05,
            context_dim = None,
            **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (ck, cv))
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)


class LowRankSelfAttention(nn.Module):
    """
    Approximates self-attention using a low-rank projection matrix that reduces the number of
    items in the sequence to a low and constant number. Extended with multiple heads
    https://www.frontiersin.org/articles/10.3389/fpls.2022.978564/full
    """
    def __init__(self, in_ch, width, height, k=256, heads=4):
        super().__init__()

        self.in_ch = in_ch
        self.k = k
        self.heads = math.gcd(in_ch, heads)
        self.qk_ch = max(in_ch // 8, 4) * self.heads

        self.in_conv = nn.Conv2d(in_ch, self.qk_ch * 2 + in_ch, kernel_size=1)
        self.key_proj = nn.Linear(width * height, k)
        self.value_proj = nn.Linear(width * height, k)
        self.att_dropout = nn.Dropout(p=0.2)

        self.scale = 1. / math.sqrt(self.qk_ch // self.heads)

        self.out_dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        b, _, h, w = x.size()

        # [B, C, H, W]
        query, key, value = self.in_conv(x).split([self.qk_ch, self.qk_ch, self.in_ch], dim=1)
        query = query.flatten(start_dim=2).transpose(1, 2)
        # [B, H*W, qk_ch]

        # Project to low rank spatial dimentionality
        key = self.key_proj(key.flatten(start_dim=2)).transpose(1, 2)
        # [B, k, qk_ch]

        value = self.value_proj(value.flatten(start_dim=2)).transpose(1, 2)
        # [B, k, in_ch]

        # Add heads dimension
        query = query.view(b, h*w, self.heads, self.qk_ch // self.heads).transpose(1, 2)
        # [B, nh, H*W, qk_ch/nh]

        key = key.view(b, self.k, self.heads, self.qk_ch // self.heads).transpose(1, 2)
        # [B, nh, k, qk_ch/nh]

        value = value.view(b, self.k, self.heads, self.in_ch // self.heads).transpose(1, 2)
        # [B, nh, k, in_ch/nh]

        # Calculate attention scores
        att = query @ key.transpose(-2, -1) * self.scale
        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)
        # [B, nh, H*W, k]

        # Retrieve attention values
        y = att @ value
        # [B, nh, H*W, in_ch/nh]

        y = y.transpose(1, 2).view(b, h, w, self.in_ch).movedim(-1, 1).contiguous()
        # [B, C, H, W]

        # Output
        y = self.out_dropout(y)
        return x + y


class LinearAttention(nn.Module):
    """
    Approximates self-attention using a low-rank projection matrix
    https://www.frontiersin.org/articles/10.3389/fpls.2022.978564/full
    """
    def __init__(self, in_ch, k=512):
        super().__init__()


    def forward(self, x):
        # [B, C, H, W]
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # [B, C, H*W]
        query = query.flatten(start_dim=2).transpose(1, 2)
        key = key.flatten(start_dim=2).transpose(1, 2)
        value = value.flatten(start_dim=2).transpose(1, 2)

        # Project key and value to low-rank dimensionality
        # [B, K, C]
        key = self.key_proj


class ConvNeXt(nn.Sequential):
    """
    ConvNeXt module with optional channel attention
    """
    def __init__(self, in_ch, kernel_size=3, groups=1, expansion_ratio=4, attention=True):
        mid_ch = int(in_ch * expansion_ratio)

        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, groups=groups),
            BiasedSqueezeAndExcitation(mid_ch, expansion_ratio=0.5) if attention else nn.Identity(),
            nn.Mish(),
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, groups=groups),
        )

    def forward(self, x):
        return x + super().forward(x)


class SpatialPyramidPool(nn.Module):
    """
    Mix of SPP and ConvNeXt with optional channel and spatial attention
    """
    def __init__(self, in_ch, attention=True, channel_add=False):
        super().__init__()
        self.channel_add = channel_add
        self.in_ch = in_ch
        self.mid_ch = max(in_ch // 2, 4)
        out_ch = in_ch * 2 if not channel_add else self.mid_ch

        self.convin = nn.Conv2d(in_ch, self.mid_ch, kernel_size=3, padding=1, groups=math.gcd(in_ch, self.mid_ch))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        self.att1 = BiasedSqueezeAndExcitation(self.mid_ch * 4) if attention else nn.Identity()
        self.norm = nn.BatchNorm2d(self.mid_ch * 4)

        #self.shuffle = ChannelShuffle(4)
        #self.convmid = nn.Conv2d(mid_ch * 4, in_ch, kernel_size=1, groups=4) if not channel_add else nn.Identity()
        self.shuffle = nn.Identity()
        self.convmid = nn.Conv2d(self.mid_ch * 4, out_ch, kernel_size=1) if not channel_add else nn.Identity()

        self.att2 = ChannelAndSpatialAttention(out_ch) if attention else nn.Identity()
        self.act = nn.Mish()
        self.convout = nn.Conv2d(out_ch, in_ch, kernel_size=1)

    def forward(self, x):
        y1 = self.convin(x)
        y2 = self.pool1(y1)
        y3 = self.pool2(y2)
        y4 = self.pool3(y3)

        y = torch.cat([y1, y2, y3, y4], 1)
        y = self.att1(y)
        y = self.norm(y)

        if self.channel_add:
            y = y[:, 0:self.mid_ch] + \
                y[:, self.mid_ch:(self.mid_ch * 2)] + \
                y[:, (self.mid_ch * 2):(self.mid_ch * 3)] + \
                y[:, (self.mid_ch * 3):(self.mid_ch * 4)]
        else:
            y = self.shuffle(y)
            y = self.convmid(y)

        y = self.att2(y)
        y = self.act(y)
        y = self.convout(y)
        return x + y


class SSPF(nn.Module):
    """
    Spatial Pyramid Pool Fast from YOLOv5/v8 with optional channel attention
    """
    def __init__(self, in_ch, out_ch, attention=True):
        super().__init__()
        mid_ch = in_ch // 2

        self.convin = ConvNormAct(in_ch, mid_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.att = BiasedSqueezeAndExcitation(mid_ch * 4) if attention else nn.Identity()
        self.convout = nn.Conv2d(mid_ch * 4, out_ch, kernel_size=1)

    def forward(self, x):
        y1 = self.convin(x)

        y2 = self.pool(y1)
        y3 = self.pool(y2)
        y4 = self.pool(y3)

        y = torch.cat([y1, y2, y3, y4], 1)
        y = self.att(y)
        y = self.convout(y)
        return y


class MVNetModel(nn.Module):
    """
    UNet-like image segmentation model with addition of
     - Spatial pyramid pooling
     - Optional channel-and-spatial attention
     - Optional cross time-step working memory
    """
    def __init__(self, config):
        super().__init__()
        channels = config.channels
        memory = config.memory

        self.memory = WorkingMemory(enabled=memory)

        # Let attention modules know about the working memory being enabled/disabled
        with self.memory.enter(None):
            self.backbone = nn.Sequential(
                # 80x48 input
                WorkingMemoryConv2d(channels[0], channels[1], kernel_size=1) if memory else nn.Identity(),

                # 40x24
                ConvNormAct(channels[1 if memory else 0], channels[2], kernel_size=3, stride=2, bias=False, memory=memory),

                # 20x12
                DWConvNormAct(channels[2], channels[3], kernel_size=3, stride=2, bias=False, attention=config.attention, memory=memory),
                SpatialPyramidPool(channels[3], attention=config.attention),

                DWConvNormAct(channels[3], channels[3], kernel_size=3, attention=config.attention),
                SpatialPyramidPool(channels[3], attention=config.attention, channel_add=True),

                DenseBlock(
                    # 10x6
                    DownsampleConv(channels[3], channels[4]),

                    ConvNeXt(channels[4], expansion_ratio=2, groups=4, attention=config.attention),

                    DenseBlock(
                        # 5x3
                        DownsampleConv(channels[4], channels[5]),

                        ConvNeXt(channels[5], expansion_ratio=2, groups=4, attention=config.attention),
                        nn.Conv2d(channels[5], channels[5], kernel_size=1),
                        ConvNeXt(channels[5], expansion_ratio=2, groups=4, attention=config.attention),

                        WorkingMemoryUpdate(channels[5]) if memory else nn.Identity(),

                        # 10x6
                        UpsampleConv(channels[5], channels[4], scale_factor=2)
                    ),

                    #WorkingMemoryUpdate(channels[4] * 2) if memory else nn.Identity(),

                    # 20x12
                    UpsampleConv(channels[4] * 2, channels[3], scale_factor=2)
                ),

                #WorkingMemoryUpdate(channels[3] * 2) if memory else nn.Identity(),
                SSPF(channels[3] * 2, 1, attention=config.attention)
            )

        self.head = nn.Identity()

    def forward(self, x, prev_state=None, state_mask=None):
        if self.memory.is_enabled:
            init_state = prev_state * state_mask.view((state_mask.shape[0], 1, 1, 1)) if prev_state is not None else None
            with self.memory.enter(init_state) as context:
                x = self.backbone(x)
                x = self.head(x)
                return x, context.get()
        else:
            x = self.backbone(x)
            x = self.head(x)
            return x

    def detect(self, x):
        return DetectionHead()(x.to('cpu'))

    def deploy(self):
        # Deploy sub modules
        for m in self.modules():
            if hasattr(m, 'deploy') and self != m:
                m.deploy()

        # Add the final detection head directly into the model
        self.head = DetectionHead()

        # Perform activation inplace
        for m in self.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
