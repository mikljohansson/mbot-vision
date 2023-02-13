import torch

from torch import nn
from einops import einsum

from src.models.detection import DetectionHead


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
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1, groups=out_ch),
            nn.BatchNorm2d(out_ch),
        )


class DownsampleNearest2d(nn.UpsamplingNearest2d):
    def __init__(self):
        super().__init__(scale_factor=0.5)


class DownsampleBilinear2d(nn.UpsamplingBilinear2d):
    def __init__(self):
        super().__init__(scale_factor=0.5)


class DownsampleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), groups=in_ch, stride=stride),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, attention=False, **kwargs):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_ch),
            BiasedSqueezeAndExcitation(out_ch) if attention else nn.Identity(),
            nn.ReLU(),
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
    def forward(self, x):
        return torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]], 1)


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
        self.conv_map = nn.Conv2d(in_ch * 2, mid_ch, kernel_size=1)
        self.mid_act = nn.ReLU()
        self.conv_bias = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
        self.conv_mul = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
        self.act_mul = nn.Sigmoid()

    def factors(self, x):
        y1 = self.avgpool(x)
        y2 = self.maxpool(x)

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
        self.in_ch = in_ch
        self.sqe1 = BiasedSqueezeAndExcitation(in_ch)
        self.meanmax = MeanMax()

        self.conv = nn.Sequential(
            #nn.Conv2d(2, heads, kernel_size, padding=(kernel_size // 2), bias=False),
            nn.Conv2d(2, heads, kernel_size=(1, kernel_size), stride=1,
                      padding=(0, (kernel_size * dilation - dilation) // 2),
                      dilation=dilation, bias=False),
            nn.Conv2d(heads, heads, kernel_size=(kernel_size, 1), stride=1,
                      padding=((kernel_size * dilation - dilation) // 2, 0),
                      dilation=dilation, groups=heads, bias=False),
            nn.Conv2d(heads, heads, kernel_size=1),
            BiasedSqueezeAndExcitation(heads),
            nn.ReLU(),
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


class GlobalAttention(nn.Module):
    """
    Spatial attention Global Context Block modified to support multi-head attention
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
            nn.ReLU(),
            nn.Conv2d(mid_ch, in_ch, 1),
            nn.Sigmoid()
        )

        self.bias_term = nn.Sequential(
            nn.Conv2d(in_ch * self.attention_heads, mid_ch, 1),
            nn.ReLU(),
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


class GhostAttention(nn.Sequential):
    """
    Gated spatial attention module (DFC attention) using only convolutions
    https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, **kwargs):
        super().__init__(
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


class ParallelGhostAttention(nn.Sequential):
    """
    Gated spatial attention module (DFC attention) with channel attention
    https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        mid_ch = max(in_ch // 2, out_ch // 2)
        super().__init__(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(mid_ch),
            DenseBlock(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                ConcatParallel(
                    nn.Conv2d(mid_ch, mid_ch, kernel_size=(1, kernel_size), stride=1,
                              padding=(0, (kernel_size * dilation - dilation) // 2),
                              dilation=dilation, groups=out_ch, bias=False),
                    nn.Conv2d(mid_ch, mid_ch, kernel_size=(kernel_size, 1), stride=1,
                              padding=((kernel_size * dilation - dilation) // 2, 0),
                              dilation=dilation, groups=out_ch, bias=False),
                ),
            ),
            BiasedSqueezeAndExcitation(mid_ch * 3),
            #ChannelShuffle(3),
            #nn.Conv2d(mid_ch * 3, out_ch, 1, groups=mid_ch),
            nn.Conv2d(mid_ch * 3, out_ch, 1),
            nn.Sigmoid(),
            nn.UpsamplingNearest2d(scale_factor=2)
        )


class ConvNeXt(nn.Sequential):
    """
    ConvNeXt module with optional channel attention
    """
    def __init__(self, in_ch, kernel_size=3, groups=1, expansion_ratio=4, attention=False):
        mid_ch = int(in_ch * expansion_ratio)

        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, groups=groups),
            BiasedSqueezeAndExcitation(mid_ch, expansion_ratio=0.5) if attention else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, groups=groups),
        )

    def forward(self, x):
        return x + super().forward(x)


class SpatialPyramidPool(nn.Module):
    """
    Mix of SPP and ConvNeXt with optional channel attention
    """
    def __init__(self, in_ch, attention=False, channel_add=False):
        super().__init__()
        self.channel_add = channel_add
        self.in_ch = in_ch
        mid_ch = max(in_ch // 2, 4)

        self.convin = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, groups=min(in_ch, mid_ch))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        self.att1 = BiasedSqueezeAndExcitation(mid_ch * 4) if attention else nn.Identity()
        self.norm = nn.BatchNorm2d(mid_ch * 4)

        #self.shuffle = ChannelShuffle(4)
        #self.convmid = nn.Conv2d(mid_ch * 4, in_ch, kernel_size=1, groups=4)
        self.shuffle = nn.Identity()
        self.convmid = nn.Conv2d(mid_ch * 4, in_ch, kernel_size=1)

        self.att2 = ChannelAndSpatialAttention(in_ch) if attention else nn.Identity()
        self.act = nn.ReLU()
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
    def __init__(self, in_ch, out_ch, attention=False):
        super().__init__()
        mid_ch = in_ch // 2

        self.convin = ConvNormAct(in_ch, mid_ch, kernel_size=1, attention=attention)
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
    def __init__(self, config):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
            nn.BatchNorm2d(4)
        )

        self.backbone = nn.Sequential(
            # 80x60 input

            # 40x30
            DownsampleConv(4, 4),
            SpatialPyramidPool(4),

            # 20x15
            DownsampleConv(4, 8),
            SpatialPyramidPool(8),

            DenseBlock(
                # 10x8
                DownsampleConv(8, 16),
                ConvNeXt(16, expansion_ratio=2, groups=4),

                ResidualBlock(
                    # 5x4
                    DownsampleConv(16, 16),

                    ConvNeXt(16, expansion_ratio=2, groups=4),
                    nn.Conv2d(16, 16, kernel_size=1),
                    ConvNeXt(16, expansion_ratio=2, groups=4),

                    # 10x8
                    UpsampleConv(16, 16, scale_factor=2)
                ),

                ConvNeXt(16, expansion_ratio=2, groups=4),

                # 20x15
                UpsampleConv(16, 8, scale_factor=2)
            ),

            SSPF(16, 1)
        )

        self.head = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head(x)
        return x

    def detect(self, x):
        return DetectionHead()(x.to('cpu'))

    def deploy(self, finetuning=False):
        # Deploy sub modules
        for m in self.modules():
            if hasattr(m, 'deploy') and self != m:
                m.deploy()

        # Add the final detection head directly into the model
        if not finetuning:
            self.head = DetectionHead()

            # Perform activation inplace
            for m in self.modules():
                if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True
