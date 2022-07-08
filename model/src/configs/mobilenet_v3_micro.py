from functools import partial

from torch.hub import load_state_dict_from_url
from torchvision.models.mobilenetv3 import InvertedResidualConfig, MobileNetV3, model_urls

from src.models.mobilenet import MobileNetModel

# See https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L239
def _mobilenet_v3_micro_conf(width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return inverted_residual_setting, last_channel

def _mobilenet_v3_micro(
        arch,
        inverted_residual_setting,
        last_channel,
        pretrained=True,
        progress=True):
    model = MobileNetV3(inverted_residual_setting, last_channel)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        # Remove weights that would cause shape mismatch
        prefixes = ['features.9', 'features.10', 'features.11', 'features.12', 'classifier.']
        for key in list(state_dict.keys()):
            for prefix in prefixes:
                if key.startswith(prefix):
                    del state_dict[key]
                    break

        model.load_state_dict(state_dict, strict=False)
    return model

model = dict(
    type=MobileNetModel,
    backbone=_mobilenet_v3_micro("mobilenet_v3_small", *_mobilenet_v3_micro_conf(reduced_tail=True)),
    backbone_out_ch=288,
    pretrained=True,
    input_size=(160, 120),  # WxH
    output_size=(20, 16),   # WxH
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)
