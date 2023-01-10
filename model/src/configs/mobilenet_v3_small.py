from torchvision.models import mobilenet_v3_small

from src.models.mobilenet import MobileNetModel

model = dict(
    type=MobileNetModel,
    backbone=mobilenet_v3_small(pretrained=True),
    backbone_out_ch=96,
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
