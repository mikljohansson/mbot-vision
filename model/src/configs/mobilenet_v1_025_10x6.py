from pytorchcv.models.mobilenet import mobilenet_wd4

from src.models.mobilenet_v1 import MobileNetSegmentV1, CaptureTensor

backbone = mobilenet_wd4(pretrained=True)

# Remove the object classifier
del backbone.output

# Delete the AvgPool
del backbone.features[-1:]

# Remove the last downsampling block which has too many channels
del backbone.features[-1]

# Inject tensor capturing to emulate a UNet by concatenating high-resolution tensors to the SegmentationNeck
captures = []
backbone.features[-2].append(CaptureTensor(captures))
backbone.captures = (captures, (64,))

model = dict(
    type=MobileNetSegmentV1,
    backbone=backbone,
    backbone_out_ch=128,
    input_size=(160, 96),   # WxH
    output_size=(20, 12),   # WxH
    upsampling_factor=1
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
