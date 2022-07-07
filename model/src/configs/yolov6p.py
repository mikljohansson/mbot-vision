from src.models.yolo import YOLOv6Model

model = dict(
    type=YOLOv6Model,
    pretrained=None,
    input_size=(160, 120),  # WxH
    output_size=(80, 60),   # WxH
    depth_multiple=0.33,
    width_multiple=0.25,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[32, 32, 64, 128, 256],
        ),
    neck=dict(
        type='RepPAN',
        num_repeats=[12, 12, 12, 12],
        out_channels=[128, 64, 64, 128, 128, 256],
        ),
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
