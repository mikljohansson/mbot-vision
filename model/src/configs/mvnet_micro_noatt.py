from src.models.mvnet import MVNetModel

model = dict(
    type=MVNetModel,
    input_size=(160, 96),   # WxH
    output_size=(20, 12),   # WxH,
    attention=False,
    memory=False,
    channels=[3, 6, 6, 12, 24, 48]
)

solver = dict(
    optim='Adam',
    lr_scheduler='Cosine',
    lr0=1e-4,
    lrf=1e-5,
    momentum=0.9,
    weight_decay=0.0,
    warmup_epochs=100.0,
    warmup_momentum=0.9,
    warmup_bias_lr=1e-4
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
