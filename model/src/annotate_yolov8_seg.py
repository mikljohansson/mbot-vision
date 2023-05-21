import os
import glob
import argparse

import numpy as np
import torch
import torchvision.transforms
import tqdm
import torch.nn.functional as F

from PIL import Image
from ultralytics import YOLO

from src.image import get_crop_box

parser = argparse.ArgumentParser(description='Annotate any COCO objects with YOLOv8 in image segmentation mode')
parser.add_argument('-i', '--input', required=True, help='Directory of images to process')
parser.add_argument('-a', '--annotated', required=True, help='Directory to store annotated images')
parser.add_argument('-t', '--train', required=True, help='Directory to store training images')
parser.add_argument('-c', '--classes', required=True, help='Comma separated list of classes of interest')
parser.add_argument('-m', '--model', default='yolov8x-seg.pt', help='What YOLOv8 segmentation model to use, see Segmentation at https://github.com/ultralytics/ultralytics#models')
parser.add_argument('--batch-size', type=int, help='Batch size', default=32)
parser.add_argument('--input-width', type=int, help='Input width', default=160)
parser.add_argument('--input-height', type=int, help='Input height', default=120)
args = parser.parse_args()


def output_exists(filename):
    targetname = os.path.join(args.train, os.path.splitext(os.path.basename(filename))[0] + '.png')
    return os.path.exists(targetname)


labels_of_interest = set(filter(None, args.classes.split(',')))
files = glob.glob(os.path.join(args.input, '*.jpg'))
files = list(filter(lambda f: not output_exists(f), files))

model = YOLO(args.model)
model.overrides['verbose'] = False
pbar = tqdm.tqdm(total=len(files))

for i in range(0, len(files), args.batch_size):
    batch = files[i:(i + args.batch_size)]
    try:
        results = model(batch)
    except OSError:
        # Some training images are likely corrupt
        filtered_batch = []
        for filename in batch:
            try:
                image = Image.open(filename).load()
                filtered_batch.append(filename)
            except OSError:
                print(filename, "is corrupt")

        batch = filtered_batch
        results = model(batch)

    for filename, result in zip(batch, results):
        if not result.boxes or not result.masks:
            continue

        image = Image.open(filename).convert('RGB')
        mask = torch.zeros((1, image.height, image.width))
        found = False

        for clsid, confidence, instance_mask in zip(result.boxes.cls, result.boxes.conf, result.masks):
            classname = result.names[int(clsid)]
            if classname not in labels_of_interest or confidence < 0.05:
                continue

            instance_mask = F.interpolate(instance_mask.data.unsqueeze(0), size=mask.shape[1:], mode='bilinear')
            mask += instance_mask[0].cpu()
            found = True

        if not found:
            continue

        mask = mask.clamp(0., 1.)
        mask = torchvision.transforms.functional.to_pil_image(mask, mode='L')

        # Adjust to the target aspect ratio
        box = get_crop_box(np.asarray(mask, dtype=np.uint8), args.input_width, args.input_height)
        image = image.resize((args.input_width, args.input_height), box=box, resample=Image.Resampling.BILINEAR)
        mask = mask.resize((args.input_width, args.input_height), box=box, resample=Image.Resampling.BILINEAR)

        #maskname = os.path.join(args.train, os.path.splitext(os.path.basename(filename))[0] + '-mask.png')
        #mask.save(maskname)

        # Show the background with 25% transparency, just for debugging purposes.
        mask = Image.fromarray(np.maximum(np.asarray(mask, dtype=np.uint8), 64), 'L')
        image = Image.merge('RGBA', (*image.split(), *mask.split()))

        targetname = os.path.join(args.train, os.path.splitext(os.path.basename(filename))[0] + '.png')
        image.save(targetname)

    #results.display(save=True, save_dir=Path(args.annotated))
    pbar.update(len(batch))
