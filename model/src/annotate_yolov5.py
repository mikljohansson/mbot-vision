import os
import glob
import argparse

import numpy as np
import torch
import tqdm
from PIL import Image, ImageDraw

from src.image import get_crop_box

parser = argparse.ArgumentParser(description='Annotate sports balls and other round objects with YOLOv5')
parser.add_argument('-i', '--input', required=True, help='Directory of images to process')
parser.add_argument('-a', '--annotated', required=True, help='Directory to store annotated images')
parser.add_argument('-t', '--train', required=True, help='Directory to store training images')
parser.add_argument('-c', '--classes', required=True, help='Comma separated list of classes of interest')
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

model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
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
                image = Image.open(filename)
                image.load()
                filtered_batch.append(filename)
            except OSError:
                print(filename, "is corrupt")

        batch = filtered_batch
        results = model(batch)

    for filename, image, predictions in zip(results.files, results.ims, results.pred):
        image = Image.fromarray(image, 'RGB') if isinstance(image, np.ndarray) else image
        mask = Image.new('L', image.size)
        draw = ImageDraw.Draw(mask)
        found = False

        for *xyxy, confidence, clsid in predictions:
            classname = results.names[int(clsid)]
            if classname not in labels_of_interest or confidence < 0.25:
                continue

            draw.ellipse(((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))), fill=255)
            found = True

        if not found:
            continue

        # Adjust to the target aspect ratio
        box = get_crop_box(np.asarray(mask, dtype=np.uint8), args.input_width, args.input_height)
        image = image.resize((args.input_width, args.input_height), box=box, resample=Image.Resampling.BILINEAR)
        mask = mask.resize((args.input_width, args.input_height), box=box, resample=Image.Resampling.BILINEAR)

        # Show the background with 25% transparency, just for debugging purposes.
        mask = Image.fromarray(np.maximum(np.asarray(mask, dtype=np.uint8), 64), 'L')
        image = Image.merge('RGBA', (*image.split(), *mask.split()))

        targetname = os.path.join(args.train, os.path.splitext(os.path.basename(filename))[0] + '.png')
        image.save(targetname)

    #results.display(save=True, save_dir=Path(args.annotated))
    pbar.update(len(batch))
