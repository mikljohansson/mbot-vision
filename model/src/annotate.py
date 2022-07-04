import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.image import get_input_box

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument('-i', '--input', required=True, help='Directory of images to process')
parser.add_argument('-a', '--annotated', required=True, help='Directory to store annotated images')
parser.add_argument('-t', '--train', required=True, help='Directory to store training images')
parser.add_argument('-c', '--classes', required=True, help='Comma separated list of classes of interest')
parser.add_argument('--batch-size', type=int, help='Batch size', default=64)
parser.add_argument('--target-width', type=int, help='Input width', default=160)
parser.add_argument('--target-height', type=int, help='Input height', default=120)
args = parser.parse_args()

labels_of_interest = set(args.classes.split(','))
files = glob.glob(os.path.join(args.input, '*.jpg'))

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', trust_repo=True)

for i in range(0, len(files), args.batch_size):
    batch = files[i:(i + args.batch_size)]
    results = model(batch)

    for filename, image, predictions in zip(results.files, results.imgs, results.pred):
        image = Image.fromarray(image, 'RGB') if isinstance(image, np.ndarray) else image
        mask = Image.new('L', image.size)
        draw = ImageDraw.Draw(mask)

        for *xyxy, confidence, clsid in predictions:
            classname = results.names[int(clsid)]
            if classname not in labels_of_interest:
                continue

            draw.ellipse(((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))), fill=255)

        box = get_input_box(np.asarray(mask, dtype=np.uint8), args.target_width, args.target_height)
        image = image.resize((args.target_width, args.target_height), box=box, resample=Image.Resampling.LANCZOS)
        mask = mask.resize((args.target_width, args.target_height), box=box, resample=Image.Resampling.LANCZOS)
        image = Image.merge('RGBA', (*image.split(), *mask.split()))

        targetname = os.path.join(args.train, os.path.splitext(os.path.basename(filename))[0] + '.png')
        image.save(targetname)

    results.display(save=True, save_dir=Path(args.annotated))
