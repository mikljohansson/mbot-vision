import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument('-i', '--input', required=True, help='Directory of images to process')
parser.add_argument('-a', '--annotated', required=True, help='Directory to store annotated images')
parser.add_argument('-t', '--train', required=True, help='Directory to store training images')
parser.add_argument('--batch-size', type=int, help='Batch size', default=64)
parser.add_argument('--target-width', type=int, help='Input width', default=160)
parser.add_argument('--target-height', type=int, help='Input height', default=120)
args = parser.parse_args()

labels_of_interest = {'sports ball', 'apple', 'orange'}
files = glob.glob(os.path.join(args.input, '*.jpg'))

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', trust_repo=True)


def get_input_box(image):
    # https://stackoverflow.com/a/4744625
    input_aspect = float(image.width) / float(image.height)
    target_aspect = float(args.target_width) / float(args.target_height)

    if input_aspect > target_aspect:
        # Then crop the left and right edges:
        new_width = int(target_aspect * image.height)
        offset = (image.width - new_width) / 2
        resize = (offset, 0, image.width - offset, image.height)
    else:
        # ... crop the top and bottom:
        new_height = int(image.width / target_aspect)
        offset = (image.height - new_height) / 2
        resize = (0, offset, image.width, image.height - offset)

    return resize


for i in range(0, len(files), args.batch_size):
    batch = files[i:(i + args.batch_size)]
    results = model(files)

    for filename, image, predictions in zip(results.files, results.imgs, results.pred):
        image = Image.fromarray(image, 'RGB') if isinstance(image, np.ndarray) else image
        mask = Image.new('L', image.size)
        draw = ImageDraw.Draw(mask)

        for *xyxy, confidence, clsid in predictions:
            classname = results.names[int(clsid)]
            if classname not in labels_of_interest:
                continue

            draw.ellipse(((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))), fill=255)

        image = image.resize((args.target_width, args.target_height), box=get_input_box(image), resample=Image.Resampling.LANCZOS)
        mask = mask.resize((args.target_width, args.target_height), box=get_input_box(mask), resample=Image.Resampling.LANCZOS)
        image = Image.merge('RGBA', (*image.split(), *mask.split()))

        targetname = os.path.join(args.train, os.path.splitext(os.path.basename(filename))[0] + '.png')
        image.save(targetname)

    results.display(save=True, save_dir=Path(args.annotated))
