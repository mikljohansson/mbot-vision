import os
import glob
import json
import argparse


import numpy as np
import torch
import torchvision.transforms.functional
from tqdm import tqdm
from PIL import Image, ImageDraw
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.segment.general import process_mask

from src.image import get_crop_box, square_pad

parser = argparse.ArgumentParser(description='Annotate any COCO objects with YOLOv5 in image segmentation mode')
parser.add_argument('-i', '--input', required=True, help='Directory of images to process')
parser.add_argument('-a', '--annotated', required=True, help='Directory to store annotated images')
parser.add_argument('-t', '--train', required=True, help='Directory to store training images')
parser.add_argument('-c', '--classes', required=True, help='Comma separated list of classes of interest')
parser.add_argument('-p', '--parallel', type=int, help='Number of worker processes', default=1)
parser.add_argument('--batch-size', type=int, help='Batch size', default=32)
parser.add_argument('--input-width', type=int, help='Input width', default=160)
parser.add_argument('--input-height', type=int, help='Input height', default=120)
args = parser.parse_args()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        super(ImageDataset).__init__()
        self.images = images
        self.sample_count = len(images)
        self.size = 640

    def __len__(self):
        return self.sample_count

    def __getitem__(self, item):
        image, padding = self.preprocess(self.images[item])
        return image, json.dumps({'path': self.images[item], 'padding': padding})

    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = torchvision.transforms.functional.to_tensor(image)

        # Resize to fit within the desired size
        height, width = image.shape[-2:]
        if height != self.size and width != self.size:
            max_hw = max(width, height)
            new_width = int(self.size * (width / max_hw))
            new_height = int(self.size * (height / max_hw))
            image = torchvision.transforms.functional.resize(image, (new_height, new_width))

        # Pad to square
        image, padding = square_pad(image, self.size, fill=0.)
        return image, padding

def output_exists(filename):
    targetname = os.path.join(args.train, os.path.splitext(os.path.basename(filename))[0] + '.png')
    return os.path.exists(targetname)


labels_of_interest = set(filter(None, args.classes.split(',')))
files = glob.glob(os.path.join(args.input, '*.jpg'))
files = list(filter(lambda f: not output_exists(f), files))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataloader = torch.utils.data.DataLoader(ImageDataset(files),
                                         num_workers=args.parallel, batch_size=args.batch_size)

model = DetectMultiBackend('yolov5x-seg.pt')
model.eval()
model.to(device)

for images, metadatas in tqdm(dataloader):
    images = images.to(device)
    images = images.half() if model.fp16 else images.float()  # uint8 to fp16/32

    with torch.inference_mode():
        predictions, proto = model(images)[:2]
        predictions = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45, nm=32)

        for i, detection in enumerate(predictions):
            if not len(detection):
                continue

            metadata = json.loads(metadatas[i])
            filename = metadata['path']
            original = Image.open(filename).convert('RGB')
            image = images[i]

            # https://github.com/ultralytics/yolov5/blob/master/segment/predict.py
            masks = process_mask(proto[i], detection[:, 6:], detection[:, :4], image.shape[1:], upsample=True)
            mask = torch.zeros(masks.shape[1:], device=masks.device)
            found = False

            for j, (*xyxy, confidence, clsid) in enumerate(detection[:, :6]):
                classname = model.names[int(clsid)]
                if classname in labels_of_interest and confidence >= 0.25:
                    found = True
                    mask += masks[j]

            if not found:
                continue

            mask = mask.clamp(0, 1).unsqueeze(0)

            # Remove padding and resize to original size
            ph, pw = mask.shape[-2:]
            left, top, right, bottom = metadata['padding']
            mask = mask[:, top:(ph - bottom), left:(pw - right)]
            mask = torchvision.transforms.functional.resize(mask, (original.height, original.width))
            mask = torchvision.transforms.functional.to_pil_image(mask, mode='L')

            # Adjust to the target aspect ratio
            box = get_crop_box(np.asarray(mask, dtype=np.uint8), args.input_width, args.input_height)
            image = original.resize((args.input_width, args.input_height), box=box, resample=Image.Resampling.BILINEAR)
            mask = mask.resize((args.input_width, args.input_height), box=box, resample=Image.Resampling.BILINEAR)

            # Show the background with 25% transparency, just for debugging purposes.
            mask = Image.fromarray(np.maximum(np.asarray(mask, dtype=np.uint8), 64), 'L')
            image = Image.merge('RGBA', (*image.split(), *mask.split()))

            targetname = os.path.join(args.train, os.path.splitext(os.path.basename(filename))[0] + '.png')
            image.save(targetname)
