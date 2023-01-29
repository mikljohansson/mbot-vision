import os
import argparse

import numpy as np
from PIL import Image
import fiftyone.zoo as foz

from src.image import get_input_box

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument('-c', '--classes', required=True, help='Comma separated list of classes of interest')
parser.add_argument('-e', '--exclude-classes', default='', help='Comma separated list of classes to exclude from negative sampling')
parser.add_argument('-t', '--train', required=True, help='Directory to store training images')
parser.add_argument('--input-width', type=int, help='Input width', default=160)
parser.add_argument('--input-height', type=int, help='Input height', default=120)
parser.add_argument('--max-sample-count', required=True, type=int, help='Number of samples to try and get')
args = parser.parse_args()

# See https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4

labels_of_interest = list(filter(None, args.classes.split(',')))
exclude_labels = list(filter(None, args.exclude_classes.split(',')))
coco_labels = set(line.strip() for line in open(os.path.join(os.path.dirname(__file__), 'coco-labels-2014_2017.txt'), 'r'))
coco_bad_samples = set(line.strip() for line in open(os.path.join(os.path.dirname(__file__), 'coco-bad-samples.txt'), 'r'))


def render_sample(sample, classes, positive_sample=True):
    if os.path.basename(sample.filepath) in coco_bad_samples:
        return False

    try:
        image = Image.open(sample.filepath).convert('RGB')
    except OSError:
        return False

    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    found_roi = not positive_sample

    for detection in sample.ground_truth.detections:
        # Don't output masks for detections not in the positive classes
        if positive_sample and detection.label not in classes:
            continue

        # Don't output samples which include one of the positive classes
        if not positive_sample and detection.label in classes:
            return False

        # Render a target mask for positive samples
        if positive_sample:
            # Avoid rendering too small objects
            segmentation = detection.to_segmentation(frame_size=(image.width, image.height))
            target_ratio = np.sum(segmentation.mask) / 255 / (image.width * image.height)
            if target_ratio < 0.00025:
                continue

            mask += segmentation.mask
            found_roi = True

    if not found_roi:
        return False

    box = get_input_box(mask, args.input_width, args.input_height)
    mask = Image.fromarray(np.minimum(mask, 255), 'L')

    assert image.width == mask.width and image.height == mask.height
    mask = mask.resize((args.input_width, args.input_height), box=box, resample=Image.Resampling.LANCZOS)
    image = image.resize((args.input_width, args.input_height), box=box, resample=Image.Resampling.LANCZOS)

    # Show the background with 25% transparency, just for debugging purposes.
    mask = Image.fromarray(np.maximum(np.asarray(mask, dtype=np.uint8), 64), 'L')

    image = Image.merge('RGBA', (*image.split(), *mask.split()))

    targetname = os.path.join(args.train, os.path.splitext(os.path.basename(sample.filepath))[0] + '.png')
    image.save(targetname)
    return True


dataset = foz.load_zoo_dataset(
    "coco-2017",
    dataset_name='coco-positive',
    drop_existing_dataset=True,
    split="train",
    label_types=["segmentations"],
    classes=labels_of_interest,
    max_samples=args.max_sample_count
)

try:
    positive_sample_count = 0
    rejected_positive_samples = 0

    negative_sample_count = 0
    rejected_negative_samples = 0

    for sample in dataset:
        if render_sample(sample, labels_of_interest):
            positive_sample_count += 1
        else:
            rejected_positive_samples += 1

    negative_dataset = foz.load_zoo_dataset(
        "coco-2017",
        dataset_name='coco-negative',
        drop_existing_dataset=True,
        split="train",
        label_types=["segmentations"],
        classes=list(coco_labels - set(labels_of_interest) - set(exclude_labels)),
        max_samples=len(dataset) * 3
    )

    for sample in negative_dataset:
        if render_sample(sample, set(labels_of_interest + exclude_labels), False):
            negative_sample_count += 1
        else:
            rejected_negative_samples += 1
        if negative_sample_count >= positive_sample_count:
            break

    print({'positive': positive_sample_count, 'rejected_positive': rejected_positive_samples,
           'negative': negative_sample_count, 'rejected_negative': rejected_negative_samples})
finally:
    dataset.delete()
    negative_dataset.delete()
