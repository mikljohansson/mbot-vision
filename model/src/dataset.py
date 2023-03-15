import os
import glob

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import transforms


def normalize(image):
    return image * 2. - 1.


def denormalize(image):
    return (image + 1.) / 2.


class ImageDataset(Dataset):
    def __init__(self, images_path, input_size, target_size, apply_transforms=False):
        super(ImageDataset, self).__init__()
        self.target_size = target_size
        self.images = glob.glob(os.path.join(images_path, '*.png'))
        self.images.sort()

        self.spatial_transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation((-180, 180))]), p=0.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(size=tuple(reversed(input_size)), scale=(0.5, 1.0),)]), p=0.2),
            transforms.RandomPerspective(p=0.2)
        ) if apply_transforms else nn.Identity()

        self.color_transforms = torch.nn.Sequential(
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAdjustSharpness(sharpness_factor=2)]), p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3)]), p=0.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.2),
            #transforms.RandomGrayscale(p=0.2),
        ) if apply_transforms else nn.Identity()

        self.gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGBA')

        # Convert images to tensors, PIL transforms don't work well with RGBA format because it'll convert to
        # RGBa (premultiplied alpha) when doing resizing and other operations.
        image = torchvision.transforms.functional.to_tensor(image)

        # Apply randomized spatial transforms
        image = self.spatial_transforms(image)

        # Separate the RGB and alpha channels
        target = image[[3]]
        image = image[0:3]

        # Apply randomized color transforms
        image = self.color_transforms(image)

        # Clip transparency less than 75%
        target = F.threshold(target, 0.5, 0.)

        # Downsample to same size as model output
        output_shape = (self.target_size[1], self.target_size[0])
        target = F.interpolate(target.unsqueeze(0), size=output_shape, mode='bilinear', align_corners=False)[0]

        # Generate a unknown mask around the edges, so the loss function can ignore that. Creates the
        # object outline by taking the difference between dilation and erosion
        unknown_mask = cv2.morphologyEx((target[-2:] * 255).numpy().astype(dtype=np.float32),
                                        cv2.MORPH_GRADIENT, self.gradient_kernel, iterations=1)
        unknown_mask = torch.tensor(unknown_mask, dtype=torch.float32) / 255

        # Change to range [-1, 1]
        image = normalize(image)

        return image, target, unknown_mask, image_path


class StridedSampler(Sampler):
    """
    Samples randomly but with a stride, to get the next image from a sequence at the same position in the next batch
    """
    def __init__(self, dataset, stride):
        super().__init__(dataset)
        self.dataset = dataset
        self.stride = stride

    def __iter__(self):
        dataset_len = len(self.dataset)
        partition_len = dataset_len // self.stride

        for offset in range(partition_len):
            for partition in range(self.stride):
                yield partition_len * partition + offset

    def __len__(self):
        return len(self.dataset) // self.stride * self.stride


def is_next_frame(prev, next):
    prev_parts = prev.split('.')[0].split('_')
    next_parts = next.split('.')[0].split('_')

    prev_prefix = '_'.join(prev_parts[0:-1])
    next_prefix = '_'.join(next_parts[0:-1])

    prev_seqno = prev_parts[-1]
    next_seqno = next_parts[-1]

    try:
        return prev_prefix == next_prefix and int(prev_seqno) + 1 == int(next_seqno)
    except ValueError:
        return False
