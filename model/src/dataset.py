import os
import glob
import random

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def normalize(image):
    return image * 2. - 1.


def denormalize(image):
    return (image + 1.) / 2.


class ImageDataset(Dataset):
    def __init__(self, images_path, input_size, target_size, apply_transforms=True):
        super(ImageDataset, self).__init__()
        self.target_size = target_size

        self.images = glob.glob(os.path.join(images_path, '*.png'))
        random.shuffle(self.images)

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
        image = Image.open(self.images[index]).convert('RGBA')

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

        return image, target, unknown_mask
