import os
import glob

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, images_path, target_size):
        super(ImageDataset, self).__init__()
        self.target_size = target_size

        self.images = glob.glob(os.path.join(images_path, '*.png'))
        self.transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.GaussianBlur(3),
            transforms.ColorJitter(),
            transforms.RandomGrayscale(p=0.2),
        )

        self.gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGBA')

        # Convert images to tensors, PIL transforms don't work well with RGBA format because it'll convert to
        # RGBa (premultiplied alpha) when doing resizing and other operations.
        image = torchvision.transforms.functional.to_tensor(image)

        # Apply randomized transforms
        #image = self.transforms(image)

        target = image[[3]]
        image = image[0:3]

        # Downsample to same size as model output
        output_shape = (self.target_size[1], self.target_size[0])
        target = F.interpolate(target.unsqueeze(0), size=output_shape, mode='bilinear', align_corners=False)[0]

        # Generate a unknown mask around the edges, so the loss function can ignore that. Creates the
        # object outline by taking the difference between dilation and erosion
        unknown_mask = cv2.morphologyEx((target[-2:] * 255).numpy().astype(dtype=np.float32),
                                        cv2.MORPH_GRADIENT, self.gradient_kernel, iterations=1)
        unknown_mask = torch.tensor(unknown_mask, dtype=torch.float32) / 255

        return image, target, unknown_mask
