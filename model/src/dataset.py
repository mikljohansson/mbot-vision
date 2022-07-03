import os
import glob

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

def normalize(image):
    # Normalize color values
    image = (image - mean) / std

    # Change from range [0, 1] to [-1, 1]
    image = (image - 0.5) / 0.5

    return image

def denormalize(image):
    # Change from range [-1, 1] to [0, 1]
    image = image * 0.5 + 0.5

    # Normalize color values
    image = (image * std) + mean

    return image

class ImageDataset(Dataset):
    def __init__(self, images_path):
        super(ImageDataset, self).__init__()

        self.images = glob.glob(os.path.join(images_path, '*.png'))
        self.transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.GaussianBlur(3),
            transforms.ColorJitter(),
            transforms.RandomGrayscale(p=0.2),
        )

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
        image = normalize(image)
        return image, target
