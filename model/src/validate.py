import argparse
import os.path

import numpy as np
import torch
import tensorflow as tf
import torchvision.transforms.functional
from PIL import Image

from src.dataset import ImageDataset
from src.model import create_model_cfg

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument("-m", "--torch-model", required=True, help="Input model in .pth format")
parser.add_argument("-t", "--tflite-model", required=True, help="Input model in .tflite format")
parser.add_argument("-d", "--dataset", required=True, help="Directory of sample images")
args = parser.parse_args()

model, cfg = create_model_cfg()
model.load_state_dict(torch.load(args.torch_model))
model.deploy()
model.eval()

dataset = ImageDataset(args.dataset, target_size=cfg.model.output_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Run some sample images through the TFLite model
interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
signature = interpreter.get_signature_runner()

image_count = 0
for inputs, _, _ in dataloader:
    os.makedirs(os.path.join(os.path.dirname(args.torch_model), 'validation'), exist_ok=True)
    image = torchvision.transforms.functional.to_pil_image(inputs[0])
    image.save(os.path.join(os.path.dirname(args.torch_model), 'validation/%03d-image.png' % image_count))

    results = model(inputs)
    mask = torchvision.transforms.functional.to_pil_image(results[0])
    mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)
    mask.save(os.path.join(os.path.dirname(args.torch_model), 'validation/%03d-torch.png' % image_count))

    inputs_tf = tf.convert_to_tensor((inputs * 255).numpy(), dtype=tf.uint8)
    #inputs_tf = tf.convert_to_tensor(inputs.numpy(), dtype=tf.float32)
    results_tf = signature(input=inputs_tf)

    mask_tf = np.moveaxis(results_tf['output'][0], 0, 2)
    mask_tf = torchvision.transforms.functional.to_pil_image(mask_tf)
    #mask_tf = torchvision.transforms.functional.to_pil_image((mask_tf * 255).astype(np.uint8))
    mask_tf = mask_tf.resize(image.size, resample=Image.Resampling.NEAREST)
    mask_tf.save(os.path.join(os.path.dirname(args.torch_model), 'validation/%03d-tflite.png' % image_count))

    image_count += 1

    if image_count >= 5:
        break
