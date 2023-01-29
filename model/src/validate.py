import argparse
import os.path

import numpy as np
import torch
import tensorflow as tf
import torchvision.transforms.functional
from PIL import Image

from src.dataset import ImageDataset, denormalize
from src.model import create_model

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument("-m", "--torch-model", required=True, help="Input model in .pth format")
parser.add_argument("-t", "--tflite-model", required=True, help="Input model in .tflite format")
parser.add_argument("-d", "--dataset", required=True, help="Directory of sample images")
parser.add_argument("--channels-last", action="store_true", help="Use channels-last format for the input image")
args = parser.parse_args()

checkpoint = torch.load(args.torch_model)
model, cfg = create_model(checkpoint['model'])
dataset = ImageDataset(args.dataset, input_size=cfg.model.input_size, target_size=cfg.model.output_size, apply_transforms=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
tflite_inputs = interpreter.get_input_details()
tflite_outputs = interpreter.get_output_details()

print("\nTflite model details:")
for i, d in enumerate(tflite_inputs):
    print(f"Input {i} name: {d['name']}, shape: {d['shape']}, type: {d['dtype'].__name__}")
for i, d in enumerate(tflite_outputs):
    print(f"Output {i} name: {d['name']}, shape: {d['shape']}, type: {d['dtype'].__name__}")

# Load the PyTorch model
model.load_state_dict(checkpoint['state'])
model.deploy()
model.eval()

image_count = 0
for inputs, _, _ in dataloader:
    os.makedirs(os.path.join(os.path.dirname(args.torch_model), 'validation'), exist_ok=True)
    image = torchvision.transforms.functional.to_pil_image(denormalize(inputs[0]))
    image.save(os.path.join(os.path.dirname(args.torch_model), 'validation/%03d-image.png' % image_count))

    with torch.inference_mode():
        results = model(inputs)
    mask = results[0]
    mask = torchvision.transforms.functional.to_pil_image(mask)
    mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)
    mask.save(os.path.join(os.path.dirname(args.torch_model), 'validation/%03d-torch.png' % image_count))

    inputs_np = (inputs * 127).numpy()
    if args.channels_last:
        inputs_np = np.moveaxis(inputs_np, 1, 3)

    inputs_tf = tf.convert_to_tensor(inputs_np, dtype=tf.int8)
    interpreter.allocate_tensors()
    interpreter.set_tensor(tflite_inputs[0]['index'], inputs_tf)
    interpreter.invoke()
    results_tf = interpreter.get_tensor(tflite_outputs[0]['index'])

    mask_tf = (results_tf[0].astype(np.float32) + 127.) / 255.
    mask_tf = (mask_tf.clip(0., 1.) * 255.).astype(np.uint8)

    if not args.channels_last:
        mask_tf = np.moveaxis(mask_tf, 0, 2)

    mask_tf = torchvision.transforms.functional.to_pil_image(mask_tf)
    #mask_tf = torchvision.transforms.functional.to_pil_image((mask_tf * 255).astype(np.uint8))
    mask_tf = mask_tf.resize(image.size, resample=Image.Resampling.NEAREST)
    mask_tf.save(os.path.join(os.path.dirname(args.torch_model), 'validation/%03d-tflite.png' % image_count))

    image_count += 1
    print('.', end='')

    if image_count >= 5:
        break
