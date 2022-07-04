import argparse
import os

import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

from src.dataset import ImageDataset
from src.model import create_model_cfg

# See https://github.com/sithu31296/PyTorch-ONNX-TFLite

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument("-m", "--model", type=str, default=None, help="Input model in .pth format")
parser.add_argument("-d", "--dataset", required=True, help="Directory of sample images")
args = parser.parse_args()

model, cfg = create_model_cfg()
model.load_state_dict(torch.load(args.model))
model.deploy()
model.eval()

dataset = ImageDataset(args.dataset, target_size=cfg.model.output_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Convert to ONNX and TFLite
inputs, _, _ = next(iter(dataloader))
onnx_model_path = os.path.splitext(args.model)[0] + '.onnx'
torch.onnx.export(model, inputs, onnx_model_path,
                  opset_version=12, export_params=True, verbose=False,
                  input_names=['input'], output_names=['output'])

# Verify the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
#print(onnx.helper.printable_graph(onnx_model.graph))

# Convert to TF
tf_model_path = os.path.splitext(args.model)[0] + '.tf'
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

# Convert to TFLite and quantize to int8
num_calibration_steps = 100

def representative_dataset_gen():
    step = 0
    for inputs, _, _ in dataloader:
        # get sample input data as numpy array
        yield [inputs.numpy()]

        step += 1
        if step >= num_calibration_steps:
            break

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

tflite_model_path = os.path.splitext(args.model)[0] + '.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
