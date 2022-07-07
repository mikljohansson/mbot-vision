import argparse
import os

import numpy as np
import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

from src.dataset import ImageDataset
from src.model import create_model

# See https://github.com/sithu31296/PyTorch-ONNX-TFLite

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument("-m", "--model", type=str, default=None, help="Input model in .pth format")
parser.add_argument("-d", "--dataset", required=True, help="Directory of sample images")
args = parser.parse_args()

checkpoint = torch.load(args.model)
model, cfg = create_model()
model.load_state_dict(checkpoint['state'])
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
        yield {'input': tf.dtypes.cast(inputs.numpy(), tf.float32)}

        step += 1
        if step >= num_calibration_steps:
            break

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

tflite_model_path = os.path.splitext(args.model)[0] + '.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Load the model file and show details
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

print("\nModel details:")
for i, d in enumerate(interpreter.get_input_details()):
    print(f"Input {i} shape: {d['shape']} type: {d['dtype'].__name__}")
for i, d in enumerate(interpreter.get_output_details()):
    print(f"Output {i} shape: {d['shape']} type: {d['dtype'].__name__}")

import tflite
with open(tflite_model_path, 'rb') as f:
    buf = f.read()
    parsed_tflite_model = tflite.Model.GetRootAsModel(buf, 0)

opcodes = set()
for i in range(parsed_tflite_model.OperatorCodesLength()):
    op = parsed_tflite_model.OperatorCodes(i)
    if op.CustomCode() is not None:
        opcodes.add("CUSTOM_" + i)
    else:
        opcodes.add(tflite.opcode2name(op.BuiltinCode()))

print("Opcodes used:", ' '.join(sorted(opcodes)))
print("\nEnsure you add all opcodes to the MicroMutableOpResolver in eye/src/detector/objectdetector.cpp")
