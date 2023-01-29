import argparse
import os
import shutil

import numpy as np
import torch
import onnx
import tensorflow as tf
from onnx2tf import convert
from onnx_tf.backend import prepare

from src.convert_onnx_int64 import convert_model_to_int32
from src.dataset import ImageDataset
from src.model import create_model

# See https://github.com/sithu31296/PyTorch-ONNX-TFLite

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument("-m", "--model", type=str, default=None, help="Input model in .pth format")
parser.add_argument("-d", "--dataset", required=True, help="Directory of sample images")
parser.add_argument("--channels-last", action="store_true",
                    help="Convert to channels-last format. Avoids lots of TRANSPOSE operators and improves performance")
args = parser.parse_args()

checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
model, cfg = create_model(checkpoint['model'])
model.load_state_dict(checkpoint['state'])
if hasattr(model, 'deploy'):
    model.deploy()
model.eval()

dataset = ImageDataset(args.dataset, input_size=cfg.model.input_size, target_size=cfg.model.output_size, apply_transforms=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Convert to ONNX
inputs, _, _ = next(iter(dataloader))
onnx_model_path = os.path.splitext(args.model)[0] + '.onnx'
torch.onnx.export(model, inputs, onnx_model_path + '.int64',
                  opset_version=12, export_params=True, verbose=False,
                  input_names=['input'], output_names=['output'])

# Convert INT64 tensors to INT32
#convert_model_to_int32(onnx_model_path + '.int64', onnx_model_path)
shutil.copyfile(onnx_model_path + '.int64', onnx_model_path)

# Verify the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

# Print a summary of the ONNX model
print(onnx.helper.printable_graph(onnx_model.graph))

# Convert to tflite
num_calibration_steps = 100
tflite_model_path = os.path.splitext(args.model)[0] + '.tflite'

if args.channels_last:
    tf_model_path = os.path.splitext(args.model)[0] + '-tflite'
    np_input0_path = os.path.splitext(args.model)[0] + '-input0.npy'

    def representative_dataset_gen():
        step = 0
        images = []

        for inputs, _, _ in dataloader:
            # get sample input data as numpy array
            images.append(np.moveaxis(inputs.numpy(), 1, 3))

            step += 1
            if step >= num_calibration_steps:
                break

        return np.asarray(images, np.float32)

    np.save(np_input0_path, representative_dataset_gen())
    mean = np.asarray([[[[0., 0., 0.]]]], np.float32)
    std = np.asarray([[[[1., 1., 1.]]]], np.float32)

    tf_rep = convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=tf_model_path,
        output_integer_quantized_tflite=True,
        quant_calib_input_op_name_np_data_path=[['input', np_input0_path, mean, std]],
        input_output_quant_dtype="int8"
    )

    shutil.copyfile(tf_model_path + '/model_full_integer_quant.tflite', tflite_model_path)
else:
    # Convert to TF
    tf_model_path = os.path.splitext(args.model)[0] + '.tf'
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)

    # Convert to TFlite
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
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)


# Load the model file and show details
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

print("\nTflite model details:")
for i, d in enumerate(interpreter.get_input_details()):
    print(f"Input {i} name: {d['name']}, shape: {d['shape']}, type: {d['dtype'].__name__}")
for i, d in enumerate(interpreter.get_output_details()):
    print(f"Output {i} name: {d['name']}, shape: {d['shape']}, type: {d['dtype'].__name__}")

signatures = interpreter.get_signature_list()
print("Signatures:", signatures)

# Print a summary of the tflite model
tf.lite.experimental.Analyzer.analyze(model_path=tflite_model_path)

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
print("\nEnsure you add all opcodes to the MicroMutableOpResolver in eye/src/detector/objectdetector-tflite.h")
print("\nModel saved in", tflite_model_path)