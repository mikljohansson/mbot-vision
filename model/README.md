# Installation

Use IntelliJ settings to make a new venv Python interpreter in u-2-net/venv

```
. venv/bin/activate

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
pip install -r requirements.txt

cd ..
git clone https://github.com/onnx/onnx-tensorflow.git
cd onnx-tensorflow
pip install -e .
```

# Training

```
CUDA_VISIBLE_DEVICES=0 make dataset
CUDA_VISIBLE_DEVICES=0 make train
```
