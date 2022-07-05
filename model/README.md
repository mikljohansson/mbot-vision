# Installation

Use IntelliJ settings to make a new venv Python interpreter in u-2-net/venv

```
sudo apt-get install ffmpeg

. venv/bin/activate

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
pip install -r requirements.txt
```

# Training

```
CUDA_VISIBLE_DEVICES=0 make dataset
CUDA_VISIBLE_DEVICES=0 PARALLEL=2 make train
```
