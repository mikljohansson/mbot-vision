# Setup

Use IntelliJ settings to make a new venv Python interpreter in model/venv or execute `python3 -m venv venv`

```
sudo apt-get install ffmpeg

. venv/bin/activate

pip install torch==1.11.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
pip install -r requirements.txt
```

# Model summary

```
# Show PyTorch model summary
make summary

# Show information about a quantized TFLite model
EXPERIMENT=experiments/mobilenet-micro/20220708-100725F make summary-tflite
```

# Training

Put training videos into `dataset/recorded/videos` and training images into `dataset/recorded/images`. 

Edit Makefile PRIMARY_CLASSES to include the classes you're interested in detecting, see `src/coco-labels-2014_2017.txt` for a list of available labels.

Edit Makefile SECONDARY_CLASSES to include any classes that are often confused with the primary classes.

Run `make dataset` to create the training dataset. This will:

* Download relevant images from the MS COCO dataset
* Extract frames from your recorded videos
* Use YOLOv5 to annotate your recorded frames and images
* Create training samples from COCO and recorded frames/images

Run `make train` to train and package the model. This will:

* Pre-train on both MS COCO images labelled with your selected classes and some negative samples
* Fine-tune on your record videos and images
* Quantize to int8 and convert to TFlite
* Convert to C++ files which are symlinked into the `mbot-vision/eye` project

Examples:
```
# Start tensorboard in the background
make tensorboard

# Will use GPU if available to run the YOLO model used for annotating images
make dataset

# Use specific GPU for training. Will use all available GPU's if CUDA_VISIBLE_DEVICES isn't set.
CUDA_VISIBLE_DEVICES=0 PARALLEL=2 make train
```

If you use multiple GPU's to train, or if you change the batch size then you'd need to also tune 
the *_ACCUMULATION_STEPS variables in the Makefile. Ensure that the total batch size 
(`BATCH_SIZE * ACCUMULATION_STEPS * GPU_COUNT`) is 64-256 samples.
