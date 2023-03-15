# Coco classes of interest, see src/coco-labels-2014_2017.txt
PRIMARY_CLASSES = sports ball
SECONDARY_CLASSES = apple,orange
CLASSES = $(PRIMARY_CLASSES),$(SECONDARY_CLASSES)

INFERENCE_ENGINE ?= TFMICRO
#INFERENCE_ENGINE ?= TINYMAIX
INFERENCE_EXTENSION = $(shell echo $(INFERENCE_ENGINE) | sed "s/TINYMAIX/tmdl/" | sed "s/TFMICRO.*/tflite/")

# See this link for lots of pretrained models
# https://github.com/osmr/imgclsmob/tree/master/pytorch

# Pretrained FD-MobileNet v1 with depth_multiplier=0.25 with some model surgery to insert a UNet segmentation head into it
#MODEL ?= fdmobilenet_025_10x6
#MODEL ?= fdmobilenet_025_5x3

# Pretrained MobileNet v1 with depth_multiplier=0.25 with some model surgery to insert a UNet segmentation head into it
# https://github.com/osmr/imgclsmob/tree/master/pytorch (lots of other pretrained models available here too)
#MODEL ?= mobilenet_v1_025_10x6
#MODEL ?= mobilenet_v1_025_5x3

# Slimmed down MobileNet v3 (not pretrained)
#MODEL ?= mobilenet_v3_micro

# Pretrained MobileNet v3 small
#MODEL ?= mobilenet_v3_small

# Custom models (not pretrained)
#MODEL ?= mvnet_micro
#MODEL ?= mvnet_micro_noatt
MODEL ?= mvnet_micro_memory

#MODEL ?= tmnet_micro
#MODEL ?= yolov6p

# Number of frames in a sliding context window (e.g. for training a model with multiple time-step working memory)
CONTEXT_WINDOW_STEPS ?= 1

DEPLOY ?= no

# Model input and output dimensions
INPUT_WIDTH	= 80
INPUT_HEIGHT = 48
OUTPUT_WIDTH = 20
OUTPUT_HEIGHT = 12

# PyTorch uses channels-first but tflite uses channels-last. By default this will cause onnx_tf to put a lot of TRANSPOSE
# operators into the graph, which makes the model slower. Setting this to channels-last makes convert.py try to convert
# the model to channels-last. See https://github.com/PINTO0309/onnx2tf
#MEMORY_FORMAT = CHANNELS_FIRST
MEMORY_FORMAT = CHANNELS_LAST

# This teacher/annotator can only do round/elliptical objects, e.g. sport balls and oranges
TEACHER_MODEL = yolov5

# This teacher/annotator can handle any COCO object type and uses YOLOv5 for image segmentation, but the edges of the masks aren't perfect
#TEACHER_MODEL = yolov5_seg

# This teacher/annotator can handle any COCO object type and uses YOLOv5 for image segmentation, but the aspect ratio of the output masks from this model is incorrect for some reason
#TEACHER_MODEL = yolov8_seg

# Batch size used when annotating your recorded data, you might need to lower this if your GPU runs out of memory
TEACHER_BATCH_SIZE ?= 32

# Hyperparams for the pre-training on MS COCO images
# - YOLOv6 trains with total batch size 64-256 for 500k-2M optimization steps (400 epochs) on COCO dataset (328k images, 80 classes)
# - 10k steps seems fine for this little model
COCO_EPOCHS ?= 100
COCO_BATCH_SIZE ?= 64
COCO_ACCUMULATION_STEPS ?= 1
COCO_LEARNING_RATE ?= 1e-4
COCO_MODEL_PATH ?= $(EXPERIMENT)/coco.pth
COCO_MAX_SAMPLE_COUNT ?= 25000

# Hyperparams for the fine-tuning on recorded data
# - 3-5k optimization steps seems to be fine for this little model
REC_EPOCHS ?= 300
REC_BATCH_SIZE ?= 64
REC_ACCUMULATION_STEPS ?= 1
REC_LEARNING_RATE ?= 5e-4

PARALLEL ?= $(shell grep -c ^processor /proc/cpuinfo)
GPU_COUNT ?= $(shell if [ "${CUDA_VISIBLE_DEVICES}" != "" ]; then echo "${CUDA_VISIBLE_DEVICES}" | tr "," " "| wc -w; else nvidia-smi --query-gpu=name --format=csv,noheader | grep -v -e '^$$' | wc -l; fi)

LOGGING_DIR ?= experiments/$(shell git rev-parse --abbrev-ref HEAD)
LOGGING_PATH := $(shell echo "$(LOGGING_DIR)/`date +'%Y%m%d-%H%M%S'`")
EXPERIMENT ?= $(LOGGING_PATH)

#ACCELERATE_ARGS := --multi_gpu --num_machines 1 --num_processes $(GPU_COUNT) --mixed_precision fp16
ACCELERATE_ARGS := --mixed_precision fp16

# Download training samples from MS COCO dataset
dataset/coco/train: src/coco.py src/image.py
	mkdir -p dataset/coco/train
	rm -f dataset/coco/train/*
	PYTHONPATH=. python src/coco.py -c "$(PRIMARY_CLASSES)" -e "$(SECONDARY_CLASSES)" -t dataset/coco/train \
		--input-width $(INPUT_WIDTH) --input-height $(INPUT_HEIGHT) --max-sample-count $(COCO_MAX_SAMPLE_COUNT)
	touch dataset/coco/train

# Extract 2 frames per second from a video
dataset/recorded/frames/%_000001.jpg: dataset/recorded/videos/%.mp4
	mkdir -p $(dir $@)
	ffmpeg -i $< -r 4 $(dir $@)/$(basename $(notdir $<))_%06d.jpg

# Extract frames from all videos
dataset/recorded/frames: $(addprefix dataset/recorded/frames/,$(addsuffix _000001.jpg,$(basename $(notdir $(shell find dataset/recorded/videos -type f)))))
	touch $(dir $@)

# Annotate all images and frames
dataset/recorded/train: src/annotate_$(TEACHER_MODEL).py src/image.py dataset/recorded/frames $(shell find dataset/recorded/images -type f)
	mkdir -p dataset/recorded/annotated
	mkdir -p dataset/recorded/train

	PYTHONPATH=. python src/annotate_$(TEACHER_MODEL).py -c "$(CLASSES)" \
		-i dataset/recorded/images -a dataset/recorded/annotated -t dataset/recorded/train \
		--input-width $(INPUT_WIDTH) --input-height $(INPUT_HEIGHT) --batch-size $(TEACHER_BATCH_SIZE)

	PYTHONPATH=. python src/annotate_$(TEACHER_MODEL).py -c "$(CLASSES)" \
		-i dataset/recorded/frames -a dataset/recorded/annotated -t dataset/recorded/train \
		--input-width $(INPUT_WIDTH) --input-height $(INPUT_HEIGHT) --batch-size $(TEACHER_BATCH_SIZE)

	touch dataset/recorded/train

dataset: dataset/coco/train dataset/recorded/train
#dataset: dataset/recorded/train

$(EXPERIMENT)/coco.pth:
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) --gradient_accumulation_steps $(COCO_ACCUMULATION_STEPS) \
		src/train.py -m $(MODEL) -t dataset/coco/train -o $@ -p $(PARALLEL) --epochs $(COCO_EPOCHS) --batch-size $(COCO_BATCH_SIZE) \
		--learning-rate $(COCO_LEARNING_RATE) --accumulation-steps $(COCO_ACCUMULATION_STEPS)
#--unknown-mask

$(EXPERIMENT)/recorded.pth:# $(COCO_MODEL_PATH)
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) --gradient_accumulation_steps $(REC_ACCUMULATION_STEPS) \
		src/train.py -m $(MODEL) -p $(PARALLEL) -t dataset/recorded/train -o $@ --epochs $(REC_EPOCHS) --batch-size $(REC_BATCH_SIZE) \
		--learning-rate $(REC_LEARNING_RATE) --accumulation-steps $(REC_ACCUMULATION_STEPS) --context-window-steps $(CONTEXT_WINDOW_STEPS)
		#--resume $<
#--unknown-mask

$(EXPERIMENT)/recorded.tflite: $(EXPERIMENT)/recorded.pth
	PYTHONPATH=. python src/convert.py -m $< -d dataset/recorded/train $(shell if [ "$(MEMORY_FORMAT)" = "CHANNELS_LAST" ]; then echo "--channels-last"; fi)

$(EXPERIMENT)/recorded.tmdl: $(EXPERIMENT)/recorded.tflite
	python ../../TinyMaix/tools/tflite2tmdl.py $< $@ int8 0 $(INPUT_HEIGHT),$(INPUT_WIDTH),3 $(OUTPUT_HEIGHT),$(OUTPUT_WIDTH),1

output/mbot-vision-model.cpp: $(EXPERIMENT)/recorded.$(INFERENCE_EXTENSION)
	mkdir -p output

	cp $< output/mbot-vision-model
	cd output && xxd -i mbot-vision-model $(notdir $@)
	rm -f output/mbot-vision-model

	sed -i 's/unsigned char/#include "mbot-vision-model.h"\n\nalignas\(8\) const unsigned char/g' $@
	sed -i 's/unsigned int/const unsigned int/g' $@

	cp src/cpp/mbot-vision-model.h $(dir $@)
	sed -i 's/<INPUT_WIDTH>/$(INPUT_WIDTH)/g' $(dir $@)/mbot-vision-model.h
	sed -i 's/<INPUT_HEIGHT>/$(INPUT_HEIGHT)/g' $(dir $@)/mbot-vision-model.h
	sed -i 's/<OUTPUT_WIDTH>/$(OUTPUT_WIDTH)/g' $(dir $@)/mbot-vision-model.h
	sed -i 's/<OUTPUT_HEIGHT>/$(OUTPUT_HEIGHT)/g' $(dir $@)/mbot-vision-model.h
	sed -i 's/<INFERENCE_ENGINE>/$(INFERENCE_ENGINE)/g' $(dir $@)/mbot-vision-model.h
	sed -i 's/<MEMORY_FORMAT>/$(MEMORY_FORMAT)/g' $(dir $@)/mbot-vision-model.h

train: output/mbot-vision-model.cpp

validate: $(EXPERIMENT)/recorded.pth $(EXPERIMENT)/recorded.tflite
	PYTHONPATH=. python src/validate.py -d dataset/recorded/train \
		-m $(EXPERIMENT)/recorded.pth -t $(EXPERIMENT)/recorded.tflite \
		$(shell if [ "$(MEMORY_FORMAT)" = "CHANNELS_LAST" ]; then echo "--channels-last"; fi)

# Print a model summary
summary:
	CUDA_VISIBLE_DEVICES= PYTHONPATH=. python src/summary.py -m $(MODEL) $(shell if [ "$(DEPLOY)" != "no" ]; then echo "-d"; fi)

summary-tflite:
	CUDA_VISIBLE_DEVICES= PYTHONPATH=. python src/summary.py -m $(EXPERIMENT)/recorded.tflite

# Start tensorboard in the background
tensorboard:
	tensorboard --bind_all --samples_per_plugin images=500 --logdir experiments &

test:
	./test/run_tests.sh

all: dataset

.PHONY: dataset all test
