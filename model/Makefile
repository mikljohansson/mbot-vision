PRIMARY_CLASSES = sports ball
SECONDARY_CLASSES = apple,orange
CLASSES = $(PRIMARY_CLASSES),$(SECONDARY_CLASSES)

PARALLEL ?= $(shell grep -c ^processor /proc/cpuinfo)
GPU_COUNT ?= $(shell if [ "${CUDA_VISIBLE_DEVICES}" != "" ]; then echo "${CUDA_VISIBLE_DEVICES}" | tr "," " "| wc -w; else nvidia-smi --query-gpu=name --format=csv,noheader | grep -v -e '^$$' | wc -l; fi)

LOGGING_DIR ?= experiments/$(shell git rev-parse --abbrev-ref HEAD)
LOGGING_PATH := $(shell echo "$(LOGGING_DIR)/`date +'%Y%m%d-%H%M%S'`")
EXPERIMENT ?= $(LOGGING_PATH)

ACCELERATE_ARGS := --multi_gpu --num_machines 1 --num_processes $(GPU_COUNT) --mixed_precision fp16

# YOLOv6 trains with total batch size 64-256 for 500k-2M optimization steps (400 epochs) on COCO dataset (328k images, 80 classes)
COCO_EPOCHS ?= 200
COCO_BATCH_SIZE ?= 64
COCO_ACCUMULATION_STEPS ?= 1
COCO_MODEL_PATH ?= $(EXPERIMENT)/coco.pth

REC_EPOCHS ?= 500
REC_BATCH_SIZE ?= 64
REC_ACCUMULATION_STEPS ?= 1

# Need to use a really low learning rate to avoid the "dying ReLU problem". It manifests as the loss sharply flat-lining and the network stops learning
COCO_LEARNING_RATE ?= 5e-5
REC_LEARNING_RATE ?= 1e-4

dataset/coco/train: src/coco.py src/image.py
	mkdir -p dataset/coco/train
	touch dataset/coco/train
	rm -f dataset/coco/train/*
	PYTHONPATH=. python src/coco.py -c "$(PRIMARY_CLASSES)" -e "$(SECONDARY_CLASSES)" -t dataset/coco/train

# Extract 2 frames per second from a video
dataset/recorded/frames/%_000001.jpg: dataset/recorded/videos/%.mp4
	mkdir -p $(dir $@)
	ffmpeg -i $< -r 4 $(dir $@)/$(basename $(notdir $<))_%06d.jpg

# Extract frames from all videos
dataset/recorded/frames: $(addprefix dataset/recorded/frames/,$(addsuffix _000001.jpg,$(basename $(notdir $(shell find dataset/recorded/videos -type f)))))
	touch $(dir $@)

# Annotate all images and frames
dataset/recorded/train: src/annotate.py src/image.py dataset/recorded/frames $(shell find dataset/recorded/images -type f)
	mkdir -p dataset/recorded/annotated
	mkdir -p dataset/recorded/train

	PYTHONPATH=. python src/annotate.py -c "$(CLASSES)" \
		-i dataset/recorded/images -a dataset/recorded/annotated -t dataset/recorded/train

	PYTHONPATH=. python src/annotate.py -c "$(CLASSES)" \
		-i dataset/recorded/frames -a dataset/recorded/annotated -t dataset/recorded/train

	touch dataset/recorded/train

dataset: dataset/coco/train dataset/recorded/train

$(EXPERIMENT)/coco.pth:
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) --gradient_accumulation_steps $(COCO_ACCUMULATION_STEPS) \
		src/train.py -t dataset/coco/train -o $@ -p $(PARALLEL) --epochs $(COCO_EPOCHS) --batch-size $(COCO_BATCH_SIZE) \
		--learning-rate $(COCO_LEARNING_RATE) --accumulation-steps $(COCO_ACCUMULATION_STEPS) --unknown-mask

$(EXPERIMENT)/recorded.pth:
#$(COCO_MODEL_PATH)
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) --gradient_accumulation_steps $(REC_ACCUMULATION_STEPS) \
		src/train.py -p $(PARALLEL) -t dataset/recorded/train -o $@ --epochs $(REC_EPOCHS) --batch-size $(REC_BATCH_SIZE) \
		--learning-rate $(REC_LEARNING_RATE) --accumulation-steps $(REC_ACCUMULATION_STEPS) --unknown-mask
#-m $<

$(EXPERIMENT)/recorded.tflite: $(EXPERIMENT)/recorded.pth
	PYTHONPATH=. python src/convert.py -m $< -d dataset/recorded/train

train: $(EXPERIMENT)/recorded.tflite

validate: $(EXPERIMENT)/recorded.pth $(EXPERIMENT)/recorded.tflite
	PYTHONPATH=. python src/validate.py -m $(EXPERIMENT)/recorded.pth -t $(EXPERIMENT)/recorded.tflite -d dataset/recorded/train

# Print a model summary
summary:
	CUDA_VISIBLE_DEVICES= PYTHONPATH=. python src/summary.py

# Start tensorboard in the background
tensorboard:
	tensorboard --bind_all --samples_per_plugin images=500 --logdir experiments &

all: dataset

.PHONY: dataset all
