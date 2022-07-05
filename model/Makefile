CLASSES = sports ball,apple,orange

PARALLEL ?= $(shell grep -c ^processor /proc/cpuinfo)
GPU_COUNT ?= $(shell if [ "${CUDA_VISIBLE_DEVICES}" != "" ]; then echo "${CUDA_VISIBLE_DEVICES}" | tr "," " "| wc -w; else nvidia-smi --query-gpu=name --format=csv,noheader | grep -v -e '^$$' | wc -l; fi)

LOGGING_DIR ?= experiments/$(shell git rev-parse --abbrev-ref HEAD)
LOGGING_PATH := $(shell echo "$(LOGGING_DIR)/`date +'%Y%m%d-%H%M%S'`")
EXPERIMENT ?= $(LOGGING_PATH)

ACCELERATE_ARGS := --multi_gpu --num_machines 1 --num_processes $(GPU_COUNT) --mixed_precision no

# YOLOv6 trains with total batch size 64 for 2M optimization steps
COCO_EPOCHS ?= 400
COCO_BATCH_SIZE ?= 64
COCO_LEARNING_RATE ?= 1e-4

REC_EPOCHS ?= 1000
REC_BATCH_SIZE ?= 4
REC_LEARNING_RATE ?= 5e-5

dataset/coco/train: src/coco.py src/image.py
	mkdir -p dataset/coco/train
	touch dataset/coco/train
	rm -f dataset/coco/train/*
	PYTHONPATH=. python src/coco.py -c "sports ball" -e "orange,apple" -t dataset/coco/train

# Extract 2 frames per second from a video
dataset/recorded/frames/%_000001.jpg: dataset/recorded/videos/%.mp4
	mkdir -p $(dir $@)
	ffmpeg -i $< -r 2 $(basename $@)_%06d.jpg

# Extract frames from all videos
dataset/recorded/frames: $(addprefix dataset/recorded/frames/,$(addsuffix _000001.jpg,$(basename $(notdir $(shell find dataset/recorded/videos -type f)))))
	touch $(dir $@)

# Annotate all images and frames
dataset/recorded/train: src/annotate.py src/image.py $(shell find dataset/recorded/images -type f) $(shell find dataset/recorded/frames -type f)
	mkdir -p dataset/recorded/annotated
	mkdir -p dataset/recorded/train

	PYTHONPATH=. python src/annotate.py -c "$(CLASSES)" \
		-i dataset/recorded/images -a dataset/recorded/annotated -t dataset/recorded/train

	PYTHONPATH=. python src/annotate.py -c "$(CLASSES)" \
		-i dataset/recorded/frames -a dataset/recorded/annotated -t dataset/recorded/train

	touch dataset/recorded/train

dataset: dataset/coco/train dataset/recorded/train

$(EXPERIMENT)/coco.pth: dataset/coco/train
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) src/train.py -p $(PARALLEL) \
		-t dataset/coco/train -o $@ --epochs $(COCO_EPOCHS) --batch-size $(COCO_BATCH_SIZE) \
		--learning-rate $(COCO_LEARNING_RATE)

$(EXPERIMENT)/recorded.pth: $(EXPERIMENT)/coco.pth dataset/recorded/train
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) src/train.py -p $(PARALLEL) \
		-t dataset/recorded/train -m $< -o $@ --epochs $(REC_EPOCHS) --batch-size $(REC_BATCH_SIZE) \
		--learning-rate $(REC_LEARNING_RATE)

$(EXPERIMENT)/recorded.tflite: $(EXPERIMENT)/recorded.pth
	PYTHONPATH=. python src/convert.py -m $< -d dataset/recorded/train

train: $(EXPERIMENT)/recorded.tflite

# Print a model summary
summary:
	CUDA_VISIBLE_DEVICES= PYTHONPATH=. python src/summary.py

# Start tensorboard in the background
tensorboard:
	tensorboard --bind_all --samples_per_plugin images=500 --logdir experiments &

all: dataset

.PHONY: dataset all
