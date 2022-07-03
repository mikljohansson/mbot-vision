PARALLEL ?= $(shell grep -c ^processor /proc/cpuinfo)
GPU_COUNT ?= $(shell if [ "${CUDA_VISIBLE_DEVICES}" != "" ]; then echo "${CUDA_VISIBLE_DEVICES}" | tr "," " "| wc -w; else nvidia-smi --query-gpu=name --format=csv,noheader | grep -v -e '^$$' | wc -l; fi)

LOGGING_DIR ?= experiments/$(shell git rev-parse --abbrev-ref HEAD)
LOGGING_PATH := $(shell echo "$(LOGGING_DIR)/`date +'%Y%m%d-%H%M%S'`")
EXPERIMENT ?= $(LOGGING_PATH)

ACCELERATE_ARGS := --multi_gpu --num_machines 1 --num_processes $(GPU_COUNT) --mixed_precision no

COCO_EPOCHS ?= 100
COCO_BATCH_SIZE ?= 64

REC_EPOCHS ?= 100
REC_BATCH_SIZE ?= 16

dataset/coco/train: src/coco.py src/image.py
	mkdir -p dataset/coco/train
	touch dataset/coco/train
	rm -f dataset/coco/train/*
	PYTHONPATH=. python src/coco.py -c "sports ball" -e "orange,apple" -t dataset/coco/train

dataset/recorded/train: src/annotate.py src/image.py
	mkdir -p dataset/recorded/annotated
	mkdir -p dataset/recorded/train
	touch dataset/recorded/train
	PYTHONPATH=. python src/annotate.py -i dataset/recorded/images -a dataset/recorded/annotated -t dataset/recorded/train

dataset: dataset/coco/train dataset/recorded/train

$(EXPERIMENT)/coco.pth: dataset/coco/train
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) src/train.py -p $(PARALLEL) \
		-t dataset/coco/train -o $@ --epochs $(COCO_EPOCHS) --batch-size $(COCO_BATCH_SIZE)

$(EXPERIMENT)/recorded.pth: $(EXPERIMENT)/coco.pth dataset/recorded/train
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) src/train.py -p $(PARALLEL) \
		-t dataset/recorded/train -m $< -o $@ --epochs $(REC_EPOCHS) --batch-size $(REC_BATCH_SIZE)

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
