PARALLEL ?= $(shell grep -c ^processor /proc/cpuinfo)

LOGGING_DIR ?= experiments/$(shell git rev-parse --abbrev-ref HEAD)
LOGGING_PATH := $(shell echo "$(LOGGING_DIR)/`date +'%Y%m%d-%H%M%S'`")
EXPERIMENT ?= $(LOGGING_PATH)

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
	PYTHONPATH=. accelerate launch --mixed_precision no src/train.py -p $(PARALLEL) \
		-t dataset/coco/train -o $@

$(EXPERIMENT)/recorded.pth: $(EXPERIMENT)/coco.pth dataset/recorded/train
	PYTHONPATH=. accelerate launch --mixed_precision no src/train.py -p $(PARALLEL) \
		-t dataset/recorded/train -m $< -o $@

$(EXPERIMENT)/recorded.tflite: $(EXPERIMENT)/recorded.pth
	PYTHONPATH=. python src/convert.py -m $< -d dataset/recorded/train

train: $(EXPERIMENT)/recorded.tflite

# Start tensorboard in the background
tensorboard:
	tensorboard --bind_all --samples_per_plugin images=500 --logdir experiments &

all: dataset

.PHONY: dataset all
