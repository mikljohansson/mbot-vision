PRIMARY_CLASSES = sports ball
SECONDARY_CLASSES = apple,orange
CLASSES = $(PRIMARY_CLASSES),$(SECONDARY_CLASSES)

PARALLEL ?= $(shell grep -c ^processor /proc/cpuinfo)
GPU_COUNT ?= $(shell if [ "${CUDA_VISIBLE_DEVICES}" != "" ]; then echo "${CUDA_VISIBLE_DEVICES}" | tr "," " "| wc -w; else nvidia-smi --query-gpu=name --format=csv,noheader | grep -v -e '^$$' | wc -l; fi)

LOGGING_DIR ?= experiments/$(shell git rev-parse --abbrev-ref HEAD)
LOGGING_PATH := $(shell echo "$(LOGGING_DIR)/`date +'%Y%m%d-%H%M%S'`")
EXPERIMENT ?= $(LOGGING_PATH)

#ACCELERATE_ARGS := --multi_gpu --num_machines 1 --num_processes $(GPU_COUNT) --mixed_precision fp16
ACCELERATE_ARGS := --mixed_precision fp16

INFERENCE_ENGINE ?= TFMICRO
#INFERENCE_ENGINE ?= TINYMAIX
INFERENCE_EXTENSION = $(shell echo $(INFERENCE_ENGINE) | sed s/TINYMAIX/tmdl/ | sed s/TFMICRO/tflite/)

#MODEL ?= yolov6p
#MODEL ?= mobilenet_v3_micro
#MODEL ?= mvnet_micro
MODEL ?= tmnet_micro
DEPLOY ?= no

# Model input and output dimensions
INPUT_WIDTH	= 80
INPUT_HEIGHT = 60
OUTPUT_WIDTH = 20
OUTPUT_HEIGHT = 15

# YOLOv6 trains with total batch size 64-256 for 500k-2M optimization steps (400 epochs) on COCO dataset (328k images, 80 classes)
COCO_EPOCHS ?= 200
COCO_BATCH_SIZE ?= 64
COCO_ACCUMULATION_STEPS ?= 1
COCO_LEARNING_RATE ?= 1e-4
COCO_MODEL_PATH ?= $(EXPERIMENT)/coco.pth

# 10k optimization steps seems to be fine for this little model
REC_EPOCHS ?= 1
REC_BATCH_SIZE ?= 64
REC_ACCUMULATION_STEPS ?= 1
REC_LEARNING_RATE ?= 1e-4

dataset/coco/train: src/coco.py src/image.py
	mkdir -p dataset/coco/train
	touch dataset/coco/train
	rm -f dataset/coco/train/*
	PYTHONPATH=. python src/coco.py -c "$(PRIMARY_CLASSES)" -e "$(SECONDARY_CLASSES)" -t dataset/coco/train \
		--input-width $(INPUT_WIDTH) --input-height $(INPUT_HEIGHT)

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
		-i dataset/recorded/images -a dataset/recorded/annotated -t dataset/recorded/train \
		--input-width $(INPUT_WIDTH) --input-height $(INPUT_HEIGHT)

	PYTHONPATH=. python src/annotate.py -c "$(CLASSES)" \
		-i dataset/recorded/frames -a dataset/recorded/annotated -t dataset/recorded/train \
		--input-width $(INPUT_WIDTH) --input-height $(INPUT_HEIGHT)

	touch dataset/recorded/train

dataset: dataset/coco/train dataset/recorded/train

$(EXPERIMENT)/coco.pth:
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) --gradient_accumulation_steps $(COCO_ACCUMULATION_STEPS) \
		src/train.py -m $(MODEL) -t dataset/coco/train -o $@ -p $(PARALLEL) --epochs $(COCO_EPOCHS) --batch-size $(COCO_BATCH_SIZE) \
		--learning-rate $(COCO_LEARNING_RATE) --accumulation-steps $(COCO_ACCUMULATION_STEPS) --unknown-mask

$(EXPERIMENT)/recorded.pth:
#$(COCO_MODEL_PATH)
	PYTHONPATH=. accelerate launch $(ACCELERATE_ARGS) --gradient_accumulation_steps $(REC_ACCUMULATION_STEPS) \
		src/train.py -m $(MODEL) -p $(PARALLEL) -t dataset/recorded/train -o $@ --epochs $(REC_EPOCHS) --batch-size $(REC_BATCH_SIZE) \
		--learning-rate $(REC_LEARNING_RATE) --accumulation-steps $(REC_ACCUMULATION_STEPS)
#--unknown-mask
#--resume $<

$(EXPERIMENT)/recorded.tflite: $(EXPERIMENT)/recorded.pth
	PYTHONPATH=. python src/convert.py -m $< -d dataset/recorded/train

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

train: output/mbot-vision-model.cpp

validate: $(EXPERIMENT)/recorded.pth $(EXPERIMENT)/recorded.tflite
	PYTHONPATH=. python src/validate.py -d dataset/recorded/train \
		-m $(EXPERIMENT)/recorded.pth -t $(EXPERIMENT)/recorded.tflite

# Print a model summary
summary:
	CUDA_VISIBLE_DEVICES= PYTHONPATH=. python src/summary.py -m $(MODEL) $(shell if [ "$(DEPLOY)" != "no" ]; then echo "-d"; fi)

summary-tflite:
	CUDA_VISIBLE_DEVICES= PYTHONPATH=. python src/summary.py -m $(EXPERIMENT)/recorded.tflite

# Start tensorboard in the background
tensorboard:
	tensorboard --bind_all --samples_per_plugin images=500 --logdir experiments &

all: dataset

.PHONY: dataset all
