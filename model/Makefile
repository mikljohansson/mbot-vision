PARALLEL ?= $(shell grep -c ^processor /proc/cpuinfo)

LOGGING_DIR ?= experiments/$(shell git rev-parse --abbrev-ref HEAD)
LOGGING_PATH = $(shell echo "$(LOGGING_DIR)/`date +'%Y%m%d-%H%M%S'`")
EXPERIMENT ?= $(LOGGING_PATH)

dataset/train:
	mkdir -p dataset/annotated
	mkdir -p dataset/train
	python src/annotate.py -i dataset/images -a dataset/annotated -t dataset/train

dataset: dataset/train

train:
	PYTHONPATH=. accelerate launch --mixed_precision no src/train.py -t dataset/train -p $(PARALLEL) \
		--output_dir "$(LOGGING_PATH)"

# Start tensorboard in the background
tensorboard:
	tensorboard --bind_all --samples_per_plugin images=500 --logdir experiments &

all: dataset

.PHONY: dataset all
