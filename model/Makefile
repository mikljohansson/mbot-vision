WEIGHTS = efficientnet-b0_weights_tf_dim_ordering_tf_kernels.h5

download:
	cd checkpoints
	wget https://github.com/Callidior/keras-applications/releases/download/efficientnet/$(WEIGHTS)

tflite:
	python3 build.py checkpoints/$(WEIGHTS) checkpoints/efficientnet.tflite dataset/imagen/imagen/

flash:
	esptool.py --port /dev/ftdi --baud 460800 write_flash -z --flash_mode dio --flash_size detect 0x210000 checkpoints/efficientnet.tflite

