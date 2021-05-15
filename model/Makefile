all:
	mkdir -p checkpoints
	cd checkpoints
	wget https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_uint8.tgz
	tar -xzf *.tgz
	ln -s v3-small_224_1.0_uint8/v3-small_224_1.0_uint8.tflite mv_mobilenet_model.tflite
