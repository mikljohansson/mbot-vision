Get tfmicro library using this method
https://github.com/atomic14/platformio-tensorflow-lite

https://github.com/atomic14/tensorflow-lite-esp32/tree/master/firmware

Fix std::to_string problems
https://community.platformio.org/t/to-string-is-not-a-member-of-std/20681/8

Use https://github.com/google/automl/tree/master/efficientdet

Convert tflite to h/cpp
https://www.tensorflow.org/lite/microcontrollers/build_convert

Install CUDA and TensorRT
https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_101

Run model from flash
https://en.bbs.sipeed.com/t/topic/1811


# Seting up /dev aliases

https://medium.com/@darshankt/setting-up-the-udev-rules-for-connecting-multiple-external-devices-through-usb-in-linux-28c110cf9251
```
vi /etc/udev/rules.d/99-usb-aliases.rules

SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", SYMLINK+="ftdi"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="auriga"
```