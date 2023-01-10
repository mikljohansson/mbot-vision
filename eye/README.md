# Wiring

## FDTI programmer

The ESP32 unfortunately doesn't have a RST pin, so in order to get the programmer to automatically put the 
ESP32 into programming/flashing mode you need to solder a wire onto the inner connector of the RST button 
and connect that to the RTS pin on the FTDI programmer. Doing this will allow you to upload new firmware
without having to manually hold down the I00 and press the RST button.

|Programmer|ESP32 CAM|
|----------|---------|
|5V        |5V       |
|GND       |GND      |
|TXD       |U0R      |
|RXD       |U0T      |
|DTR       |I00      |
|RTS       |RST      |

# Setup

You need to prepare the ESP32 specific tflite-micro distribution a bit in order to make it usable from PlatformIO
and get the ESP32 optimized kernels. These kernels can make a 3x factor change in inference performance on the 
ESP32, or even more on an ESP32-S3.

```
cd ..
git clone --recurse-submodules https://github.com/espressif/tflite-micro-esp-examples.git
cd tflite-micro-esp-examples

# May need to update the tflite-micro base to get compatiblity with latest TFLite ops
scripts/sync_from_tflite_micro.sh

# If you're not targeting an new ESP32-S3 MCU then remove all the optimized kernels for that architecture
find components/esp-nn -name '*esp32s3*' -exec rm -f {} ';'

# Remove all the default tfmicro kernels in favor of the ESP32 optimized kernels
# https://github.com/espressif/tflite-micro-esp-examples#esp-nn-integration
for f in components/tflite-lib/tensorflow/lite/micro/kernels/esp_nn/*.cc; do rm components/tflite-lib/tensorflow/lite/micro/kernels/`basename $f`; done
```

# Setting up /dev aliases

https://medium.com/@darshankt/setting-up-the-udev-rules-for-connecting-multiple-external-devices-through-usb-in-linux-28c110cf9251
```
sudo vi /etc/udev/rules.d/99-usb-aliases.rules

SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", SYMLINK+="ftdi"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="auriga"

# sudo rm -f /usr/lib/udev/rules.d/90-brltty-device.rules

sudo systemctl mask brltty-udev
sudo systemctl mask brltty

sudo systemctl stop brltty-udev
sudo systemctl stop brltty

sudo udevadm control --reload-rules && sudo udevadm trigger
```

# Common issues

## Invalid/corrupt model file

```
Initializing tflite

Guru Meditation Error: Core  1 panic'ed (LoadProhibited). Exception was unhandled.
```

This probably means that the tflite file that is encoded into the model buffer in `mbot-vision-model.cpp` is not a valid for some reason.
