#ifndef _MV_WIRING_H_
#define _MV_WIRING_H_

#include <driver/ledc.h>

// Number of milliseconds between saved training data images
#define MV_SAVE_IMAGE_INTERVAL   1000



// See pinout at https://github.com/Freenove/Freenove_ESP32_S3_WROOM_Board
#ifdef BOARD_ESP32S3CAM

// Joystick comms with MBot (used if you program the mBot using Scratch / Blocks)
#define MV_PWMX_CHAN    LEDC_CHANNEL_4
#define MV_PWMX_PIN     14                  // Maps to port 6 and pin 64 (A10) on Auriga
#define MV_PWMY_CHAN    LEDC_CHANNEL_5
#define MV_PWMY_PIN     3                   // Maps to port 6 and pin 69 (A15) on Auriga

// Flash / floodlights
#define MV_LED_PIN      2
#define MV_FLASH_PIN    21
#define MV_FLASH_CHAN   LEDC_CHANNEL_0
#define MV_FLASH_MAX    255

// I2C bus for display
#define MV_SDA_PIN      41
#define MV_SCL_PIN      42

// Camera (also seems to use LEDC_CHANNEL_2 and LEDC_CHANNEL_3?)
#define MV_CAM_CHAN     LEDC_CHANNEL_1
#define MV_CAM_TIMER    LEDC_TIMER_1

// Enable logging images to SD card
#define MV_SAVE_IMAGE_ENABLED   true

// If the SDcard pins are used for other things too
#define BOARD_OVERLOADED_SDCARD_PINS    false

// Overloaded pins used for SD card
#define MV_HS2_DATA2     -1
#define MV_HS2_DATA3     -1
#define MV_HS2_CMD       38
#define MV_HS2_CLK       39
#define MV_HS2_DATA0     40
#define MV_HS2_DATA1     -1
#endif



// See pinout at https://randomnerdtutorials.com/esp32-cam-ai-thinker-pinout/
#ifdef BOARD_ESP32CAM

// Joystick comms with MBot (used if you program the mBot using Scratch / Blocks)
#define MV_PWMX_CHAN    LEDC_CHANNEL_4
#define MV_PWMX_PIN     2                   // Maps to port 6 and pin 64 (A10) on Auriga
#define MV_PWMY_CHAN    LEDC_CHANNEL_5
#define MV_PWMY_PIN     14                  // Maps to port 6 and pin 69 (A15) on Auriga

// Flash / floodlights
#define MV_LED_PIN      33
#define MV_FLASH_PIN    12
#define MV_FLASH_CHAN   LEDC_CHANNEL_0
#define MV_FLASH_MAX    255

// I2C bus for display
#define MV_SDA_PIN      13
#define MV_SCL_PIN      15

// Camera (also seems to use LEDC_CHANNEL_2 and LEDC_CHANNEL_3?)
#define MV_CAM_CHAN     LEDC_CHANNEL_1
#define MV_CAM_TIMER    LEDC_TIMER_1

// Enable logging images to SDcard. If you enable this you must also physically disconnect all other 
// peripherals and pins, like the flash and mBot UART, since the same pins are used for the SDcard 
// on the ESP32-CAM board.
#define MV_SAVE_IMAGE_ENABLED   false

// If the SDcard pins are used for other things too
#define BOARD_OVERLOADED_SDCARD_PINS    true

// Overloaded pins used for SD card
#define MV_HS2_DATA2     12
#define MV_HS2_DATA3     13
#define MV_HS2_CMD       15
#define MV_HS2_CLK       14
#define MV_HS2_DATA0     2
#define MV_HS2_DATA1     4
#endif



#endif // _MV_WIRING_H_
