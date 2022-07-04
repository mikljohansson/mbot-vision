#ifndef _MV_WIRING_H_
#define _MV_WIRING_H_

#include <driver/ledc.h>

// https://randomnerdtutorials.com/esp32-cam-ai-thinker-pinout/

// Camera (also seems to use LEDC_CHANNEL_2 and LEDC_CHANNEL_3?)
#define MV_CAM_CHAN     LEDC_CHANNEL_1
#define MV_CAM_TIMER    LEDC_TIMER_1

// Flash / floodlights
#define MV_LED_PIN      33
#define MV_FLASH_PIN    12
#define MV_FLASH_CHAN   LEDC_CHANNEL_0
#define MV_FLASH_MAX    255

// I2C bus for display
#define MV_SDA_PIN      13
#define MV_SCL_PIN      15

// UART comms with MBot
#define MV_UTX_PIN      14
#define MV_URX_PIN      2

// Joystick comms with MBot
#define MV_PWMX_CHAN    LEDC_CHANNEL_4
#define MV_PWMX_PIN     MV_URX_PIN          // Maps to port 6 and pin 64 (A10) on Auriga
#define MV_PWMY_CHAN    LEDC_CHANNEL_5
#define MV_PWMY_PIN     MV_UTX_PIN          // Maps to port 6 and pin 69 (A15) on Auriga

// Overloaded pins used for SD card
#define MV_HS2_DATA2     12
#define MV_HS2_DATA3     13
#define MV_HS2_CMD       15
#define MV_HS2_CLK       14
#define MV_HS2_DATA0     2
#define MV_HS2_DATA1     4

#endif
