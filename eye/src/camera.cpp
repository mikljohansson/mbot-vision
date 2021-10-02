#include "camera.h"

// https://github.com/geeksville/Micro-RTSP/blob/master/src/OV2640.cpp
camera_config_t mv_camera_aithinker_config {
    .pin_pwdn = 32,
    .pin_reset = -1,

    .pin_xclk = 0,

    .pin_sscb_sda = 26,
    .pin_sscb_scl = 27,

    // Note: LED GPIO is apparently 4 not sure where that goes
    // per https://github.com/donny681/ESP32_CAMERA_QR/blob/e4ef44549876457cd841f33a0892c82a71f35358/main/led.c
    .pin_d7 = 35,
    .pin_d6 = 34,
    .pin_d5 = 39,
    .pin_d4 = 36,
    .pin_d3 = 21,
    .pin_d2 = 19,
    .pin_d1 = 18,
    .pin_d0 = 5,
    .pin_vsync = 25,
    .pin_href = 23,
    .pin_pclk = 22,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    // .frame_size = FRAMESIZE_UXGA, // needs 234K of framebuffer space
    // .frame_size = FRAMESIZE_SXGA, // needs 160K for framebuffer
    // .frame_size = FRAMESIZE_XGA, // needs 96K or even smaller FRAMESIZE_SVGA - can work if using only 1 fb
    //.frame_size = FRAMESIZE_QVGA,
    .frame_size = FRAMESIZE_VGA,
    .jpeg_quality = 20, //0-63 lower numbers are higher quality
    .fb_count = 2       // if more than one i2s runs in continous mode.  Use only with jpeg
};
